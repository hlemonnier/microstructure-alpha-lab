"""Test recent residual probability correction without replacing a frozen parent."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_morning_adaptation import morning_masks
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_residual_boost import ResidualBoostForecaster, fit_residual_boost, select_residual_rounds
from lob_forge.boundary_tardis_inputs import load_tardis_inputs, select_tardis_variant
from run_boundary_confirmation import noon, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tardis_screen import BASELINES, BLENDS, FITS

ROOT = Path(__file__).resolve().parents[1]
PARENT_MODELS = (*BASELINES, *FITS, *BLENDS)


def run_fold(protocol, output, identity):
    day = protocol["date"]
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"model_fits": len(protocol["models"]), "assessment_scores_sealed": True})
    parent = ROOT / protocol["parent_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    partitions = json.loads((parent / "available_partitions.json").read_text())
    if not all(d < day for d in partitions["past_training_dates"]) or partitions["past_validation_day"] >= day:
        raise ValueError("The residual parent must precede every current-day outcome")
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    sessions = [r for r in manifest["sessions"] if r["session_date"] == day]
    partition = folder / "inputs.json"
    write_json(partition, {"sessions": sessions})
    x, y, clocks = load_tardis_inputs(ROOT, partition, ROOT / protocol["native_manifest"],
                                     ROOT / protocol["bybit_manifest"], ROOT / protocol["spot_manifest"])
    xs, ys, ts, prior = {}, {}, {}, {}
    for symbol in SYMBOLS:
        record = next(r for r in sessions if r["symbol"] == symbol)
        released = pd.read_parquet(ROOT / record["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clocks[symbol, day]]
        np.testing.assert_array_equal(released.label.to_numpy(), y[symbol, day])
        training, validation = morning_masks(clocks[symbol, day], released.future_event_time.to_numpy(), utc_ms(day), window="full_morning")
        masks = {"train": training, "validation": validation, "test": noon(clocks[symbol, day], day)}
        for name, mask in masks.items():
            xs[name, symbol] = x[symbol, day].loc[mask].copy()
            ys[name, symbol], ts[name, symbol] = y[symbol, day][mask], clocks[symbol, day][mask]
        prior[symbol] = np.array([(ys["train", symbol] == k).mean() for k in (-1, 0, 1)])
        np.testing.assert_array_equal(prior[symbol], partitions["morning_priors"][symbol])
        if any(len(ys[name, symbol]) != partitions["partitions"][symbol][field]
               for name, field in (("train", "training_rows"), ("validation", "validation_rows"), ("test", "assessment_rows"))):
            raise ValueError("Residual and parent partitions differ")
    del x, y, clocks
    gc.collect()
    begin = time.monotonic()
    hgb = joblib.load(parent / "past_context_hgb.joblib")
    neural = PooledForecaster.load(parent / "past_context_neural")
    base = {}
    for name in ("train", "validation", "test"):
        for asset, symbol in enumerate(SYMBOLS):
            features = select_tardis_variant(xs[name, symbol], "observations")
            base[name, symbol] = (hgb.predict_proba(features, symbol) + neural.predict_proba(features, asset)) / 2
            if name == "test":
                with np.load(parent / "predictions" / f"{symbol}_past_context_blend_forecast_3600.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(base[name, symbol], saved["probabilities"])
                    np.testing.assert_array_equal(ys[name, symbol], saved["labels"])
                    np.testing.assert_array_equal(ts[name, symbol], saved["decision_times"])
    base_seconds = time.monotonic() - begin
    del hgb, neural
    metadata, predictions = {}, {s: {} for s in SYMBOLS}
    for name, setting in protocol["models"].items():
        representation = setting["representation"]
        training_x = {s: select_tardis_variant(xs["train", s], representation) for s in SYMBOLS}
        validation_x = {s: select_tardis_variant(xs["validation", s], representation) for s in SYMBOLS}
        test_x = {s: select_tardis_variant(xs["test", s], representation) for s in SYMBOLS}
        write_json(output / "progress.json", {"stage": "training", "model": name, "assessment_scores_sealed": True})
        begin = time.monotonic()
        with threadpool_limits(limits=2):
            model = fit_residual_boost(training_x, {s: ys["train", s] for s in SYMBOLS}, {s: base["train", s] for s in SYMBOLS},
                max_depth=setting["max_depth"], class_balanced=setting["class_balanced"])
            selection = select_residual_rounds(model, validation_x, {s: ys["validation", s] for s in SYMBOLS},
                {s: base["validation", s] for s in SYMBOLS}, prior)
        model.save(folder / name)
        write_json(folder / name / "selection.json", selection)
        restored = ResidualBoostForecaster.load(folder / name)
        ready_seconds = time.monotonic() - begin + base_seconds
        if ready_seconds > 290:
            raise ValueError("Residual fitting and cached base inference exceed the causal computation budget")
        for asset, symbol in enumerate(SYMBOLS):
            np.testing.assert_array_equal(restored.priors[asset], prior[symbol])
            p = restored.predict_proba(test_x[symbol], symbol, base["test", symbol])
            np.testing.assert_array_equal(p, model.predict_proba(test_x[symbol], symbol, base["test", symbol]))
            np.testing.assert_array_equal(restored.predict_proba(test_x[symbol], symbol, base["test", symbol], rounds=0), base["test", symbol])
            predictions[symbol][name] = p
        metadata[name] = {"setting": setting, "selected_rounds": restored.selected_rounds, "columns": restored.columns,
            "training_rows": {s: len(ys["train", s]) for s in SYMBOLS}, "checkpoint_parity": True,
            "zero_correction_parent_exact": True, "cached_base_prediction_seconds": base_seconds,
            "fit_selection_checkpoint_and_base_seconds": ready_seconds}
        print(f"residual_fit={name} seconds={ready_seconds:.1f} selected_rounds={restored.selected_rounds}", flush=True)
        del model, restored
        gc.collect()
    for symbol in SYMBOLS:
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], ts["test", symbol])
            initial = saved["labels_cumulative"][0].copy()
        for name, p in predictions[symbol].items():
            for policy in protocol["policies"]:
                pi = prior[symbol] if policy == "morning_full" else forecast_priors(p, ts["test", symbol], initial, half_life_seconds=3600)
                path = folder / "predictions" / f"{symbol}_{name}_{policy}.npz"
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, probabilities=p, labels=ys["test", symbol], decision_times=ts["test", symbol], decision_priors=pi)
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["probabilities"], p)
                    np.testing.assert_array_equal(saved["decision_priors"], pi)
        for name in PARENT_MODELS:
            for policy in protocol["policies"]:
                with np.load(parent / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["labels"], ys["test", symbol])
                    np.testing.assert_array_equal(saved["decision_times"], ts["test", symbol])
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "model_fits": len(protocol["models"]), "parent_checkpoint_and_row_parity": True,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(protocol, output, identity):
    day = protocol["date"]
    folder, parent = output / "dates" / day, ROOT / protocol["parent_run"] / "dates" / day
    check_completed(folder, identity)
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    records = []
    for symbol in SYMBOLS:
        for model in [*PARENT_MODELS, *protocol["models"]]:
            source = parent if model in PARENT_MODELS else folder
            for policy in protocol["policies"]:
                with np.load(source / "predictions" / f"{symbol}_{model}_{policy}.npz", allow_pickle=False) as values:
                    metrics = classification_metrics(SimpleNamespace(priors=values["decision_priors"]), values["probabilities"], values["labels"])
                records.append({"date": day, "symbol": symbol, "model": model, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete residual development family")
    board = []
    for model, policy in sorted({(r["model"], r["policy"]) for r in records}):
        selected = [r for r in records if (r["model"], r["policy"]) == (model, policy)]
        board.append({"model": model, "policy": policy, "means": {k: float(np.mean([r["metrics"][k] for r in selected]))
            for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {r["symbol"]: r["metrics"]["balanced_accuracy"] for r in selected}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(protocol["models"]),
        "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_date" or protocol["policies"] != ["morning_full", "forecast_3600"]:
        raise ValueError("Fixed residual development scope and common decision policies required")
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Frozen residual source changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "pyarrow")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the original residual experiment after source changes")
    write_json(frozen, identity)
    run_fold(protocol, output, identity)
    write_json(output / "summary.json", reveal(protocol, output, identity))
    print(f"residual_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the fixed residual-correction development experiment.")
