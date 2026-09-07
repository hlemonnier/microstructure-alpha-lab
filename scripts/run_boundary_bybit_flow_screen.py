"""Test native queue flow against matched BBO-flow and existing predictor controls."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_bybit_flow_inputs import FLOW_VARIANTS, load_flow_inputs, select_flow_variant
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "observations_blend", "spot100_tree", "top1_100_tree_old_neural",
             "combined_hgb", "combined_neural", "observations_neural", "combined_blend")
MODEL_SPECS = {f"{variant}_{kind}": (variant, kind) for variant in FLOW_VARIANTS for kind in ("hgb", "neural")}
BLENDS = {**{f"{v}_blend": (f"{v}_hgb", f"{v}_neural") for v in FLOW_VARIANTS},
          **{f"{v}_hgb_old_neural": (f"{v}_hgb", "observations_neural") for v in FLOW_VARIANTS},
          **{f"{v}_neural_combined_hgb": (f"{v}_neural", "combined_hgb") for v in ("deep100", "deep500")}}


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"day": day, "model_fits": len(MODEL_SPECS)})
    source = ROOT / protocol["basis_run"] / "dates" / day
    newton_source = ROOT / protocol["newton_run"] / "dates" / day
    context_source = ROOT / protocol["context_run"] / "dates" / day
    for prior_source in (source, context_source, newton_source):
        check_completed(prior_source, json.loads((prior_source.parent.parent / "frozen_screen.json").read_text()))
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    training_x, training_y, validation_x, validation_y, test_x, test_y, test_times = {}, {}, {}, {}, {}, {}, {}
    for current in [*training_days, validation_day, day]:
        partition = folder / "inputs" / f"{current}.json"
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Every registered context day requires both assets")
        write_json(partition, {"sessions": sessions})
        x, y, times = load_flow_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"], ROOT / protocol["flow_manifest"])
        for symbol in SYMBOLS:
            if current in training_days:
                mask = calendar_stride_mask(times[symbol, current], utc_ms(current), stride_seconds=4)
                training_x[symbol, current] = x[symbol, current].loc[mask].copy()
                training_y[symbol, current] = y[symbol, current][mask]
            else:
                mask = noon(times[symbol, current], current)
                if current == validation_day:
                    validation_x[symbol], validation_y[symbol] = x[symbol, current].loc[mask].copy(), y[symbol, current][mask]
                else:
                    test_x[symbol] = x[symbol, current].loc[mask].copy()
                    test_y[symbol], test_times[symbol] = y[symbol, current][mask], times[symbol, current][mask]
        del x, y, times
        gc.collect()
    training_x = {s: pd.concat([training_x[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    training_y = {s: np.concatenate([training_y[s, d] for d in training_days]) for s in SYMBOLS}
    forecasts = {}
    original_tree = joblib.load(context_source / "observations_tree.joblib")
    original_neural = PooledForecaster.load(context_source / "observations_neural")
    combined_hgb = joblib.load(newton_source / "combined_hgb.joblib")
    combined_neural = PooledForecaster.load(source / "combined_neural")
    for asset, symbol in enumerate(SYMBOLS):
        forecasts[symbol] = {}
        for name in BASELINES:
            with np.load(source / "predictions" / f"{symbol}_{name}_registered.npz") as archive:
                forecasts[symbol][name] = {k: archive[k].copy() for k in archive.files}
        np.testing.assert_array_equal(test_y[symbol], forecasts[symbol]["observations_blend"]["labels"])
        np.testing.assert_array_equal(test_times[symbol], forecasts[symbol]["observations_blend"]["decision_times"])
        with np.load(context_source / "predictions" / f"{symbol}_observations_tree_registered.npz") as archive:
            np.testing.assert_array_equal(original_tree.predict_proba(test_x[symbol][original_tree.columns], symbol), archive["probabilities"])
        np.testing.assert_array_equal(combined_hgb.predict_proba(test_x[symbol][combined_hgb.columns], symbol), forecasts[symbol]["combined_hgb"]["probabilities"])
        np.testing.assert_array_equal(combined_neural.predict_proba(test_x[symbol][combined_neural.columns], asset), forecasts[symbol]["combined_neural"]["probabilities"])
        np.testing.assert_array_equal(original_neural.predict_proba(test_x[symbol][original_neural.columns], asset), forecasts[symbol]["observations_neural"]["probabilities"])
        expected_priors = np.array([(training_y[symbol] == k).mean() for k in (-1, 0, 1)])
        np.testing.assert_array_equal(expected_priors, forecasts[symbol]["combined_hgb"]["train_priors"])
    del original_tree, original_neural, combined_hgb, combined_neural
    metadata = {}
    for name, (representation, kind) in MODEL_SPECS.items():
        tx = {s: select_flow_variant(training_x[s], representation) for s in SYMBOLS}
        vx = {s: select_flow_variant(validation_x[s], representation) for s in SYMBOLS}
        test_selected = {s: select_flow_variant(test_x[s], representation) for s in SYMBOLS}
        columns = list(tx[SYMBOLS[0]].columns)
        begin = time.monotonic()
        write_json(output / "progress.json", {"day": day, "stage": "training", "model": name, "scores_sealed": True})
        if kind == "hgb":
            with threadpool_limits(limits=2):
                model = fit_weighted_boost(tx, training_y, leaves=7, class_balanced=True)
            joblib.dump(model, folder / f"{name}.joblib")
            restored = joblib.load(folder / f"{name}.joblib")
        else:
            model, history = fit_confirm_neural(tx, training_y, vx, validation_y)
            model.save(folder / name)
            write_json(folder / name / "training_history.json", history)
            restored = PooledForecaster.load(folder / name)
        for asset, symbol in enumerate(SYMBOLS):
            identifier = asset if kind == "neural" else symbol
            p = restored.predict_proba(test_selected[symbol], identifier)
            np.testing.assert_array_equal(p, model.predict_proba(test_selected[symbol], identifier))
            np.testing.assert_array_equal(restored.priors[asset], forecasts[symbol]["combined_hgb"]["train_priors"])
            forecasts[symbol][name] = {**forecasts[symbol]["combined_hgb"], "probabilities": p}
        metadata[name] = {"representation": representation, "training_dates": training_days,
            "train_rows": {s: len(training_y[s]) for s in SYMBOLS}, "columns": columns,
            "seconds": time.monotonic() - begin, "checkpoint_parity": True}
        print(f"native_flow_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
        del model, restored
        gc.collect()
    for symbol in SYMBOLS:
        for name, (left, right) in BLENDS.items():
            a, b = forecasts[symbol][left], forecasts[symbol][right]
            forecasts[symbol][name] = {**a, "probabilities": (a["probabilities"] + b["probabilities"]) / 2}
    for symbol in SYMBOLS:
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], test_times[symbol])
            initial = archive["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in ("registered", "forecast_3600"):
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], test_times[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], test_y[symbol], test_times[symbol], values["train_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": len(MODEL_SPECS), "seconds": time.monotonic() - started,
              "original_checkpoint_parity": True, "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    records = []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        names = [*BASELINES, *MODEL_SPECS, *BLENDS]
        for symbol in SYMBOLS:
            for name in names:
                for policy in ("registered", "forecast_3600"):
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as archive:
                        metrics = classification_metrics(SimpleNamespace(priors=archive["decision_priors"]), archive["probabilities"], archive["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete causal native_flow-state development family")
    index = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in records}
    leaderboard = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        leaderboard.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "delta_to_same_policy_observation_blend": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "observations_blend", policy]["metrics"]["balanced_accuracy"] for r in rows])),
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(MODEL_SPECS) * len(protocol["dates"]), "post_fit_panels": len(records),
            "substantial_gain_confirmed": False, "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates":
        raise ValueError("This fixed screen is development only")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered causal native_flow-state input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing causal native_flow-state experiment after source changes")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"native_flow_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"native_flow_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_bybit_flow_screen_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_bybit_flow_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the fixed exposed-date causal native_flow-state experiment.")
