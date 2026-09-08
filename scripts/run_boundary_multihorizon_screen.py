"""Test joint-horizon training while preserving every primary assessment target."""

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

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_combined_inputs import load_combined_inputs, original_combined_columns
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_multihorizon_model import MultihorizonForecaster, fit_multihorizon_neural
from lob_forge.boundary_multihorizon_targets import HORIZONS_MS, MULTIHORIZON_SEMANTICS
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_embedding_screen import BASELINES as EMBEDDING_BASELINES, BLENDS as EMBEDDING_BLENDS, MODEL_SPECS as EMBEDDING_MODELS
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*EMBEDDING_BASELINES, *EMBEDDING_MODELS, *EMBEDDING_BLENDS)
MODEL_SPECS = {f"{representation}_aux{int(weight * 100)}": (representation, weight)
               for representation in ("observations", "combined") for weight in (0, .25, .5)}
BLENDS = {f"{name}_{suffix}": (name, "deep500_hgb" if suffix == "deep500_hgb" else
           "observations_tree" if representation == "observations" else "combined_hgb")
          for name, (representation, _) in MODEL_SPECS.items() for suffix in ("matched_hgb", "deep500_hgb")}


def aligned_historical_targets(record, selected_times, primary_labels, *, cutoff_ms):
    if record["semantics"] != MULTIHORIZON_SEMANTICS or sha256_file(ROOT / record["targets_path"]) != record["sha256"]:
        raise ValueError("Frozen multi-horizon target artifact changed")
    with np.load(ROOT / record["targets_path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["horizons_ms"], HORIZONS_MS)
        positions = np.searchsorted(saved["decision_times"], selected_times)
        if (positions >= len(saved["decision_times"])).any():
            raise ValueError("Every original training decision requires a target record")
        np.testing.assert_array_equal(saved["decision_times"][positions], selected_times)
        labels, available = saved["labels"][positions].copy(), saved["available"][positions].copy()
        np.testing.assert_array_equal(labels[:, 0], primary_labels)
        if not available[:, 0].all() or (saved["future_times"][positions][available] >= cutoff_ms).any():
            raise ValueError("All used historical labels must be released before the validation day")
    return labels, available


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"date": day, "planned_fits": list(MODEL_SPECS)})
    parent = ROOT / protocol["embedding_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    target_manifest = json.loads((ROOT / protocol["target_manifest"]).read_text())
    target_index = {(r["symbol"], r["session_date"]): r for r in target_manifest["sessions"]}
    tx, ty, ta, vx, vy, test_x, test_y, test_times = {}, {}, {}, {}, {}, {}, {}, {}
    for current in [*training_days, validation_day, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Every frozen source date requires both assets")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, times = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for symbol in SYMBOLS:
            key = symbol, current
            if current in training_days:
                mask = calendar_stride_mask(times[key], utc_ms(current), stride_seconds=4)
                tx[key] = x[key].loc[mask].copy()
                ty[key], ta[key] = aligned_historical_targets(target_index[key], times[key][mask], y[key][mask], cutoff_ms=utc_ms(validation_day))
            else:
                mask = noon(times[key], current)
                if mask.sum() != 7070:
                    raise ValueError("The unchanged full noon universe is required")
                if current == validation_day:
                    vx[symbol], vy[symbol] = x[key].loc[mask].copy(), y[key][mask]
                else:
                    test_x[symbol], test_y[symbol], test_times[symbol] = x[key].loc[mask].copy(), y[key][mask], times[key][mask]
        del x, y, times
        gc.collect()
    tx = {s: pd.concat([tx[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    ty = {s: np.concatenate([ty[s, d] for d in training_days]) for s in SYMBOLS}
    ta = {s: np.concatenate([ta[s, d] for d in training_days]) for s in SYMBOLS}
    forecasts = {s: {} for s in SYMBOLS}
    for symbol in SYMBOLS:
        for name in BASELINES:
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz") as archive:
                forecasts[symbol][name] = {k: archive[k].copy() for k in archive.files}
            np.testing.assert_array_equal(test_y[symbol], forecasts[symbol][name]["labels"])
            np.testing.assert_array_equal(test_times[symbol], forecasts[symbol][name]["decision_times"])
    for representation in ("observations", "combined"):
        source = ROOT / protocol["context_run" if representation == "observations" else "basis_run"] / "dates" / day
        model = PooledForecaster.load(source / f"{representation}_neural")
        columns = original_combined_columns(tx[SYMBOLS[0]]) if representation == "observations" else list(tx[SYMBOLS[0]].columns)
        if columns != model.columns or len(columns) != protocol["representations"][representation]:
            raise ValueError("Original observation-only schema changed")
        for asset, symbol in enumerate(SYMBOLS):
            np.testing.assert_array_equal(model.predict_proba(test_x[symbol][columns], asset), forecasts[symbol][f"{representation}_neural"]["probabilities"])
            np.testing.assert_array_equal(model.priors[asset], np.array([(ty[symbol][:, 0] == k).mean() for k in (-1, 0, 1)]))
        del model
    metadata = {}
    for name, (representation, auxiliary_weight) in MODEL_SPECS.items():
        begin = time.monotonic()
        columns = original_combined_columns(tx[SYMBOLS[0]]) if representation == "observations" else list(tx[SYMBOLS[0]].columns)
        write_json(output / "progress.json", {"date": day, "stage": "training", "model": name, "assessment_scores_sealed": True})
        model, history = fit_multihorizon_neural({s: tx[s][columns] for s in SYMBOLS}, ty, ta,
            {s: vx[s][columns] for s in SYMBOLS}, vy, auxiliary_weight=auxiliary_weight)
        model.save(folder / name)
        write_json(folder / name / "training_history.json", history)
        restored = MultihorizonForecaster.load(folder / name)
        tree_name = "observations_tree" if representation == "observations" else "combined_hgb"
        for asset, symbol in enumerate(SYMBOLS):
            p = restored.predict_proba(test_x[symbol][columns], asset)
            np.testing.assert_array_equal(p, model.predict_proba(test_x[symbol][columns], asset))
            np.testing.assert_array_equal(restored.priors[asset], forecasts[symbol][tree_name]["train_priors"])
            if auxiliary_weight == 0:
                np.testing.assert_array_equal(p, forecasts[symbol][f"{representation}_neural"]["probabilities"])
            forecasts[symbol][name] = {**forecasts[symbol][tree_name], "probabilities": p}
        metadata[name] = {"representation": representation, "auxiliary_weight": auxiliary_weight, "training_dates": training_days,
            "training_rows": {s: len(ty[s]) for s in SYMBOLS}, "columns": columns, "seconds": time.monotonic() - begin,
            "best_epoch": history["best_epoch"], "parameter_count": history["parameter_count"], "checkpoint_parity": True,
            "zero_auxiliary_reproduction": auxiliary_weight == 0, "primary_training_labels_and_input_schema_exact": True}
        print(f"multihorizon_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
        del model, restored
        gc.collect()
    for symbol in SYMBOLS:
        for name, (left, right) in BLENDS.items():
            a, b = forecasts[symbol][left], forecasts[symbol][right]
            np.testing.assert_array_equal(a["train_priors"], b["train_priors"])
            forecasts[symbol][name] = {**a, "probabilities": (a["probabilities"] + b["probabilities"]) / 2}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], test_times[symbol])
            initial = archive["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], test_times[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], test_y[symbol],
                    test_times[symbol], values["train_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": len(MODEL_SPECS), "seconds": time.monotonic() - started,
        "all_original_rows_labels_and_checkpoint_predictions_exact": True, "zero_auxiliary_reproduction_exact": True,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    records = []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            for name in [*BASELINES, *MODEL_SPECS, *BLENDS]:
                for policy in protocol["decision_policies"]:
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as saved:
                        metrics = classification_metrics(SimpleNamespace(priors=saved["decision_priors"]), saved["probabilities"], saved["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete joint-horizon development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(MODEL_SPECS) * len(protocol["dates"]),
        "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the exposed-date fixed primary policies are registered")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered joint-horizon input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing joint-horizon experiment after source changes")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"multihorizon_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"multihorizon_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_multihorizon_screen_20260908.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_multihorizon_screen_20260908")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the frozen joint future-horizon training experiment.")
