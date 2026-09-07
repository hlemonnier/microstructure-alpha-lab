"""Fixed five-date development screen for event-clock forecast information."""

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
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_event_clock import EVENT_CLOCK_FEATURES, PEER_CLOCK_FEATURES
from lob_forge.boundary_event_clock_inputs import load_clock_inputs
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_weighted_boost import assert_matching_asset_priors, fit_weighted_boost
from run_boundary_confirmation import noon, save_predictions, write_json

ROOT = Path(__file__).resolve().parents[1]
FITS = ("clock_tree_7", "clock_tree_31", "base_tree_31", "clock_neural")
BLENDS = {
    "old_neural_clock_tree_7": ("old_neural", "clock_tree_7"),
    "old_neural_clock_tree_31": ("old_neural", "clock_tree_31"),
    "old_neural_base_tree_31": ("old_neural", "base_tree_31"),
    "clock_neural_old_tree": ("clock_neural", "old_pooled_tree"),
    "clock_neural_clock_tree_7": ("clock_neural", "clock_tree_7"),
    "clock_neural_clock_tree_31": ("clock_neural", "clock_tree_31"),
}


def check_completed(folder, identity):
    record = json.loads((folder / "completed.json").read_text())
    if record["identity"] != identity:
        raise ValueError("Frozen development identity changed")
    for name, checksum in record["artifact_hashes"].items():
        if sha256_file(folder / name) != checksum:
            raise ValueError("Frozen development artifact changed")
    return record


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"day": day, "planned_fits": list(FITS), "evidence_status": protocol["evidence_status"]})
    primary = ROOT / protocol["primary_run"] / "dates" / day
    secondary = ROOT / protocol["secondary_run"] / "dates" / day
    primary_record = verify_frozen_fold(primary, json.loads((primary.parent.parent / "frozen_study.json").read_text()))
    verify_frozen_fold(secondary, json.loads((secondary.parent.parent / "frozen_study.json").read_text()))
    training_days, validation_day = primary_record["train_dates"], primary_record["validation_date"]
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    required = {*training_days, validation_day, day}
    write_json(folder / "partition_manifest.json", {"sessions": [s for s in manifest["sessions"] if s["session_date"] in required]})
    x, y, clocks = load_clock_inputs(ROOT, folder / "partition_manifest.json")
    extra_columns = {*EVENT_CLOCK_FEATURES, *("peer_" + name for name in PEER_CLOCK_FEATURES)}
    base_columns = [name for name in x[SYMBOLS[0], day].columns if name not in extra_columns]
    train_x = {s: pd.concat([x[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    train_y = {s: np.concatenate([y[s, d] for d in training_days]) for s in SYMBOLS}
    val_x = {s: x[s, validation_day].loc[noon(clocks[s, validation_day], validation_day)] for s in SYMBOLS}
    val_y = {s: y[s, validation_day][noon(clocks[s, validation_day], validation_day)] for s in SYMBOLS}
    test_x = {s: x[s, day].loc[noon(clocks[s, day], day)] for s in SYMBOLS}
    frozen = {}
    # The old checkpoint must see exactly the same base schema and yield exactly
    # the same predictions after adding observations to the source parquet files.
    old_tree = joblib.load(secondary / "pooled_tree.joblib")
    old_neural = PooledForecaster.load(primary / "neural")
    for asset, symbol in enumerate(SYMBOLS):
        frozen[symbol] = {}
        for name, source, kind in (
            ("original_reference", primary, "original_reference"), ("matched_data_control", primary, "matched_data_control"),
            ("original_blend", primary, "candidate"), ("pooled_blend", secondary, "candidate"),
            ("old_neural", primary, "neural_component"), ("old_pooled_tree", secondary, "tree_component"),
        ):
            with np.load(source / "predictions" / f"{symbol}_{kind}.npz") as archive:
                frozen[symbol][name] = {k: archive[k].copy() for k in archive.files}
        canonical = frozen[symbol]["original_blend"]
        mask = noon(clocks[symbol, day], day)
        np.testing.assert_array_equal(canonical["labels"], y[symbol, day][mask])
        np.testing.assert_array_equal(canonical["decision_times"], clocks[symbol, day][mask])
        np.testing.assert_array_equal(old_tree.predict_proba(test_x[symbol][base_columns], symbol), frozen[symbol]["old_pooled_tree"]["probabilities"])
        np.testing.assert_array_equal(old_neural.predict_proba(test_x[symbol][base_columns], asset), frozen[symbol]["old_neural"]["probabilities"])
    del old_tree, old_neural, x, y
    gc.collect()
    fit_metadata = {}
    for name in FITS:
        fit_start = time.monotonic()
        write_json(output / "progress.json", {"date": day, "stage": "training", "model": name, "development_scores_sealed": True})
        if name == "clock_neural":
            model, metadata = fit_confirm_neural(train_x, train_y, val_x, val_y)
            model.save(folder / name)
            restored = PooledForecaster.load(folder / name)
            write_json(folder / name / "training_history.json", metadata)
            active_test = test_x
        else:
            active_train = {s: frame[base_columns] for s, frame in train_x.items()} if name == "base_tree_31" else train_x
            active_test = {s: frame[base_columns] for s, frame in test_x.items()} if name == "base_tree_31" else test_x
            with threadpool_limits(limits=2):
                model = fit_weighted_boost(active_train, train_y, leaves=int(name.rsplit("_", 1)[1]), class_balanced=True, seed=20260907)
            joblib.dump(model, folder / f"{name}.joblib")
            restored = joblib.load(folder / f"{name}.joblib")
        for asset, symbol in enumerate(SYMBOLS):
            canonical = frozen[symbol]["original_blend"]
            identifier = asset if name == "clock_neural" else symbol
            p = restored.predict_proba(active_test[symbol], identifier)
            np.testing.assert_array_equal(p, model.predict_proba(active_test[symbol], identifier))
            np.testing.assert_array_equal(restored.priors[asset], canonical["train_priors"])
            frozen[symbol][name] = {**canonical, "probabilities": p}
        fit_metadata[name] = {"seconds": time.monotonic() - fit_start, "checkpoint_parity": True, "training_rows": {s: len(v) for s, v in train_y.items()}, "features": restored.columns}
        print(f"event_clock_fit={day}/{name} seconds={fit_metadata[name]['seconds']:.1f}", flush=True)
        del model, restored
        gc.collect()
    # The neural prior map and every tree prior array must describe the same
    # full-day training rows; this assertion does not read an assessment class mix.
    assert_matching_asset_priors(np.array([frozen[s]["clock_tree_7"]["train_priors"] for s in SYMBOLS]), {i: frozen[s]["clock_neural"]["train_priors"] for i, s in enumerate(SYMBOLS)})
    for symbol in SYMBOLS:
        for name, (left, right) in BLENDS.items():
            frozen[symbol][name] = {**frozen[symbol]["original_blend"], "probabilities": (frozen[symbol][left]["probabilities"] + frozen[symbol][right]["probabilities"]) / 2}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], frozen[symbol]["original_blend"]["decision_times"])
            initial = archive["labels_cumulative"][0].copy()
        for name, data in frozen[symbol].items():
            for policy in protocol["decision_policies"]:
                prior = data["decision_priors"] if policy == "registered" else forecast_priors(data["probabilities"], data["decision_times"], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", data["probabilities"], data["labels"], data["decision_times"], data["train_priors"], prior)
    write_json(folder / "fit_metadata.json", fit_metadata)
    record = {"identity": identity, "date": day, "model_fits": len(FITS), "seconds": time.monotonic() - started,
              "original_checkpoint_and_row_parity": True, "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    records = []
    names = ["original_reference", "matched_data_control", "original_blend", "pooled_blend", "old_neural", "old_pooled_tree", *FITS, *BLENDS]
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            for name in names:
                for policy in protocol["decision_policies"]:
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as archive:
                        metrics = classification_metrics(SimpleNamespace(priors=archive["decision_priors"]), archive["probabilities"], archive["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    index = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in records}
    leaderboard = []
    for name in names:
        for policy in protocol["decision_policies"]:
            rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
            leaderboard.append({
                "model": name, "policy": policy, "asset_dates": len(rows),
                "means": {key: float(np.mean([r["metrics"][key] for r in rows])) for key in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
                "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS},
                "delta_to_original_registered_reference": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "original_reference", "registered"]["metrics"]["balanced_accuracy"] for r in rows])),
                "delta_to_same_policy_original_blend": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "original_blend", policy]["metrics"]["balanced_accuracy"] for r in rows])),
            })
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(FITS) * len(protocol["dates"]), "post_fit_panels": len(records),
            "substantial_gain_confirmed": False, "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["fits"] != list(FITS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the fixed exposed-data development family is authorized by this protocol")
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered development input changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this development run after any identity change")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"event_clock_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"event_clock_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_event_clock_screen_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_event_clock_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for the fixed, development-only event-clock screen.")
