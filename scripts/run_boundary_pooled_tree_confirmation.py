"""One fixed secondary confirmation registered before either outcome is inspected."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirmation_family import family_interval
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_event_inputs import load_event_inputs
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, reveal, save_predictions, write_json

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def run_fold(day, primary, output, primary_identity, identity):
    source = primary / "dates" / day
    source_record = verify_frozen_fold(source, primary_identity)
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return verify_frozen_fold(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "partition_manifest.json"
    shutil.copyfile(source / "partition_manifest.json", manifest)
    x, y, clocks = load_event_inputs(ROOT, manifest)
    training_days = source_record["train_dates"]
    train_x = {s: pd.concat([x[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    train_y = {s: np.concatenate([y[s, d] for d in training_days]) for s in SYMBOLS}
    write_json(output / "progress.json", {"current_date": day, "stage": "training_pooled_balanced_tree", "performance_sealed": True})
    with threadpool_limits(limits=2):
        model = fit_weighted_boost(train_x, train_y, leaves=7, class_balanced=True, seed=20260907)
    path = folder / "pooled_tree.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    neural = PooledForecaster.load(source / "neural")
    np.testing.assert_array_equal(restored.priors, neural.priors)
    for asset, symbol in enumerate(SYMBOLS):
        validation = x[symbol, source_record["validation_date"]].iloc[:128]
        np.testing.assert_array_equal(model.predict_proba(validation, symbol), restored.predict_proba(validation, symbol))
        mask = noon(clocks[symbol, day], day)
        assessment_x, assessment_y, assessment_times = x[symbol, day].loc[mask], y[symbol, day][mask], clocks[symbol, day][mask]
        with np.load(source / "predictions" / f"{symbol}_candidate.npz") as old:
            np.testing.assert_array_equal(assessment_y, old["labels"])
            np.testing.assert_array_equal(assessment_times, old["decision_times"])
            decision_priors = old["decision_priors"].copy()
        with np.load(source / "predictions" / f"{symbol}_neural_component.npz") as old:
            neural_p = old["probabilities"].copy()
            # The reused component really belongs to this frozen checkpoint and schema.
            np.testing.assert_array_equal(neural_p, neural.predict_proba(assessment_x, asset))
        tree_p = restored.predict_proba(assessment_x, symbol)
        for name, p in [("candidate", 0.5 * neural_p + 0.5 * tree_p), ("neural_component", neural_p), ("tree_component", tree_p)]:
            save_predictions(folder / "predictions" / f"{symbol}_{name}.npz", p, assessment_y, assessment_times, restored.priors[asset], decision_priors)
        for name in ["original_reference", "matched_data_control"]:
            shutil.copyfile(source / "predictions" / f"{symbol}_{name}.npz", folder / "predictions" / f"{symbol}_{name}.npz")
    write_json(folder / "fit_metadata.json", {
        "source_completed_sha256": sha256_file(source / "completed.json"),
        "train_dates": training_days, "symbols": restored.symbols, "features": restored.columns,
        "class_balanced": restored.class_balanced, "asset_class_priors": restored.priors.tolist(),
        "neural_checkpoint_and_prediction_parity": True, "tree_checkpoint_parity": True,
    })
    record = {
        "identity": identity, "assessment_date": day, "train_dates": training_days,
        "validation_date": source_record["validation_date"], "model_fits": 1,
        "assessment_performance_revealed": False, "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()},
    }
    write_json(folder / "completed.json", record)
    return record


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Registered secondary procedure changed")
    primary = ROOT / protocol["primary_output"]
    frozen = primary / "frozen_study.json"
    if sha256_file(frozen) != protocol["primary_frozen_study_sha256"]:
        raise ValueError("Primary confirmation identity changed")
    primary_identity = json.loads(frozen.read_text())
    if platform.python_version() != primary_identity["python"] or any(
        version(name) != expected for name, expected in primary_identity["dependencies"].items()
    ):
        raise ValueError("Secondary procedure must use the same frozen dependency environment")
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dates = protocol["assessment_dates"]
    # This reads identities and hashes, never the primary summary or scores.
    for day in dates:
        verify_frozen_fold(primary / "dates" / day, primary_identity)
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "primary_frozen_study_sha256": sha256_file(frozen),
        "code_hashes": protocol["code_hashes"],
    }
    destination = output / "frozen_study.json"
    if destination.exists() and json.loads(destination.read_text()) != identity:
        raise ValueError("Preserve existing secondary confirmation after any source change")
    write_json(destination, identity)
    for index, day in enumerate(dates, 1):
        record = run_fold(day, primary, output, primary_identity, identity)
        write_json(output / "progress.json", {"completed_dates": index, "planned_dates": len(dates), "latest_date": day, "model_fits": index, "performance_sealed": True})
        print(f"secondary_dates_frozen={index}/{len(dates)} model_fits={index} date={day} seconds={record['seconds']:.1f}", flush=True)
    # Only after both complete procedures are frozen do we inspect either set of scores.
    results = {"original_blend": reveal(primary, dates, primary_identity), "pooled_balanced_tree_blend": reveal(output, dates, identity)}
    for result in results.values():
        deltas = [r["original_reference"] for r in result["paired_date_balanced_accuracy_deltas"]]
        result["two_procedure_family_interval"] = family_interval(deltas)
        result["two_procedure_family_block_sensitivity"] = family_interval(deltas, block_length=3)
        result["substantial_predictive_gain_confirmed_with_family_control"] = (
            result["substantial_predictive_gain_confirmed"] and result["two_procedure_family_interval"]["lower_97_5"] > 0
        )
    primary_records = {(r["symbol"], r["assessment_date"]): r for r in results["original_blend"]["records"]}
    secondary_records = {(r["symbol"], r["assessment_date"]): r for r in results["pooled_balanced_tree_blend"]["records"]}
    extra = [np.mean([
        secondary_records[s, d]["candidate"]["balanced_accuracy"] - primary_records[s, d]["candidate"]["balanced_accuracy"]
        for s in SYMBOLS
    ]) for d in dates]
    write_json(output / "summary.json", {
        "identity": identity, "completed_additional_model_fits": 20, "procedures": results,
        "secondary_minus_primary_balanced_accuracy": family_interval(extra),
        "substantial_predictive_gain_confirmed": any(r["substantial_predictive_gain_confirmed_with_family_control"] for r in results.values()),
        "economic_performance_confirmed": False,
    })
    print(f"two_procedure_confirmation_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_pooled_tree_confirmation_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_pooled_tree_confirmation_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run only after the primary predictions for all twenty dates are frozen; do not inspect its scores.")
