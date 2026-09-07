"""Freeze all twenty independent-date predictions before revealing performance."""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS, fit_confirm_neural
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_confirmation_stats import confirmation_summary
from lob_forge.boundary_event_inputs import load_event_inputs, representation_columns, utc_ms
from lob_forge.boundary_forecasts import classification_metrics, fit_classifier
from lob_forge.boundary_pooled import PooledForecaster

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def noon(clock, day):
    return (clock >= utc_ms(day, "12:02:00")) & (clock < utc_ms(day, "13:59:50"))


def rank(metrics):
    return (metrics["balanced_accuracy"], -metrics["log_loss"])


def save_predictions(path, probabilities, labels, clock, training_priors, policy_priors):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=probabilities,
        labels=labels,
        decision_times=clock,
        train_priors=training_priors,
        decision_priors=policy_priors,
    )


def fit_tree(job, name, x, y, validation_x):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_classifier(name, x, y, seed=20260907)
    job.mkdir(parents=True, exist_ok=True)
    path = job / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        model.predict_proba(validation_x.iloc[:128]), restored.predict_proba(validation_x.iloc[:128])
    )
    write_json(
        job / "fit_metadata.json",
        {
            "model": name,
            "train_rows": len(y),
            "feature_columns": list(x.columns),
            "training_class_priors": restored.priors.tolist(),
            "checkpoint_parity": True,
            "warnings": [str(w.message) for w in caught],
        },
    )
    return restored


def run_fold(day, manifest, protocol, output, identity):
    folder = output / "dates" / day
    completed = folder / "completed.json"
    if completed.exists():
        return verify_frozen_fold(folder, identity)
    assessment = date.fromisoformat(day)
    train_days = [(assessment + timedelta(days=offset)).isoformat() for offset in [-5, -4, -3, -2]]
    validation_day = (assessment - timedelta(days=1)).isoformat()
    required_days = [*train_days, validation_day, day]
    chosen_sessions = [s for s in manifest["sessions"] if s["session_date"] in required_days]
    if {(s["symbol"], s["session_date"]) for s in chosen_sessions} != {
        (s, d) for s in SYMBOLS for d in required_days
    } or len(chosen_sessions) != 12:
        raise ValueError("Each chronological fold requires exactly six paired dates")
    if max(train_days) >= validation_day or validation_day >= day:
        raise ValueError("Training, validation and assessment dates must be strictly separated")
    started = time.monotonic()
    attempts = folder / "attempts"
    attempt_number = len(list(attempts.glob("*.json"))) + 1 if attempts.exists() else 1
    write_json(
        attempts / f"{attempt_number:03d}.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "train_dates": train_days,
            "validation_date": validation_day,
            "assessment_date": day,
        },
    )
    partition_manifest = folder / "partition_manifest.json"
    write_json(partition_manifest, {"sessions": chosen_sessions})
    write_json(
        output / "progress.json",
        {"current_date": day, "stage": "constructing_causal_inputs", "performance_sealed": True},
    )
    features, labels, times = load_event_inputs(ROOT, partition_manifest)
    partitions, frequencies = {}, {}
    for symbol in SYMBOLS:
        for current_day in required_days:
            mask = noon(times[symbol, current_day], current_day)
            expected_times = np.arange(utc_ms(current_day, "12:02:00"), utc_ms(current_day, "13:59:50"), 1000)
            np.testing.assert_array_equal(times[symbol, current_day][mask], expected_times)
            partitions[symbol, current_day] = (
                features[symbol, current_day].loc[mask],
                labels[symbol, current_day][mask],
                expected_times,
            )
            # Assessment frequencies are deliberately not computed before sealing.
            if current_day != day:
                frequencies[symbol, current_day] = np.array(
                    [(labels[symbol, current_day][mask] == value).mean() for value in [-1, 0, 1]]
                )
    train_full = {symbol: pd.concat([features[symbol, d] for d in train_days], ignore_index=True) for symbol in SYMBOLS}
    train_full_y = {symbol: np.concatenate([labels[symbol, d] for d in train_days]) for symbol in SYMBOLS}
    val_x = {symbol: partitions[symbol, validation_day][0] for symbol in SYMBOLS}
    val_y = {symbol: partitions[symbol, validation_day][1] for symbol in SYMBOLS}
    write_json(
        output / "progress.json",
        {"current_date": day, "stage": "training_pooled_neural_model", "performance_sealed": True},
    )
    neural, training = fit_confirm_neural(train_full, train_full_y, val_x, val_y)
    neural.save(folder / "neural")
    write_json(folder / "neural/training_history.json", training)
    neural = PooledForecaster.load(folder / "neural")
    selections = {}
    with threadpool_limits(limits=2):
        for asset, symbol in enumerate(SYMBOLS):
            validation_x, validation_y, _ = partitions[symbol, validation_day]
            assessment_x, assessment_y, assessment_times = partitions[symbol, day]
            base_columns = representation_columns(validation_x, "base")
            training_noon_x = pd.concat([partitions[symbol, d][0][base_columns] for d in train_days], ignore_index=True)
            training_noon_y = np.concatenate([partitions[symbol, d][1] for d in train_days])
            write_json(
                output / "progress.json",
                {
                    "current_date": day,
                    "stage": "training_reference_models",
                    "asset": symbol,
                    "performance_sealed": True,
                },
            )
            primary_candidates = []
            for name in protocol["primary_reference"]["models"]:
                model = fit_tree(
                    folder / symbol / "original_reference" / name,
                    name,
                    training_noon_x,
                    training_noon_y,
                    validation_x[base_columns],
                )
                metrics = classification_metrics(model, model.predict_proba(validation_x[base_columns]), validation_y)
                primary_candidates.append((name, model, metrics))
            primary_name, primary, primary_metrics = max(primary_candidates, key=lambda candidate: rank(candidate[2]))
            save_predictions(
                folder / "predictions" / f"{symbol}_original_reference.npz",
                primary.predict_proba(assessment_x[base_columns]),
                assessment_y,
                assessment_times,
                primary.priors,
                primary.priors,
            )
            write_json(
                output / "progress.json",
                {
                    "current_date": day,
                    "stage": "training_candidate_and_data_controls",
                    "asset": symbol,
                    "performance_sealed": True,
                },
            )
            tree = fit_tree(
                folder / symbol / "candidate_tree", "hgb_7", train_full[symbol], train_full_y[symbol], validation_x
            )
            np.testing.assert_array_equal(tree.priors, neural.priors[asset])
            neural_p = neural.predict_proba(assessment_x, asset)
            tree_p = tree.predict_proba(assessment_x)
            candidate_p = 0.5 * neural_p + 0.5 * tree_p
            candidate_priors = 0.5 * (tree.priors + frequencies[symbol, validation_day])
            for kind, p in [("candidate", candidate_p), ("neural_component", neural_p), ("tree_component", tree_p)]:
                save_predictions(
                    folder / "predictions" / f"{symbol}_{kind}.npz",
                    p,
                    assessment_y,
                    assessment_times,
                    tree.priors,
                    candidate_priors,
                )
            controls = []
            for name in protocol["matched_data_control"]["models"]:
                model = fit_tree(
                    folder / symbol / "matched_data_control" / name,
                    name,
                    train_full[symbol][base_columns],
                    train_full_y[symbol],
                    validation_x[base_columns],
                )
                p = model.predict_proba(validation_x[base_columns])
                for policy in protocol["matched_data_control"]["decision_policies"]:
                    policy_priors = (
                        model.priors
                        if policy == "training"
                        else 0.5 * (model.priors + frequencies[symbol, train_days[-1]])
                    )
                    metrics = classification_metrics(SimpleNamespace(priors=policy_priors), p, validation_y)
                    controls.append((name, policy, model, metrics))
            control_name, control_policy, control, control_metrics = max(
                controls, key=lambda candidate: rank(candidate[3])
            )
            control_priors = (
                control.priors
                if control_policy == "training"
                else 0.5 * (control.priors + frequencies[symbol, validation_day])
            )
            save_predictions(
                folder / "predictions" / f"{symbol}_matched_data_control.npz",
                control.predict_proba(assessment_x[base_columns]),
                assessment_y,
                assessment_times,
                control.priors,
                control_priors,
            )
            selections[symbol] = {
                "primary_model": primary_name,
                "primary_validation": primary_metrics,
                "primary_trials": [{"model": name, "validation": metrics} for name, _, metrics in primary_candidates],
                "control_model": control_name,
                "control_policy": control_policy,
                "control_validation": control_metrics,
                "control_trials": [
                    {"model": name, "policy": policy, "validation": metrics} for name, policy, _, metrics in controls
                ],
            }
    write_json(folder / "validation_selections.json", selections)
    saved = {
        "identity": identity,
        "assessment_date": day,
        "train_dates": train_days,
        "validation_date": validation_day,
        "model_fits": 17,
        "assessment_performance_revealed": False,
        "seconds": time.monotonic() - started,
        "artifact_hashes": {
            str(path.relative_to(folder)): sha256_file(path)
            for path in folder.rglob("*")
            if path.is_file() and path != completed
        },
    }
    write_json(completed, saved)
    return saved


def reveal(output, dates, identity):
    if any(not (output / "dates" / day / "completed.json").exists() for day in dates):
        raise ValueError("All twenty dates must have frozen predictions before performance is revealed")
    # Recheck every fold before reading a single assessment metric.
    for day in dates:
        verify_frozen_fold(output / "dates" / day, identity)
    records = []
    for day in dates:
        for symbol in SYMBOLS:
            metrics, common = {}, None
            for model in ["original_reference", "matched_data_control", "candidate"]:
                path = output / "dates" / day / "predictions" / f"{symbol}_{model}.npz"
                with np.load(path) as prediction:
                    current = prediction["decision_times"], prediction["labels"]
                    if common is not None:
                        for old, new in zip(common, current):
                            np.testing.assert_array_equal(old, new)
                    common = tuple(array.copy() for array in current)
                    metrics[model] = classification_metrics(
                        SimpleNamespace(priors=prediction["decision_priors"]),
                        prediction["probabilities"],
                        prediction["labels"],
                    )
                    if any(count == 0 for count in metrics[model]["class_counts"]):
                        raise ValueError("All three class recalls must be identifiable on every registered asset/date")
            records.append({"symbol": symbol, "assessment_date": day, **metrics})
    return {"records": records, **confirmation_summary(records, dates)}


def run(protocol_path, output, *, development_reproduction=False, development_check=False):
    protocol = json.loads(protocol_path.read_text())
    if protocol["candidate"]["neural_config"] != CONFIG or protocol["development_selection"]["neural_weight"] != 0.5:
        raise ValueError("Registered candidate configuration differs from this frozen implementation")
    for name, checksum in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Registered feature or model source changed")
    if development_reproduction and "label_amendment" in protocol:
        raise ValueError("Legacy bitwise reproduction requires the original protocol and its archived code, before corrected targets")
    manifest_path = ROOT / protocol.get("execution_manifest", (
        "data/research/boundary_event_data_20260907/dataset_manifest.json" if development_reproduction
        else "data/research/boundary_confirmation_20260907/combined_dataset_manifest.json"
    ))
    manifest = json.loads(manifest_path.read_text())
    first = date.fromisoformat(protocol["source_plan"]["first_date"])
    dates = [(first + timedelta(days=offset)).isoformat() for offset in range(20)]
    if development_reproduction or development_check:
        dates = ["2023-05-23"]
    elif manifest["confirmation_dates"] != dates or len(manifest["sessions"]) != 56:
        raise ValueError("Prepared data differs from the twenty-date confirmation registration")
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_sha256": sha256_file(manifest_path),
        "development_reproduction": development_reproduction,
        "development_check": development_check,
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_confirmation.py",
                "src/lob_forge/boundary_confirm_model.py",
                "src/lob_forge/boundary_confirmation_stats.py",
                "src/lob_forge/boundary_confirmation_integrity.py",
                "src/lob_forge/label_math.py",
                "src/lob_forge/boundary_events.py",
                "src/lob_forge/boundary_event_inputs.py",
                "src/lob_forge/boundary_forecasts.py",
                "src/lob_forge/boundary_pooled.py",
            ]
        },
        "python": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow"]
        },
    }
    frozen = output / "frozen_study.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this study and use a new output directory after any source or data change")
    write_json(frozen, identity)
    for index, day in enumerate(dates, 1):
        record = run_fold(day, manifest, protocol, output, identity)
        write_json(
            output / "progress.json",
            {
                "completed_dates": index,
                "planned_dates": len(dates),
                "latest_date": day,
                "model_fits": index * 17,
                "performance_sealed": True,
                "latest_seconds": record["seconds"],
            },
        )
        print(
            f"dates_frozen={index}/{len(dates)} model_fits={index * 17} date={day} seconds={record['seconds']:.1f}",
            flush=True,
        )
    if development_check:
        verify_frozen_fold(output / "dates/2023-05-23", identity)
        for symbol in SYMBOLS:
            with np.load(output / "dates/2023-05-23/predictions" / f"{symbol}_candidate.npz") as saved:
                probabilities = saved["probabilities"]
                assert probabilities.shape == (7070, 3) and np.isfinite(probabilities).all()
                assert (probabilities >= 0).all()
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        write_json(
            ROOT / "docs/research/boundary_exact_labels_development_check_20260907.json",
            {"status": "passed", "identity": identity, "model_fits": 17, "performance_inspected": False,
             "purpose": "Mechanical pipeline check on an exposed date, without changing the fixed candidate."},
        )
        print("exact_label_development_pipeline_check_passed", flush=True)
        return
    if development_reproduction:
        comparisons = []
        for symbol in SYMBOLS:
            current = np.load(output / "dates/2023-05-23/predictions" / f"{symbol}_candidate.npz")
            old_path = (
                ROOT
                / "results/boundary_probability_blend_20260907"
                / symbol
                / "2023-05-23/0.5/assessment_predictions.npz"
            )
            old = np.load(old_path)
            for key in ["probabilities", "labels", "decision_times"]:
                np.testing.assert_array_equal(current[key], old[key])
            np.testing.assert_array_equal(
                np.broadcast_to(current["decision_priors"], old["decision_priors"].shape), old["decision_priors"]
            )
            comparisons.append(
                {
                    "symbol": symbol,
                    "rows": len(old["labels"]),
                    "bit_identical": True,
                    "reference_sha256": sha256_file(old_path),
                }
            )
        write_json(
            ROOT / "docs/research/boundary_confirmation_pipeline_reproduction_20260907.json",
            {"status": "passed", "identity": identity, "comparisons": comparisons},
        )
        print("complete_pipeline_development_reproduction_passed", flush=True)
        return
    summary = reveal(output, dates, identity)
    write_json(output / "summary.json", {"identity": identity, "completed_model_fits": 340, **summary})
    print(f"confirmation_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_confirmation_20260907_exact_labels_revision.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-reproduction", action="store_true")
    parser.add_argument("--development-check", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    destination = args.output or ROOT / (
        "results/boundary_confirmation_pipeline_reproduction_20260907"
        if args.development_reproduction
        else "results/boundary_exact_labels_development_check_20260907" if args.development_check
        else "results/boundary_confirmation_exact_labels_20260907"
    )
    if args.run:
        if args.development_reproduction and args.development_check:
            parser.error("Choose only one development check mode")
        run(args.protocol, destination, development_reproduction=args.development_reproduction, development_check=args.development_check)
    else:
        print("Pass --run after fixed-source preparation; --development-reproduction checks only an exposed date.")
