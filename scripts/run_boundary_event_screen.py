"""Run registered training-volume and event-representation ablations."""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_event_inputs import load_event_inputs, representation_columns, utc_ms
from lob_forge.boundary_forecasts import classification_metrics, fit_classifier

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(records):
    groups, selected = {}, []
    for record in records:
        key = (record["training_window"], record["representation"], record["model"])
        groups.setdefault(key, []).append(record)
    leaderboard = [
        {
            "training_window": key[0],
            "representation": key[1],
            "model": key[2],
            "asset_folds": len(rows),
            **{
                f"mean_development_{metric}": float(np.mean([r["assessment"][metric] for r in rows]))
                for metric in ["balanced_accuracy", "natural_accuracy", "log_loss"]
            },
        }
        for key, rows in groups.items()
    ]

    def ranking(record):
        return (record["validation"]["balanced_accuracy"], -record["validation"]["log_loss"])

    for symbol, day in sorted({(r["symbol"], r["assessment_date"]) for r in records}):
        rows = [r for r in records if (r["symbol"], r["assessment_date"]) == (symbol, day)]
        if len(rows) != 12:
            continue
        reference = max([r for r in rows if r["representation"] == "base"], key=ranking)
        candidate = max([r for r in rows if r["representation"] != "base"], key=ranking)
        keys = ["model", "training_window", "representation", "validation", "assessment"]
        selected.append(
            {
                "symbol": symbol,
                "assessment_date": day,
                "reference": {k: reference[k] for k in keys},
                "challenger": {k: candidate[k] for k in keys},
                "balanced_accuracy_difference": candidate["assessment"]["balanced_accuracy"]
                - reference["assessment"]["balanced_accuracy"],
            }
        )
    return {
        "leaderboard": sorted(leaderboard, key=lambda r: r["mean_development_balanced_accuracy"], reverse=True),
        "validation_selected_comparison": selected,
        "evidence_status": "development_only_previously_exposed_dates",
        "substantial_gain_confirmed": False,
    }


def noon(times, day):
    return (times >= utc_ms(day, "12:02:00")) & (times < utc_ms(day, "13:59:50"))


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    manifest = ROOT / protocol["dataset_manifest"]
    if sha256_file(manifest) != protocol["dataset_sha256"]:
        raise ValueError("Registered dataset identity changed")
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest),
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_event_screen.py",
                "src/lob_forge/boundary_event_inputs.py",
                "src/lob_forge/boundary_forecasts.py",
                "src/lob_forge/boundary_events.py",
            ]
        },
        "python": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow"]
        },
    }
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Source changed: preserve this run and use a new output directory")
    write_json(frozen, identity)
    features, labels, times = load_event_inputs(ROOT, manifest)
    print(f"loaded {sum(len(y) for y in labels.values())} eligible full-day decisions", flush=True)
    dates = sorted({day for symbol, day in labels})
    records = []
    with threadpool_limits(limits=2):
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for index in range(5, len(dates)):
                train_dates, val_date, dev_date = dates[index - 5 : index - 1], dates[index - 1], dates[index]
                for training_window in protocol["training_windows"]:
                    train_masks = {
                        day: noon(times[symbol, day], day)
                        if training_window == "noon"
                        else np.ones(len(times[symbol, day]), dtype=bool)
                        for day in train_dates
                    }
                    train_y = np.concatenate([labels[symbol, day][train_masks[day]] for day in train_dates])
                    for representation in protocol["representations"]:
                        columns = representation_columns(features[symbol, dev_date], representation)
                        train_x = pd.concat(
                            [features[symbol, day].loc[train_masks[day], columns] for day in train_dates],
                            ignore_index=True,
                        )
                        partitions = {}
                        for partition, day in [("validation", val_date), ("assessment", dev_date)]:
                            keep = noon(times[symbol, day], day)
                            partitions[partition] = (
                                features[symbol, day].loc[keep, columns],
                                labels[symbol, day][keep],
                                times[symbol, day][keep],
                            )
                            baseline_path = (
                                ROOT
                                / "results/boundary_predictive_screen_20260907"
                                / symbol
                                / dev_date
                                / "base/logistic_0.01"
                                / f"{partition}_predictions.npz"
                            )
                            baseline = np.load(baseline_path)
                            np.testing.assert_array_equal(times[symbol, day][keep], baseline["decision_times"])
                            np.testing.assert_array_equal(labels[symbol, day][keep], baseline["labels"])
                        for name in protocol["models"]:
                            job = output / symbol / dev_date / training_window / representation / name
                            completed = job / "completed.json"
                            if completed.exists():
                                saved = json.loads(completed.read_text())
                                if saved["identity"] != identity:
                                    raise ValueError("Run identity mismatch")
                                for filename, checksum in saved["artifact_hashes"].items():
                                    if sha256_file(job / filename) != checksum:
                                        raise ValueError("Artifact checksum mismatch")
                                records.append(saved["record"])
                                continue
                            started = time.monotonic()
                            write_json(
                                job / "attempt_started.json",
                                {
                                    "identity": identity,
                                    "training_window": training_window,
                                    "model": name,
                                    "representation": representation,
                                    "train_dates": train_dates,
                                    "validation_date": val_date,
                                    "assessment_date": dev_date,
                                },
                            )
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                model = fit_classifier(name, train_x, train_y, seed=protocol["seed"])
                            joblib.dump(model, job / "model.joblib")
                            restored = joblib.load(job / "model.joblib")
                            val_x = partitions["validation"][0]
                            np.testing.assert_allclose(
                                model.predict_proba(val_x.iloc[:128]),
                                restored.predict_proba(val_x.iloc[:128]),
                                atol=0,
                                rtol=0,
                            )
                            metrics = {}
                            for partition, (x, y, clock) in partitions.items():
                                p = restored.predict_proba(x)
                                metrics[partition] = classification_metrics(restored, p, y)
                                np.savez_compressed(
                                    job / f"{partition}_predictions.npz",
                                    probabilities=p,
                                    labels=y,
                                    decision_times=clock,
                                    train_priors=restored.priors,
                                )
                            record = {
                                "symbol": symbol,
                                "assessment_date": dev_date,
                                "validation_date": val_date,
                                "train_dates": train_dates,
                                "training_window": training_window,
                                "representation": representation,
                                "model": name,
                                "train_rows": len(train_y),
                                "feature_count": len(columns),
                                "train_class_priors": restored.priors.tolist(),
                                "checkpoint_parity": True,
                                "warnings": [str(w.message) for w in caught],
                                "seconds": time.monotonic() - started,
                                **metrics,
                            }
                            write_json(
                                completed,
                                {
                                    "identity": identity,
                                    "record": record,
                                    "artifact_hashes": {
                                        p.name: sha256_file(p) for p in job.iterdir() if p.is_file() and p != completed
                                    },
                                },
                            )
                            records.append(record)
                            write_json(output / "progress.json", {"completed_fits": len(records), **summarize(records)})
                            print(
                                f"fit={len(records)}/72 {symbol} {dev_date} {training_window}/{representation}/{name} "
                                f"val_BA={metrics['validation']['balanced_accuracy']:.4f} "
                                f"dev_BA={metrics['assessment']['balanced_accuracy']:.4f} seconds={record['seconds']:.1f}",
                                flush=True,
                            )
    if len(records) != 72:
        raise ValueError("Incomplete registered grid")
    write_json(
        output / "summary.json",
        {"identity": identity, "records": records, "completed_fits": len(records), **summarize(records)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/research/boundary_event_screen_20260907_coverage_revision.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_event_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for the registered 72-fit event-data development screen.")
