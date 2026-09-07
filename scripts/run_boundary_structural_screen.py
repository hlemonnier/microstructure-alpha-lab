"""Run the registered movement/direction and reflection development screen."""

from __future__ import annotations

import argparse
import importlib.util
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
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_structural import fit_hurdle, fit_symmetric_classifier

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("registered_screen", ROOT / "scripts/run_boundary_predictive_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def summarize(records, references):
    comparison, leaderboard = [], []
    for name in sorted({r["model"] for r in records}):
        rows = [r for r in records if r["model"] == name]
        leaderboard.append(
            {
                "model": name,
                "completed_asset_folds": len(rows),
                **{
                    f"mean_development_{metric}": float(np.mean([r["assessment"][metric] for r in rows]))
                    for metric in ["balanced_accuracy", "natural_accuracy", "log_loss"]
                },
            }
        )
    for reference in references:
        rows = [
            r
            for r in records
            if (r["symbol"], r["assessment_date"]) == (reference["symbol"], reference["assessment_date"])
        ]
        if len(rows) != 5:
            continue
        best = max(rows, key=lambda r: (r["validation"]["balanced_accuracy"], -r["validation"]["log_loss"]))
        comparison.append(
            {
                "symbol": best["symbol"],
                "assessment_date": best["assessment_date"],
                "reference": reference["reference"],
                "challenger": {key: best[key] for key in ["model", "validation", "assessment"]},
                "balanced_accuracy_difference": best["assessment"]["balanced_accuracy"]
                - reference["reference"]["assessment"]["balanced_accuracy"],
            }
        )
    return {
        "leaderboard": sorted(leaderboard, key=lambda r: r["mean_development_balanced_accuracy"], reverse=True),
        "validation_selected_comparison": comparison,
        "evidence_status": "development_only_previously_exposed_dates",
        "substantial_gain_confirmed": False,
    }


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    source_protocol = ROOT / "docs/research/boundary_predictive_screen_20260907.json"
    reference_path = ROOT / "results/boundary_predictive_screen_20260907/summary.json"
    reference = json.loads(reference_path.read_text())["validation_selected_comparison"]
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "source_protocol_sha256": sha256_file(source_protocol),
        "reference_sha256": sha256_file(reference_path),
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_structural_screen.py",
                "scripts/run_boundary_predictive_screen.py",
                "src/lob_forge/boundary_forecasts.py",
                "src/lob_forge/boundary_structural.py",
            ]
        },
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
    }
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Source changed: preserve this experiment and use a new output directory")
    screen.write_json(frozen, identity)
    data_protocol = json.loads(source_protocol.read_text())
    data_protocol["first_tabular_grid"]["representations"] = ["temporal_cross"]
    features, labels, times = screen.load_frames(data_protocol)
    dates = sorted({day for symbol, day in labels})
    records = []
    with threadpool_limits(limits=2):
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for index in range(5, len(dates)):
                train_dates, validation_date, assessment_date = (
                    dates[index - 5 : index - 1],
                    dates[index - 1],
                    dates[index],
                )
                train_x = pd.concat([features[symbol, day, "temporal_cross"] for day in train_dates], ignore_index=True)
                train_y = np.concatenate([labels[symbol, day] for day in train_dates])
                val_x = features[symbol, validation_date, "temporal_cross"]
                for name in protocol["variants"]:
                    job = output / symbol / assessment_date / name
                    completed = job / "completed.json"
                    if completed.exists():
                        saved = json.loads(completed.read_text())
                        if saved["run_identity"] != screen.identity_hash(identity):
                            raise ValueError("Run identity mismatch")
                        for filename, checksum in saved["artifact_hashes"].items():
                            if sha256_file(job / filename) != checksum:
                                raise ValueError("Artifact checksum mismatch")
                        records.append(saved["record"])
                        continue
                    started = time.monotonic()
                    screen.write_json(
                        job / "attempt_started.json",
                        {
                            "identity": identity,
                            "model": name,
                            "train_dates": train_dates,
                            "validation_date": validation_date,
                            "assessment_date": assessment_date,
                        },
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model = (
                            fit_symmetric_classifier(train_x, train_y, seed=protocol["seed"])
                            if name == "symmetric_hgb"
                            else fit_hurdle(name, train_x, train_y, seed=protocol["seed"])
                        )
                    joblib.dump(model, job / "model.joblib")
                    restored = joblib.load(job / "model.joblib")
                    np.testing.assert_allclose(
                        model.predict_proba(val_x.iloc[:128]),
                        restored.predict_proba(val_x.iloc[:128]),
                        rtol=0,
                        atol=1e-14,
                    )
                    metrics = {}
                    for partition, day in [("validation", validation_date), ("assessment", assessment_date)]:
                        p = restored.predict_proba(features[symbol, day, "temporal_cross"])
                        metrics[partition] = classification_metrics(restored, p, labels[symbol, day])
                        np.savez_compressed(
                            job / f"{partition}_predictions.npz",
                            probabilities=p,
                            labels=labels[symbol, day],
                            decision_times=times[symbol, day],
                            train_priors=restored.priors,
                        )
                    record = {
                        "symbol": symbol,
                        "model": name,
                        "train_dates": train_dates,
                        "validation_date": validation_date,
                        "assessment_date": assessment_date,
                        "train_rows": len(train_y),
                        **metrics,
                        "train_class_priors": restored.priors.tolist(),
                        "checkpoint_parity": True,
                        "warnings": [str(w.message) for w in caught],
                        "seconds": time.monotonic() - started,
                    }
                    screen.write_json(
                        completed,
                        {
                            "run_identity": screen.identity_hash(identity),
                            "record": record,
                            "artifact_hashes": {
                                p.name: sha256_file(p) for p in job.iterdir() if p.is_file() and p != completed
                            },
                        },
                    )
                    records.append(record)
                    screen.write_json(
                        output / "progress.json", {"completed_fits": len(records), **summarize(records, reference)}
                    )
                    print(
                        f"fit={len(records)}/30 {symbol} {assessment_date} {name} "
                        f"val_BA={metrics['validation']['balanced_accuracy']:.4f} "
                        f"dev_BA={metrics['assessment']['balanced_accuracy']:.4f} seconds={record['seconds']:.1f}",
                        flush=True,
                    )
    if len(records) != 30:
        raise ValueError("Incomplete registered experiment")
    screen.write_json(
        output / "summary.json",
        {"identity": identity, "completed_fits": len(records), "records": records, **summarize(records, reference)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/research/boundary_structural_screen_20260907.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_structural_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run to execute the registered 30-fit development screen.")
