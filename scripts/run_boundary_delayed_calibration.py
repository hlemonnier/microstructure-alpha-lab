"""Evaluate registered, causally delayed calibration on frozen model forecasts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_online import delayed_calibration

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_job(source, symbol, day):
    root = ROOT / f"results/boundary_{source['screen']}_screen_20260907" / symbol / day
    return root / source["representation"] / source["model"] if "representation" in source else root / source["model"]


def summarize(records):
    leaderboard, comparison = [], []
    for expert, rate in sorted({(r["expert"], r["learning_rate"]) for r in records}):
        rows = [r for r in records if (r["expert"], r["learning_rate"]) == (expert, rate)]
        leaderboard.append(
            {
                "expert": expert,
                "learning_rate": rate,
                "asset_folds": len(rows),
                **{
                    f"mean_development_{metric}": float(np.mean([r["assessment"][metric] for r in rows]))
                    for metric in ["balanced_accuracy", "natural_accuracy", "log_loss"]
                },
            }
        )

    def rank(record):
        return (record["validation"]["balanced_accuracy"], -record["validation"]["log_loss"])

    for symbol, day in sorted({(r["symbol"], r["assessment_date"]) for r in records}):
        rows = [r for r in records if (r["symbol"], r["assessment_date"]) == (symbol, day)]
        if len(rows) != 24:
            continue
        baseline = max([r for r in rows if r["learning_rate"] == 0], key=rank)
        best = max(rows, key=rank)
        keys = ["expert", "learning_rate", "validation", "assessment"]
        comparison.append(
            {
                "symbol": symbol,
                "assessment_date": day,
                "reference": {k: baseline[k] for k in keys},
                "challenger": {k: best[k] for k in keys},
                "balanced_accuracy_difference": best["assessment"]["balanced_accuracy"]
                - baseline["assessment"]["balanced_accuracy"],
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
    manifest_path = ROOT / "data/research/performance_pilot_20260907_window_revision/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    input_hashes = {}
    for source in protocol["sources"]:
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in ["2023-05-21", "2023-05-22", "2023-05-23"]:
                job = source_job(source, symbol, day)
                completed = json.loads((job / "completed.json").read_text())
                for partition in ["validation", "assessment"]:
                    path = job / f"{partition}_predictions.npz"
                    checksum = sha256_file(path)
                    if checksum != completed["artifact_hashes"][path.name]:
                        raise ValueError("Frozen model predictions changed")
                    input_hashes[str(path.relative_to(ROOT))] = checksum
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_prediction_hashes": input_hashes,
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_delayed_calibration.py",
                "src/lob_forge/boundary_online.py",
                "src/lob_forge/boundary_forecasts.py",
            ]
        },
    }
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this experiment and use a new output directory after source changes")
    write_json(frozen, identity)
    canonical = {}
    for session in manifest["sessions"]:
        path = Path(session["features_path"])
        if sha256_file(path) != session["feature"]["sha256"]:
            raise ValueError("Canonical label timing source changed")
        canonical[session["symbol"], session["session_date"]] = pd.read_csv(
            path, usecols=["decision_time", "label", "future_event_time"]
        ).set_index("decision_time")
    records = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for day in ["2023-05-21", "2023-05-22", "2023-05-23"]:
            for source in protocol["sources"]:
                expert = "_".join(source.values())
                job = source_job(source, symbol, day)
                source_record = json.loads((job / "completed.json").read_text())["record"]
                val_day = source_record["validation_date"]
                for rate in protocol["learning_rates"]:
                    destination = output / symbol / day / expert / str(rate)
                    completed = destination / "completed.json"
                    if completed.exists():
                        saved = json.loads(completed.read_text())
                        if saved["identity"] != identity:
                            raise ValueError("Calibration job identity mismatch")
                        for name, checksum in saved["artifact_hashes"].items():
                            if sha256_file(destination / name) != checksum:
                                raise ValueError("Calibration artifact changed")
                        records.append(saved["record"])
                        continue
                    started = time.monotonic()
                    write_json(
                        destination / "attempt_started.json",
                        {
                            "expert": expert,
                            "learning_rate": rate,
                            "assessment_date": day,
                            "validation_date": val_day,
                            "source": str(job.relative_to(ROOT)),
                        },
                    )
                    metrics, states = {}, {}
                    for partition, date_value in [("validation", val_day), ("assessment", day)]:
                        original = np.load(job / f"{partition}_predictions.npz")
                        clock, y, pi = original["decision_times"], original["labels"], original["train_priors"]
                        rows = canonical[symbol, date_value].loc[clock]
                        np.testing.assert_array_equal(rows["label"], y)
                        release = rows["future_event_time"].to_numpy(dtype=np.int64)
                        probabilities, states[partition] = delayed_calibration(
                            original["probabilities"],
                            y,
                            clock,
                            release,
                            learning_rate=rate,
                            half_life_seconds=protocol["bias_half_life_seconds"],
                            bias_limit=protocol["bias_absolute_limit"],
                        )
                        metrics[partition] = classification_metrics(SimpleNamespace(priors=pi), probabilities, y)
                        np.savez_compressed(
                            destination / f"{partition}_predictions.npz",
                            probabilities=probabilities,
                            labels=y,
                            decision_times=clock,
                            train_priors=pi,
                            label_available_times=release,
                        )
                    record = {
                        "symbol": symbol,
                        "assessment_date": day,
                        "validation_date": val_day,
                        "expert": expert,
                        "learning_rate": rate,
                        "states": states,
                        "seconds": time.monotonic() - started,
                        **metrics,
                    }
                    write_json(
                        completed,
                        {
                            "identity": identity,
                            "record": record,
                            "artifact_hashes": {
                                path.name: sha256_file(path)
                                for path in destination.iterdir()
                                if path.is_file() and path != completed
                            },
                        },
                    )
                    records.append(record)
                    write_json(output / "progress.json", {"completed_trials": len(records), **summarize(records)})
                    print(
                        f"trial={len(records)}/144 {symbol} {day} {expert} rate={rate} "
                        f"val_BA={metrics['validation']['balanced_accuracy']:.4f} "
                        f"dev_BA={metrics['assessment']['balanced_accuracy']:.4f}",
                        flush=True,
                    )
    if len(records) != protocol["planned_trials"]:
        raise ValueError("Incomplete registered calibration grid")
    write_json(
        output / "summary.json",
        {"identity": identity, "completed_trials": len(records), "records": records, **summarize(records)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/research/boundary_delayed_calibration_20260907.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_delayed_calibration_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for 144 registered delayed-calibration development trials.")
