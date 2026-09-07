"""Test available class-prior estimates while preserving source model probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_prior_policy import decision_priors

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCES = {"original_reference", "full_base_7", "full_base_31"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_location(source, symbol, day, references):
    if source == "original_reference":
        reference = next(r["reference"] for r in references if (r["symbol"], r["assessment_date"]) == (symbol, day))
        job = (
            ROOT
            / "results/boundary_predictive_screen_20260907"
            / symbol
            / day
            / reference["representation"]
            / reference["model"]
        )
        prefix = ""
    elif source.startswith("full_"):
        _, representation, leaves = source.split("_")
        representation = {"base": "base", "event": "event_cross", "temporal": "temporal_cross"}[representation]
        job = (
            ROOT
            / "results/boundary_event_screen_20260907"
            / symbol
            / day
            / "full_day"
            / representation
            / f"hgb_{leaves}"
        )
        prefix = ""
    else:
        job = ROOT / "results/boundary_pooled_neural_20260907" / day / source / "BTCUSDT_ETHUSDT"
        prefix = symbol + "_"
    saved = json.loads((job / "completed.json").read_text())
    record = saved["record"] if "record" in saved else next(r for r in saved["records"] if r["symbol"] == symbol)
    return job, prefix, saved, record


def summarize(records):
    leaderboard, comparison = [], []

    def rank(record):
        return (record["validation"]["balanced_accuracy"], -record["validation"]["log_loss"])

    for source, policy in sorted({(r["source"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["source"], r["policy"]) == (source, policy)]
        leaderboard.append(
            {
                "source": source,
                "policy": policy,
                "asset_folds": len(rows),
                **{
                    f"mean_development_{metric}": float(np.mean([r["assessment"][metric] for r in rows]))
                    for metric in ["balanced_accuracy", "natural_accuracy", "log_loss"]
                },
            }
        )
    for symbol, day in sorted({(r["symbol"], r["assessment_date"]) for r in records}):
        rows = [r for r in records if (r["symbol"], r["assessment_date"]) == (symbol, day)]
        if len(rows) != 48:
            continue
        original = next(r for r in rows if r["source"] == "original_reference" and r["policy"] == "training")
        stronger = max([r for r in rows if r["source"] in REFERENCE_SOURCES], key=rank)
        best = max([r for r in rows if r["source"] not in REFERENCE_SOURCES], key=rank)
        keys = ["source", "policy", "validation", "assessment"]
        comparison.append(
            {
                "symbol": symbol,
                "assessment_date": day,
                "original_reference": {k: original[k] for k in keys},
                "stronger_reference": {k: stronger[k] for k in keys},
                "challenger": {k: best[k] for k in keys},
                "difference_to_original": best["assessment"]["balanced_accuracy"]
                - original["assessment"]["balanced_accuracy"],
                "difference_to_stronger": best["assessment"]["balanced_accuracy"]
                - stronger["assessment"]["balanced_accuracy"],
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
    reference_path = ROOT / "results/boundary_predictive_screen_20260907/summary.json"
    references = json.loads(reference_path.read_text())["validation_selected_comparison"]
    manifest_path = ROOT / "data/research/performance_pilot_20260907_window_revision/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs, source_hashes, canonical = {}, {}, {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for day in ["2023-05-21", "2023-05-22", "2023-05-23"]:
            for source in protocol["sources"]:
                job, prefix, saved, record = source_location(source, symbol, day, references)
                loaded = {}
                for partition in ["validation", "assessment"]:
                    path = job / f"{prefix}{partition}_predictions.npz"
                    checksum = sha256_file(path)
                    if checksum != saved["artifact_hashes"][path.name]:
                        raise ValueError("Frozen source predictions changed")
                    source_hashes[str(path.relative_to(ROOT))] = checksum
                    with np.load(path) as archive:
                        loaded[partition] = {key: archive[key] for key in archive.files}
                inputs[symbol, day, source] = (record, loaded)
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "reference_sha256": sha256_file(reference_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_prediction_hashes": source_hashes,
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_prior_decisions.py",
                "src/lob_forge/boundary_prior_policy.py",
                "src/lob_forge/boundary_forecasts.py",
            ]
        },
    }
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this experiment and use a new output directory after source changes")
    write_json(frozen, identity)
    previous_frequencies = {}
    for session in manifest["sessions"]:
        path = Path(session["features_path"])
        if sha256_file(path) != session["feature"]["sha256"]:
            raise ValueError("Canonical label timing source changed")
        day = session["session_date"]
        frame = pd.read_csv(path, usecols=["decision_time", "label", "future_event_time"]).set_index("decision_time")
        frame = frame.loc[(frame.index >= utc_ms(day, "12:02:00")) & (frame.index < utc_ms(day, "13:59:50"))]
        canonical[session["symbol"], day] = frame
        previous_frequencies[session["symbol"], day] = np.array([(frame.label == label).mean() for label in [-1, 0, 1]])
    records = []
    for (symbol, day, source), (source_record, partitions) in inputs.items():
        validation_day, previous_day = source_record["validation_date"], source_record["train_dates"][-1]
        for policy in protocol["policies"]:
            job = output / symbol / day / source / policy
            completed = job / "completed.json"
            if completed.exists():
                saved = json.loads(completed.read_text())
                if saved["identity"] != identity:
                    raise ValueError("Prior-policy job identity mismatch")
                for name, checksum in saved["artifact_hashes"].items():
                    if sha256_file(job / name) != checksum:
                        raise ValueError("Prior-policy artifact changed")
                records.append(saved["record"])
                continue
            write_json(
                job / "attempt_started.json",
                {
                    "source": source,
                    "policy": policy,
                    "symbol": symbol,
                    "assessment_date": day,
                    "validation_date": validation_day,
                },
            )
            metrics = {}
            for partition, current_day, past_day in [
                ("validation", validation_day, previous_day),
                ("assessment", day, validation_day),
            ]:
                original = partitions[partition]
                p, y, clock, train_pi = (
                    original[key] for key in ["probabilities", "labels", "decision_times", "train_priors"]
                )
                rows = canonical[symbol, current_day].loc[clock]
                np.testing.assert_array_equal(rows.label, y)
                estimated = decision_priors(
                    policy,
                    train_pi,
                    previous_frequencies[symbol, past_day],
                    y,
                    clock,
                    rows.future_event_time.to_numpy(dtype=np.int64),
                )
                metrics[partition] = classification_metrics(SimpleNamespace(priors=estimated), p, y)
                for unchanged in ["natural_accuracy", "log_loss"]:
                    if not np.isclose(
                        metrics[partition][unchanged], source_record[partition][unchanged], rtol=0, atol=1e-14
                    ):
                        raise ValueError("A decision-only policy changed source posterior metrics")
                np.savez_compressed(
                    job / f"{partition}_predictions.npz",
                    probabilities=p,
                    labels=y,
                    decision_times=clock,
                    train_priors=train_pi,
                    decision_priors=estimated,
                )
            record = {
                "symbol": symbol,
                "assessment_date": day,
                "validation_date": validation_day,
                "source": source,
                "policy": policy,
                "probabilities_unchanged": True,
                **metrics,
            }
            write_json(
                completed,
                {
                    "identity": identity,
                    "record": record,
                    "artifact_hashes": {
                        path.name: sha256_file(path) for path in job.iterdir() if path.is_file() and path != completed
                    },
                },
            )
            records.append(record)
            write_json(output / "progress.json", {"completed_trials": len(records), **summarize(records)})
            print(
                f"trial={len(records)}/288 {symbol} {day} {source}/{policy} "
                f"val_BA={metrics['validation']['balanced_accuracy']:.4f} dev_BA={metrics['assessment']['balanced_accuracy']:.4f}",
                flush=True,
            )
    if len(records) != protocol["planned_trials"]:
        raise ValueError("Incomplete registered prior-policy grid")
    write_json(
        output / "summary.json",
        {"identity": identity, "records": records, "completed_trials": len(records), **summarize(records)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_prior_decisions_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_prior_decisions_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for 288 registered class-prior decision-policy trials.")
