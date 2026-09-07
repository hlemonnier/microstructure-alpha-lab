"""Measure registered probability mixtures on explicitly reused development dates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def run():
    protocol_path = ROOT / "docs/research/boundary_probability_blend_20260907.json"
    protocol = json.loads(protocol_path.read_text())
    output = ROOT / "results/boundary_probability_blend_20260907"
    if output.exists():
        raise ValueError("Preserve the existing blend experiment")
    output.mkdir(parents=True)
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "metric_code_sha256": sha256_file(ROOT / "src/lob_forge/boundary_forecasts.py"),
    }
    (output / "frozen_screen.json").write_text(json.dumps(identity, indent=2) + "\n")
    records = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for day in ["2023-05-21", "2023-05-22", "2023-05-23"]:
            sources = {}
            source_hashes = {}
            for partition in ["validation", "assessment"]:
                base = ROOT / "results/boundary_prior_decisions_20260907" / symbol / day
                paths = [
                    base / name / "blend_training_previous" / f"{partition}_predictions.npz"
                    for name in ["pooled_mlp", "full_event_7"]
                ]
                for path in paths:
                    completed = json.loads((path.parent / "completed.json").read_text())
                    checksum = sha256_file(path)
                    if checksum != completed["artifact_hashes"][path.name]:
                        raise ValueError("Frozen prior-policy inputs changed")
                    source_hashes[str(path.relative_to(ROOT))] = checksum
                neural, tree = [dict(np.load(path)) for path in paths]
                for key in ["labels", "decision_times", "train_priors", "decision_priors"]:
                    np.testing.assert_array_equal(neural[key], tree[key])
                sources[partition] = neural, tree
            for weight in protocol["neural_weights"]:
                job = output / symbol / day / str(weight)
                job.mkdir(parents=True)
                metrics, hashes = {}, {}
                for partition, (neural, tree) in sources.items():
                    p = weight * neural["probabilities"] + (1 - weight) * tree["probabilities"]
                    metrics[partition] = classification_metrics(
                        SimpleNamespace(priors=neural["decision_priors"]), p, neural["labels"]
                    )
                    path = job / f"{partition}_predictions.npz"
                    np.savez_compressed(path, **{**neural, "probabilities": p})
                    hashes[path.name] = sha256_file(path)
                record = {"symbol": symbol, "assessment_date": day, "neural_weight": weight, **metrics}
                (job / "completed.json").write_text(
                    json.dumps(
                        {
                            "identity": identity,
                            "source_hashes": source_hashes,
                            "record": record,
                            "artifact_hashes": hashes,
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
                records.append(record)
    leaderboard = []
    for weight in protocol["neural_weights"]:
        rows = [r for r in records if r["neural_weight"] == weight]
        leaderboard.append(
            {
                "neural_weight": weight,
                **{
                    f"mean_development_{metric}": float(np.mean([r["assessment"][metric] for r in rows]))
                    for metric in ["balanced_accuracy", "natural_accuracy", "log_loss"]
                },
            }
        )
    leaderboard.sort(
        key=lambda r: (r["mean_development_balanced_accuracy"], -r["mean_development_log_loss"]), reverse=True
    )
    result = {
        "identity": identity,
        "records": records,
        "leaderboard": leaderboard,
        "completed_trials": len(records),
        "evidence_status": "development_only_previously_exposed_dates",
        "substantial_gain_confirmed": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    run()
