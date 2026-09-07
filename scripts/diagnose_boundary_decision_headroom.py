"""Hindsight-only decision diagnostics: never an implementable forecast result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "hindsight_diagnostic_not_deployable":
        raise ValueError("These outcome-conditioned diagnostics cannot be candidates")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError("Registered hindsight diagnostic input changed")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    frozen = output / "frozen_diagnostic.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing hindsight diagnostics after an identity change")
    write_json(frozen, identity)
    parent = ROOT / protocol["source_run"]
    inventory = json.loads((parent / "summary.json").read_text())
    records, source_hashes, grid_scores = [], {}, []
    grid = [(b, d) for b in protocol["neutral_biases"] for d in protocol["direction_biases"]]
    for day in protocol["dates"]:
        for source in protocol["sources"]:
            for symbol in protocol["symbols"]:
                path = parent / "predictions" / day / source / f"{symbol}_beta_0.npz"
                checksum = sha256_file(path)
                if checksum != inventory["artifact_hashes"][str(path.relative_to(parent))]:
                    raise ValueError("The frozen zero-transfer forecast artifact changed")
                source_hashes[str(path.relative_to(ROOT))] = checksum
                with np.load(path) as archive:
                    p, y = archive["probabilities"], archive["labels"].astype(int)
                    pi, registered = archive["forecast_3600_priors"], archive["registered_priors"]
                empirical = np.array([(y == k).mean() for k in (-1, 0, 1)])
                methods = {"registered": registered, "causal_forecast_3600": pi, "hindsight_full_noon_class_frequencies": empirical}
                record = {"date": day, "source": source, "symbol": symbol,
                          "methods": {name: classification_metrics(SimpleNamespace(priors=value), p, y) for name, value in methods.items()}}
                scores = p / pi
                weights = 1 / (3 * np.bincount(y + 1, minlength=3)[y + 1])
                recalls = np.array([weights[np.argmax(scores * np.exp([-.5 * d, b, .5 * d]), axis=1) - 1 == y].sum() for b, d in grid])
                control = grid.index((0.0, 0.0))
                np.testing.assert_allclose(recalls[control], record["methods"]["causal_forecast_3600"]["balanced_accuracy"], rtol=0, atol=1e-14)
                best = max(range(len(grid)), key=lambda i: (recalls[i], -sum(v * v for v in grid[i])))
                record["hindsight_best_bias"] = {"balanced_accuracy": float(recalls[best]), "neutral_log_bias": grid[best][0], "direction_log_bias": grid[best][1]}
                records.append(record)
                grid_scores.append(recalls)
        print(f"hindsight_diagnostic_date={day} asset_source_panels={len(records)}", flush=True)
    scores_path = output / "hindsight_grid_scores.npz"
    np.savez_compressed(scores_path, log_bias_grid=np.array(grid), balanced_accuracies=np.array(grid_scores))
    summary = []
    for source in protocol["sources"]:
        rows = [r for r in records if r["source"] == source]
        values = {method: float(np.mean([r["methods"][method]["balanced_accuracy"] for r in rows])) for method in methods}
        values["hindsight_best_bias"] = float(np.mean([r["hindsight_best_bias"]["balanced_accuracy"] for r in rows]))
        summary.append({"source": source, "asset_dates": len(rows), "mean_balanced_accuracy": values,
                        "hindsight_bias_minus_causal": values["hindsight_best_bias"] - values["causal_forecast_3600"]})
    write_json(output / "summary.json", {"identity": identity, "evidence_status": protocol["evidence_status"], "summary": summary, "records": records,
        "source_hashes": source_hashes, "grid_score_sha256": sha256_file(scores_path), "hindsight_grid_evaluations": len(grid) * len(records),
        "substantial_gain_confirmed": False, "model_promotion_allowed": False,
        "interpretation": "Both hindsight procedures use assessment outcomes. The grid maximum is optimistically fitted to each assessed asset/date and is only a diagnostic of these score thresholds. It is not an attainable accuracy forecast, an upper bound on all models, a deployable policy, or independent evidence."})
    print(f"hindsight_diagnostic_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_decision_headroom_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_decision_headroom_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for hindsight diagnostics only; these cannot be model candidates.")
