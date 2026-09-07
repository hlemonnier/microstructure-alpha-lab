"""Development test of simultaneous cross-asset directional forecast information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_direction_transfer import STRENGTHS, transfer_direction
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_prediction_diagnostics import forecast_diagnostics
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["strengths"] != list(STRENGTHS) or protocol["evidence_status"] != "development_only_on_exposed_dates":
        raise ValueError("Use the fixed exposed-date transfer family")
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered transfer input changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this direction-transfer experiment after an identity change")
    write_json(frozen, identity)
    records, artifacts, source_hashes = [], {}, {}
    for day in protocol["dates"]:
        for source in protocol["sources"]:
            root = ROOT / source["run"]
            folder = root / "dates" / day
            verify_frozen_fold(folder, json.loads((root / "frozen_study.json").read_text()))
            data = {}
            for symbol in SYMBOLS:
                path = folder / "predictions" / f"{symbol}_{source['prediction']}.npz"
                source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
                with np.load(path) as archive:
                    data[symbol] = {k: archive[k].copy() for k in archive.files}
            np.testing.assert_array_equal(data[SYMBOLS[0]]["decision_times"], data[SYMBOLS[1]]["decision_times"])
            for symbol, peer in ((SYMBOLS[0], SYMBOLS[1]), (SYMBOLS[1], SYMBOLS[0])):
                own = data[symbol]
                prior_path = ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz"
                if sha256_file(prior_path) != protocol["input_hashes"][str(prior_path.relative_to(ROOT))]:
                    raise ValueError("Available-label initialization changed")
                with np.load(prior_path) as archive:
                    np.testing.assert_array_equal(archive["decision_times"], own["decision_times"])
                    initial = archive["labels_cumulative"][0].copy()
                baseline_diagnostics = forecast_diagnostics(own["probabilities"], own["labels"])
                for strength in STRENGTHS:
                    p = transfer_direction(own["probabilities"], data[peer]["probabilities"], strength=strength)
                    diagnostics = forecast_diagnostics(p, own["labels"])
                    np.testing.assert_allclose(diagnostics["movement_binary_log_loss"], baseline_diagnostics["movement_binary_log_loss"], rtol=0, atol=1e-14)
                    policies = {"registered": own["decision_priors"], "forecast_3600": forecast_priors(p, own["decision_times"], initial, half_life_seconds=3600)}
                    path = output / "predictions" / day / source["name"] / f"{symbol}_beta_{strength:g}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, probabilities=p, labels=own["labels"], decision_times=own["decision_times"], **{name + "_priors": value for name, value in policies.items()})
                    artifacts[str(path.relative_to(output))] = sha256_file(path)
                    for policy, priors in policies.items():
                        metrics = classification_metrics(SimpleNamespace(priors=priors), p, own["labels"])
                        records.append({"date": day, "symbol": symbol, "source": source["name"], "strength": strength, "policy": policy, "metrics": metrics, "conditional_direction_roc_auc": diagnostics["conditional_direction_roc_auc"], "conditional_direction_log_loss": diagnostics["conditional_direction_log_loss"]})
        print(f"direction_transfer_development_date={day} post_fit_panels={len(records)}", flush=True)
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete direction-transfer family")
    index = {(r["date"], r["symbol"], r["source"], r["strength"], r["policy"]): r for r in records}
    leaderboard = []
    for source in protocol["sources"]:
        for strength in STRENGTHS:
            for policy in ("registered", "forecast_3600"):
                rows = [r for r in records if (r["source"], r["strength"], r["policy"]) == (source["name"], strength, policy)]
                leaderboard.append({
                    "source": source["name"], "strength": strength, "policy": policy, "asset_dates": len(rows),
                    "means": {key: float(np.mean([r["metrics"][key] for r in rows])) for key in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
                    "mean_conditional_direction_roc_auc": float(np.mean([r["conditional_direction_roc_auc"] for r in rows])),
                    "mean_delta_to_same_source_zero_strength": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], r["source"], 0.0, policy]["metrics"]["balanced_accuracy"] for r in rows])),
                    "delta_to_zero_strength_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], s, r["source"], 0.0, policy]["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS},
                })
    write_json(output / "summary.json", {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": 0, "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records, "source_hashes": source_hashes, "artifact_hashes": artifacts})
    print(f"direction_transfer_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_direction_transfer_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_direction_transfer_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for the fixed cross-asset direction-transfer family on exposed dates.")
