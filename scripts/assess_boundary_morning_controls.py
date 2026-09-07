"""Separate same-day data effects from fine-tuning in the morning development study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_forecasts import classification_metrics
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_morning_screen import BLENDS, NEURAL_FITS, SYMBOLS, WINDOWS

ROOT = Path(__file__).resolve().parents[1]


def assess(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered matched-control input changed: {name}")
    parent_protocol = json.loads((ROOT / protocol["parent_protocol"]).read_text())
    parent = ROOT / protocol["parent_output"]
    identity = json.loads((parent / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(ROOT / protocol["parent_protocol"]):
        raise ValueError("The completed morning procedure does not match the fixed protocol")
    summary_path = parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["identity"] != identity or summary["post_fit_panels"] != 600:
        raise ValueError("The full morning development family must finish first")
    records = []
    source_hashes, artifact_hashes = {}, {}
    for day in parent_protocol["dates"]:
        folder = parent / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            prior_path = ROOT / parent_protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz"
            if sha256_file(prior_path) != parent_protocol["input_hashes"][str(prior_path.relative_to(ROOT))]:
                raise ValueError("The available-label initialization changed")
            with np.load(prior_path) as archive:
                initial, expected_times = archive["labels_cumulative"][0].copy(), archive["decision_times"].copy()
            for window in WINDOWS:
                name = "old_neural_" + window + "_tree_blend"
                for policy in parent_protocol["policies"]:
                    neural_path = folder / "predictions" / f"{symbol}_old_neural_{policy}.npz"
                    tree_path = folder / "predictions" / f"{symbol}_{window}_tree_{policy}.npz"
                    with np.load(neural_path) as neural, np.load(tree_path) as tree:
                        np.testing.assert_array_equal(neural["labels"], tree["labels"])
                        np.testing.assert_array_equal(neural["decision_times"], tree["decision_times"])
                        np.testing.assert_array_equal(neural["decision_times"], expected_times)
                        p = (neural["probabilities"] + tree["probabilities"]) / 2
                        times, y = neural["decision_times"].copy(), neural["labels"].copy()
                        priors = neural["decision_priors"].copy() if policy != "forecast_3600" else forecast_priors(p, times, initial, half_life_seconds=3600)
                    metrics = classification_metrics(SimpleNamespace(priors=priors), p, y)
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
                    destination = output / day / f"{symbol}_{name}_{policy}.npz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination, probabilities=p, labels=y, decision_times=times, decision_priors=priors)
                    artifact_hashes[str(destination.relative_to(output))] = sha256_file(destination)
                    for path in (neural_path, tree_path):
                        source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    index = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in [*summary["records"], *records]}
    comparisons = []
    for name in [*NEURAL_FITS, *BLENDS]:
        neural_name = name if name in NEURAL_FITS else BLENDS[name][0]
        window = NEURAL_FITS[neural_name][0]
        control = "old_neural" if name in NEURAL_FITS else "original_blend" if name.endswith("_old_tree_blend") else "old_neural_" + window + "_tree_blend"
        for policy in parent_protocol["policies"]:
            pairs = [(index[d, s, name, policy], index[d, s, control, policy]) for d in parent_protocol["dates"] for s in SYMBOLS]
            comparisons.append({
                "model": name, "matched_control": control, "policy": policy, "asset_dates": len(pairs),
                "mean_finetuning_delta": {k: float(np.mean([a["metrics"][k] - b["metrics"][k] for a, b in pairs])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss")},
                "balanced_accuracy_delta_by_asset": {s: float(np.mean([a["metrics"]["balanced_accuracy"] - b["metrics"]["balanced_accuracy"] for a, b in pairs if a["symbol"] == s])) for s in SYMBOLS},
            })
    if len(records) != protocol["additional_post_fit_asset_date_panels"]:
        raise ValueError("Incomplete matched-control extension")
    write_json(output / "summary.json", {
        "protocol_sha256": sha256_file(protocol_path), "parent_summary_sha256": sha256_file(summary_path),
        "evidence_status": "development_only_on_exposed_dates", "additional_model_fits": 0, "additional_post_fit_panels": len(records),
        "substantial_gain_confirmed": False, "records": records, "matched_finetuning_comparisons": comparisons,
        "source_hashes": source_hashes, "artifact_hashes": artifact_hashes,
    })
    print(f"morning_matched_controls_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_morning_controls_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_morning_controls_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        assess(args.protocol, args.output)
    else:
        print("Pass --run only after all frozen morning development predictions are complete.")
