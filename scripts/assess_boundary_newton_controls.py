"""Add the preregistered strongest depth comparator without changing any fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered comparator input changed: {path}")
    parents = [ROOT / protocol[k] for k in ("source", "newton_run")]
    for parent in parents:
        identity = json.loads((parent / "frozen_screen.json").read_text())
        for day in protocol["dates"]:
            check_completed(parent / "dates" / day, identity)
    summary_path = parents[1] / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["model_fits"] != 21 or summary["post_fit_panels"] != 360:
        raise ValueError("The complete original Newton family must precede this comparator")
    control_records = []
    for day in protocol["dates"]:
        for symbol in ("BTCUSDT", "ETHUSDT"):
            for policy in protocol["policies"]:
                file = f"{symbol}_{protocol['control']}_{policy}.npz"
                reference = f"{symbol}_original_reference_{policy}.npz"
                with np.load(parents[0] / "dates" / day / "predictions" / file) as data:
                    with np.load(parents[1] / "dates" / day / "predictions" / reference) as other:
                        np.testing.assert_array_equal(data["labels"], other["labels"])
                        np.testing.assert_array_equal(data["decision_times"], other["decision_times"])
                    metrics = classification_metrics(SimpleNamespace(priors=data["decision_priors"]), data["probabilities"], data["labels"])
                control_records.append({"date": day, "symbol": symbol, "model": protocol["control"], "policy": policy, "metrics": metrics})
    if len(control_records) != protocol["additional_panels"]:
        raise ValueError("Incomplete registered comparator")
    index = {(r["date"], r["symbol"], r["policy"]): r["metrics"] for r in control_records}
    paired = []
    for row in summary["records"]:
        control = index[row["date"], row["symbol"], row["policy"]]
        paired.append({**{k: row[k] for k in ("date", "symbol", "model", "policy")},
                       "balanced_accuracy_delta": row["metrics"]["balanced_accuracy"] - control["balanced_accuracy"]})
    means = []
    for model, policy in sorted({(r["model"], r["policy"]) for r in paired}):
        rows = [r for r in paired if (r["model"], r["policy"]) == (model, policy)]
        means.append({"model": model, "policy": policy,
                      "balanced_accuracy_delta_to_depth_control": float(np.mean([r["balanced_accuracy_delta"] for r in rows])),
                      "delta_by_asset": {s: float(np.mean([r["balanced_accuracy_delta"] for r in rows if r["symbol"] == s])) for s in ("BTCUSDT", "ETHUSDT")}})
    result = {"protocol_sha256": sha256_file(protocol_path), "script_sha256": sha256_file(Path(__file__)),
              "parent_summary_path": str(summary_path.relative_to(ROOT)), "parent_summary_sha256": sha256_file(summary_path),
              "evidence_status": protocol["evidence_status"], "new_model_fits": 0,
              "additional_panels": len(control_records), "total_panels_with_parent": 360 + len(control_records),
              "substantial_gain_confirmed": False, "control_records": control_records,
              "means": means, "paired_deltas": paired}
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("Preserve the original comparator result after any input change")
    write_json(output, result)
    print(f"newton_matched_controls_complete {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_newton_control_extension_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_newton_control_extension_20260907/summary.json")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run only after the complete Newton family finishes.")
