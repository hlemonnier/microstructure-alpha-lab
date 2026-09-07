"""Explain the complete frozen comparison without selecting additional models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_prediction_diagnostics import forecast_diagnostics

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "movement_fraction", "movement_binary_log_loss", "conditional_direction_log_loss",
    "direction_log_loss_contribution_per_decision", "three_class_log_loss_with_shared_smoothing",
    "movement_roc_auc", "conditional_direction_roc_auc",
)


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Registered diagnostic implementation changed")
    primary = ROOT / "results/boundary_confirmation_exact_labels_20260907"
    secondary = ROOT / "results/boundary_pooled_tree_confirmation_20260907"
    if not (secondary / "summary.json").exists():
        raise ValueError("Both complete procedures must remain sealed until all forty date predictions are frozen")
    procedure = json.loads((ROOT / "docs/research/boundary_pooled_tree_confirmation_20260907.json").read_text())
    dates = procedure["assessment_dates"]
    for folder in [primary, secondary]:
        identity = json.loads((folder / "frozen_study.json").read_text())
        for day in dates:
            verify_frozen_fold(folder / "dates" / day, identity)
    models = {
        "original_reference": (primary, "original_reference"),
        "matched_data_control": (primary, "matched_data_control"),
        "original_blend": (primary, "candidate"),
        "pooled_balanced_tree_blend": (secondary, "candidate"),
    }
    records, hashes = [], {}
    for day in dates:
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            common = None
            for model, (folder, filename) in models.items():
                path = folder / "dates" / day / "predictions" / f"{symbol}_{filename}.npz"
                hashes[str(path.relative_to(ROOT))] = sha256_file(path)
                with np.load(path) as saved:
                    if common is not None:
                        np.testing.assert_array_equal(saved["decision_times"], common[0])
                        np.testing.assert_array_equal(saved["labels"], common[1])
                    common = saved["decision_times"].copy(), saved["labels"].copy()
                    records.append({"model": model, "symbol": symbol, "assessment_date": day,
                                    **forecast_diagnostics(saved["probabilities"], saved["labels"])})
    means = {
        model: {field: float(np.mean([r[field] for r in records if r["model"] == model])) for field in FIELDS}
        for model in models
    }
    by_asset = {
        symbol: {
            model: {field: float(np.mean([r[field] for r in records if r["model"] == model and r["symbol"] == symbol])) for field in FIELDS}
            for model in models
        }
        for symbol in ["BTCUSDT", "ETHUSDT"]
    }
    result = {
        "protocol_sha256": sha256_file(protocol_path),
        "confirmation_summary_sha256": sha256_file(secondary / "summary.json"),
        "mean_metrics": means, "by_asset": by_asset, "records": records, "prediction_hashes": hashes,
        "evidence_status": "Prespecified explanatory diagnostics after both fixed procedures completed. No new model selection or promotion tests.",
        "aggregation": "Equal asset/date means. Mean full log loss equals mean movement log loss plus mean direction contribution per decision; multiplying separate mean movement frequency and mean conditional direction loss is not the same identity.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"complete_prediction_diagnostics {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_confirmation_diagnostics_protocol_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/boundary_confirmation_diagnostics_20260907.json")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run only after both registered confirmation procedures are complete.")
