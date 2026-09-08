"""Inspect a fixed completed model library without fitting or changing forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_error_geometry import error_geometry
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "descriptive_only_on_exposed_dates" or protocol["model_fits"] != 0:
        raise ValueError("This is only the frozen completed-library diagnostic")
    if output.exists():
        raise ValueError("Preserve every earlier error-geometry diagnostic")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen diagnostic input changed: {path}")
    parent = ROOT / protocol["source_run"]
    identity = json.loads((parent / "frozen_screen.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if summary["identity"] != identity or summary["substantial_gain_confirmed"] is not False:
        raise ValueError("Only completed development evidence is eligible")
    names = protocol["models"]
    reference = names.index(protocol["reference"])
    members = [names.index(name) for name in protocol["equal_probability_members"]]
    if len(names) != 9 or len(set(names)) != 9 or len(members) != 8 or reference in members:
        raise ValueError("The fixed eight individual experts and reference must remain")
    records = []
    for day in protocol["dates"]:
        folder = parent / "dates" / day
        check_completed(folder, identity)
        for symbol in protocol["symbols"]:
            probabilities, decisions, labels, clocks, expected_ba = [], [], None, None, []
            for name in names:
                path = folder / "predictions" / f"{symbol}_{name}_{protocol['decision_policy']}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    if labels is None:
                        labels, clocks = saved["labels"].copy(), saved["decision_times"].copy()
                    np.testing.assert_array_equal(labels, saved["labels"])
                    np.testing.assert_array_equal(clocks, saved["decision_times"])
                    p, prior = saved["probabilities"].copy(), saved["decision_priors"].copy()
                if (prior.shape not in ((3,), p.shape) or not np.isfinite(prior).all() or (prior <= 0).any()
                    or not np.allclose(prior.sum(axis=-1), 1, rtol=0, atol=1e-12)):
                    raise ValueError("Every frozen decision prior must preserve the original probability contract")
                probabilities.append(p)
                decisions.append(np.argmax(p / prior, axis=1) - 1)
                match = [r for r in summary["records"] if (r["date"], r["symbol"], r["model"], r["policy"])
                    == (day, symbol, name, protocol["decision_policy"])]
                if len(match) != 1:
                    raise ValueError("Every original source metric must exist exactly once")
                expected_ba.append(match[0]["metrics"]["balanced_accuracy"])
            if len(labels) != 7070:
                raise ValueError("Every original assessment row must remain")
            geometry = error_geometry(np.stack(probabilities), np.stack(decisions), labels, reference=reference, ensemble_members=members)
            np.testing.assert_allclose(geometry["balanced_accuracy"], expected_ba, rtol=0, atol=1e-12)
            records.append({"date": day, "symbol": symbol, "geometry": geometry})
    if len(records) != 6:
        raise ValueError("All six frozen asset/date diagnostic panels required")
    scalars = ("hindsight_any_correct_balanced_accuracy", "hindsight_rescuable_reference_error_balanced_mass",
        "all_wrong_balanced_mass", "unanimous_balanced_mass", "unanimous_wrong_balanced_mass")
    vectors = ("balanced_accuracy", "rescue_balanced_mass", "harm_balanced_mass", "correct_expert_count_balanced_mass")
    def aggregate(rows):
        return {"asset_date_panels": len(rows),
            **{key: float(np.mean([r["geometry"][key] for r in rows])) for key in scalars},
            **{key: np.mean([r["geometry"][key] for r in rows], axis=0).tolist() for key in vectors},
            "equal_expert_brier": {key: float(np.mean([r["geometry"]["equal_expert_brier"][key] for r in rows]))
                for key in ("average_individual_brier", "mean_probability_brier", "ambiguity")}}
    result = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "evidence_status": protocol["evidence_status"], "model_fits": 0, "predictive_gain_confirmed": False,
        "models": names, "reference": protocol["reference"], "equal_probability_members": protocol["equal_probability_members"],
        "assessment_rows_per_source": sum(r["geometry"]["rows"] for r in records),
        "all_original_rows_labels_clocks_and_metrics_reproduced": True,
        "hindsight_ceiling_scope": protocol["hindsight_ceiling_scope"],
        "means": aggregate(records), "by_asset": {symbol: aggregate([r for r in records if r["symbol"] == symbol]) for symbol in protocol["symbols"]},
        "by_date": {day: aggregate([r for r in records if r["date"] == day]) for day in protocol["dates"]}, "records": records}
    write_json(output / "summary.json", result)
    print(f"error_geometry_diagnostic_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())
