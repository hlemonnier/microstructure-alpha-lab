"""Freeze matched conversion representations after their raw-source control study."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_midpoint_conversion_screen import BASELINES, BLENDS, MODEL_SPECS

ROOT = Path(__file__).resolve().parents[1]


def freeze():
    docs = ROOT / "docs/research"
    destination = docs / "boundary_midpoint_conversion_screen_20260909.json"
    if destination.exists():
        raise ValueError("Preserve every midpoint-conversion model freeze")
    inputs = {}

    def pin(path, digest=None):
        path = Path(path)
        absolute = path if path.is_absolute() else ROOT / path
        key = str(absolute.relative_to(ROOT))
        actual = sha256_file(absolute)
        if (digest is not None and digest != actual) or (key in inputs and inputs[key] != actual):
            raise ValueError(f"A frozen midpoint-model prerequisite changed: {key}")
        inputs[key] = actual
        return actual

    parent_path = docs / "boundary_quote_currency_screen_20260909.json"
    parent = json.loads(parent_path.read_text())
    parent_sha = pin(parent_path)
    for path, digest in parent["input_hashes"].items():
        pin(path, digest)
    quote_path = ROOT / "results/boundary_quote_currency_screen_20260909"
    quote_summary = json.loads((quote_path / "summary.json").read_text())
    quote_identity = json.loads((quote_path / "frozen_screen.json").read_text())
    quote_evidence_path = docs / "boundary_quote_currency_evidence_20260909.json"
    quote_evidence = json.loads(quote_evidence_path.read_text())
    pin(quote_evidence_path)
    if (quote_summary["identity"] != quote_identity or quote_identity["protocol_sha256"] != parent_sha
        or quote_evidence["reporting_panels_recomputed_exactly"] != 456
        or quote_evidence["summary_sha256"] != pin(quote_path / "summary.json")
        or quote_evidence["substantial_gain_confirmed"]):
        raise ValueError("Require the complete verified original quote-source study")
    pin(quote_path / "frozen_screen.json")
    strongest_quote = quote_evidence["primary_leaderboard"][0]["model"]
    for day in parent["dates"]:
        folder = quote_path / "dates" / day
        check_completed(folder, quote_identity)
        pin(folder / "completed.json")
        prepared = folder / "prepared"
        frozen = json.loads((prepared / "prepared.json").read_text())
        if frozen["identity"] != quote_identity:
            raise ValueError("The original selected-row matrices changed identity")
        pin(prepared / "prepared.json")
        for path, digest in frozen["artifact_hashes"].items():
            pin(prepared / path, digest)
        for symbol in SYMBOLS:
            pin(folder / "predictions" / f"{symbol}_{strongest_quote}_forecast_3600.npz")
    preparation_path = docs / "boundary_midpoint_conversion_features_evidence_20260909.json"
    preparation = json.loads(preparation_path.read_text())
    pin(preparation_path)
    if (preparation["feature_groups"] != 56 or preparation["sessions"] != 14 or preparation["model_fits"] != 0
        or preparation["target_labels_decoded"] or preparation["predictive_metrics_computed"]
        or preparation["independent_confirmation_data_opened"]):
        raise ValueError("Require the complete forecast-free matched conversion source preparation")
    for path, digest in preparation["verified_artifact_hashes"].items():
        pin(path, digest)
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_boundary_midpoint_conversion_screen.py",
        ROOT / "scripts/assess_boundary_midpoint_conversion_screen.py",
        ROOT / "src/lob_forge/boundary_midpoint_conversion_inputs.py", ROOT / "tests/test_boundary_midpoint_conversion_inputs.py",
        ROOT / "tests/test_boundary_midpoint_conversion_screen.py",
        docs / "boundary_midpoint_conversion_model_design_20260909.md",
        docs / "boundary_midpoint_conversion_features_report_20260909.md",
        docs / "boundary_midpoint_conversion_contract_20260909.md"):
        pin(path)
    names = (*BASELINES, *MODEL_SPECS, *BLENDS)
    if len(names) != 48 or len(set(names)) != 48 or len(MODEL_SPECS) != 16:
        raise ValueError("The fixed midpoint model family changed")
    protocol = {"study": "boundary_midpoint_conversion_screen_20260909", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "development_only_on_exposed_dates", "dates": parent["dates"], "symbols": list(SYMBOLS),
        "model_specs": MODEL_SPECS, "baselines": list(BASELINES), "blends": {k: list(v) for k, v in BLENDS.items()},
        "trainable_model_fits": 48, "post_fit_panels": 576, "primary_policy": "forecast_3600",
        "decision_policies": ["registered", "forecast_3600"], "neural_config": dict(CONFIG),
        "definition": "docs/research/boundary_midpoint_conversion_model_design_20260909.md",
        "feature_manifest": preparation["feature_manifest"], "feature_preparation_evidence": str(preparation_path.relative_to(ROOT)),
        "original_prepared_run": str(quote_path.relative_to(ROOT)), "original_prepared_protocol_sha256": parent_sha,
        "baseline_run": parent["baseline_run"], "prior_run": parent["prior_run"],
        "quote_reference": {"run": str(quote_path.relative_to(ROOT)), "model": strongest_quote},
        "counted_reference": parent["counted_reference"], "expert_reference": parent["expert_reference"],
        "limits": parent["limits"], "future_prefix_absolute_tolerance": 0, "query_partition_absolute_tolerance": 2e-6,
        "development_advancement": {"reference": "combined_instant_hgb_old_neural", "minimum_mean_gain_pp": .5,
            "both_assets_improve": True, "maximum_log_loss_ratio": 1.01, "minimum_attribution_gain_pp": .25,
            "require_same_complete_gate_against_stronger_measured_controls": True,
            "attribution_is_separate_from_performance": True},
        "confirmation": "Unchanged independent twenty-date, five-point and research-budget gates; no confirmation round consumed here.",
        "input_hashes": dict(sorted(inputs.items()))}
    write_json(destination, protocol)
    print(json.dumps({"protocol": str(destination.relative_to(ROOT)), "sha256": sha256_file(destination),
        "input_hashes": len(inputs), "fits": 48, "panels": 576, "strongest_quote_reference": strongest_quote}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    freeze()
