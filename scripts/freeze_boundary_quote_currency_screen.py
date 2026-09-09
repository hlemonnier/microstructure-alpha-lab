"""Freeze native quote-source models after the counted-depth study completes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_quote_currency_screen import BASELINES, BLENDS, MODEL_SPECS

ROOT = Path(__file__).resolve().parents[1]


def freeze():
    docs = ROOT / "docs/research"
    destination = docs / "boundary_quote_currency_screen_20260909.json"
    if destination.exists():
        raise ValueError("Preserve every original alternative-quote model freeze")
    inputs = {}

    def pin(path, digest=None):
        path = Path(path)
        absolute = path if path.is_absolute() else ROOT / path
        key = str(absolute.relative_to(ROOT))
        actual = sha256_file(absolute)
        if (digest is not None and digest != actual) or (key in inputs and inputs[key] != actual):
            raise ValueError(f"A frozen quote-model prerequisite changed: {key}")
        inputs[key] = actual
        return actual

    parent_path = docs / "boundary_okx_counted_screen_20260909.json"
    parent = json.loads(parent_path.read_text())
    parent_sha = pin(parent_path)
    for path, digest in parent["input_hashes"].items():
        pin(path, digest)
    counted_path = ROOT / "results/boundary_okx_counted_screen_20260909"
    counted_summary = json.loads((counted_path / "summary.json").read_text())
    counted_identity = json.loads((counted_path / "frozen_screen.json").read_text())
    counted_evidence_path = docs / "boundary_okx_counted_evidence_20260909.json"
    counted_evidence = json.loads(counted_evidence_path.read_text())
    pin(counted_evidence_path)
    if (counted_summary["identity"] != counted_identity or counted_identity["protocol_sha256"] != parent_sha
        or counted_evidence["reporting_panels_recomputed_exactly"] != 744
        or counted_evidence["summary_sha256"] != pin(counted_path / "summary.json")
        or counted_evidence["substantial_gain_confirmed"]):
        raise ValueError("Require the fully completed and verified original counted-depth screen")
    pin(counted_path / "frozen_screen.json")
    strongest_counted = counted_evidence["primary_leaderboard"][0]["model"]
    for day in parent["dates"]:
        folder = counted_path / "dates" / day
        check_completed(folder, counted_identity)
        pin(folder / "completed.json")
        prepared = folder / "prepared"
        frozen = json.loads((prepared / "prepared.json").read_text())
        if frozen["identity"] != counted_identity:
            raise ValueError("The original selected-row matrices changed identity")
        pin(prepared / "prepared.json")
        for path, digest in frozen["artifact_hashes"].items():
            pin(prepared / path, digest)
        for symbol in SYMBOLS:
            pin(folder / "predictions" / f"{symbol}_{strongest_counted}_forecast_3600.npz")
    preparation_path = docs / "boundary_quote_currency_feature_preparation_evidence_20260909.json"
    preparation = json.loads(preparation_path.read_text())
    pin(preparation_path)
    if (preparation["feature_groups"] != 56 or preparation["target_sessions"] != 28 or preparation["model_fits"] != 0
        or preparation["target_labels_decoded"] or preparation["predictive_metrics_computed"]
        or preparation["independent_confirmation_data_opened"]):
        raise ValueError("Require the original complete forecast-free alternative-quote preparation")
    for path, digest in preparation["verified_artifact_hashes"].items():
        pin(path, digest)
    expert_path = ROOT / "results/boundary_expert_feedback_screen_20260908/summary.json"
    pin(expert_path)
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_boundary_quote_currency_screen.py",
        ROOT / "scripts/assess_boundary_quote_currency_screen.py",
        ROOT / "scripts/assess_boundary_quote_currency_preparation.py",
        ROOT / "src/lob_forge/boundary_quote_currency_inputs.py", ROOT / "tests/test_boundary_quote_currency_inputs.py",
        ROOT / "tests/test_boundary_quote_currency_screen_inputs.py",
        docs / "boundary_quote_currency_model_design_20260909.md",
        docs / "boundary_quote_currency_feature_preparation_report_20260909.md"):
        pin(path)
    names = (*BASELINES, *MODEL_SPECS, *BLENDS)
    if len(names) != 38 or len(set(names)) != 38 or len(MODEL_SPECS) != 12:
        raise ValueError("The complete fixed alternative-quote model family changed")
    protocol = {"study": "boundary_quote_currency_screen_20260909", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "development_only_on_exposed_dates", "dates": parent["dates"], "symbols": list(SYMBOLS),
        "model_specs": MODEL_SPECS, "baselines": list(BASELINES), "blends": {k: list(v) for k, v in BLENDS.items()},
        "trainable_model_fits": 36, "post_fit_panels": 456, "primary_policy": "forecast_3600",
        "decision_policies": ["registered", "forecast_3600"], "neural_config": dict(CONFIG),
        "definition": "docs/research/boundary_quote_currency_model_design_20260909.md",
        "feature_manifest": preparation["feature_manifest"], "feature_preparation_evidence": str(preparation_path.relative_to(ROOT)),
        "original_prepared_run": str(counted_path.relative_to(ROOT)), "original_prepared_protocol_sha256": parent_sha,
        "baseline_run": parent["baseline_run"], "prior_run": parent["prior_run"],
        "counted_reference": {"run": str(counted_path.relative_to(ROOT)), "model": strongest_counted},
        "expert_reference": {"summary": str(expert_path.relative_to(ROOT)), "case": "confidence_context_eta10_h300"},
        "limits": parent["limits"], "future_prefix_absolute_tolerance": 0, "query_partition_absolute_tolerance": 2e-6,
        "development_advancement": {"reference": "combined_instant_hgb_old_neural", "minimum_mean_gain_pp": .5,
            "both_assets_improve": True, "maximum_log_loss_ratio": 1.01, "minimum_native_vs_fx_gain_pp": .25,
            "require_same_complete_gate_against_stronger_measured_controls": True},
        "confirmation": "Unchanged independent twenty-date, five-point and research-budget gates; no confirmation round consumed here.",
        "input_hashes": dict(sorted(inputs.items()))}
    write_json(destination, protocol)
    print(json.dumps({"protocol": str(destination.relative_to(ROOT)), "sha256": sha256_file(destination),
        "input_hashes": len(inputs), "fits": 36, "panels": 456, "strongest_counted_reference": strongest_counted}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    freeze()
