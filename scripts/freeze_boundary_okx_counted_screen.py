"""Freeze the counted-depth model family after all source caches complete."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_okx_counted_screen import BASELINES, BLENDS, GROUPS, MODEL_SPECS

ROOT = Path(__file__).resolve().parents[1]
DATES = ["2023-06-03", "2023-06-07", "2023-06-11"]


def freeze():
    docs = ROOT / "docs/research"
    final = docs / "boundary_okx_counted_screen_20260909.json"
    evidence_path = docs / "boundary_okx_counted_preparation_evidence_20260909.json"
    if final.exists() or evidence_path.exists():
        raise ValueError("Preserve previous counted-model freezes and preparation evidence")
    paths, expected = set(), {}

    def pin(path, digest=None):
        path = Path(path)
        relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        actual = sha256_file(ROOT / relative)
        if (digest is not None and actual != digest) or (relative in expected and expected[relative] != actual):
            raise ValueError(f"Frozen counted-model prerequisite changed: {relative}")
        paths.add(relative)
        expected[relative] = actual
        return actual

    prerequisites = ("boundary_temporal_attention_screen_20260908.json",
                     "boundary_basis_screen_20260907.json",
                     "boundary_okx_counted_feature_preparation_20260909.json")
    protocols = {}
    for name in prerequisites:
        path = docs / name
        protocols[name] = json.loads(path.read_text())
        pin(path)
        for source, digest in protocols[name]["input_hashes"].items():
            pin(source, digest)
    feature_protocol = protocols[prerequisites[2]]
    feature_path = ROOT / "data/research/boundary_okx_counted_features_20260909/feature_manifest.json"
    feature_manifest = json.loads(feature_path.read_text())
    if (not feature_manifest["complete"] or feature_manifest["market_model_fits"] != 0
        or feature_manifest["assessment_labels_decoded"] or feature_manifest["independent_confirmation_data_opened"]
        or feature_manifest["identity"]["protocol_sha256"] != sha256_file(docs / prerequisites[2])):
        raise ValueError("Require the complete original feature preparation without new model assessments")
    source_days = [(date(2023, 5, 29) + timedelta(days=i)).isoformat() for i in range(14)]
    universe = {(symbol, day) for symbol in SYMBOLS for day in source_days}
    sessions = feature_manifest["sessions"]
    if len(sessions) != 28 or {(r["symbol"], r["date"]) for r in sessions} != universe:
        raise ValueError("Exactly the fixed 28 source dates/assets are required")
    dependencies = {(r["symbol"], r["date"]): r for r in feature_protocol["sources"]}
    pin(feature_path)
    pin(feature_path.parent / "frozen_preparation.json")
    source_records, source_evidence = [], []
    for session in sessions:
        key = session["symbol"], session["date"]
        dependency = dependencies[key]
        source_path = ROOT / session["source_record"]
        pin(source_path, session["source_record_sha256"])
        source = json.loads(source_path.read_text())
        for field in ("symbol", "date", "protocol_sha256", "archive_sha256", "features_sha256"):
            if source[field] != dependency[field]:
                raise ValueError(f"Source lineage changed for {key}/{field}")
        if (not source["save_reload_exact"] or source["assessment_labels_decoded"]
            or source["market_model_fits"] != 0 or not source["archive_audit"]["complete_gzip_crc_and_size_trailer_verified"]):
            raise ValueError("Every source requires complete original audit and exact sidecar replay")
        if "supervision" in dependency:
            status_path = ROOT / dependency["supervision"]
            pin(status_path)
            status = json.loads(status_path.read_text())
            if status["status"] != "complete" or status["record_sha256"] != session["source_record_sha256"]:
                raise ValueError("Every corrected source must have successful original supervision")
        else:
            if session["source_record_sha256"] != dependency["record_sha256"]:
                raise ValueError("A previously completed reused source changed")
        pin(source["archive"], source["archive_sha256"])
        pin(source["observation_path"], source["observation_sha256"])
        pin(session["original_features_path"], session["original_features_sha256"])
        if source["observation_sha256"] != session["source_observation_sha256"]:
            raise ValueError("Source and feature sidecar hashes differ")
        groups = session["groups"]
        if set(groups) != {f"top{levels}_{delay}" for levels, delay in GROUPS}:
            raise ValueError("Every source requires all three fixed depth/delay groups")
        for group in groups.values():
            pin(group["features_path"], group["sha256"])
            if not group["save_reload_exact"] or group["rows"] != session["decision_rows"]:
                raise ValueError("Feature persistence must preserve all original rows")
        pin(feature_path.parent / key[0] / key[1] / "record.json")
        source_records.append(source)
        source_evidence.append({"symbol": key[0], "date": key[1], "decision_rows": session["decision_rows"],
            "source_record": session["source_record"], "source_record_sha256": session["source_record_sha256"],
            "source_observation_sha256": session["source_observation_sha256"],
            "initial_updates_ignored": source["book_checks"].get("initial_updates_ignored", 0),
            "publisher_gaps": source["book_checks"].get("publisher_gaps", 0),
            "available_rows": {name: group["available_rows"] for name, group in groups.items()},
            "feature_columns": {name: len(group["columns"]) for name, group in groups.items()}})
    correction = ROOT / "data/research/boundary_okx_depth_development_initial_delta_revision_20260909/summary.json"
    pin(correction)
    corrected = json.loads(correction.read_text())
    corrected_keys = {(r["symbol"], r["date"]) for r in feature_protocol["sources"] if "supervision" in r}
    if (not corrected["complete"] or corrected["market_model_fits"] != 0 or corrected["assessment_labels_decoded"]
        or corrected["independent_confirmation_data_opened"] or len(corrected["sessions"]) != 25
        or {(r["symbol"], r["date"]) for r in corrected["sessions"]} != corrected_keys
        or any(r["status"] != "complete" for r in corrected["supervision"])):
        raise ValueError("The corrected 25-source preparation must finish its complete final audit")
    parent = protocols[prerequisites[1]]
    for field, path_field, hash_field in (("manifest", "features_path", "sha256"),
                                        ("depth_manifest", "observation_path", "observation_sha256"),
                                        ("spot_manifest", "features_path", "sha256")):
        manifest_path = ROOT / parent[field]
        pin(manifest_path)
        rows = json.loads(manifest_path.read_text())["sessions"]
        chosen = [r for r in rows if (r["symbol"], r["session_date"]) in universe]
        if len(chosen) != 28 or {(r["symbol"], r["session_date"]) for r in chosen} != universe:
            raise ValueError("All original matching source partitions must exist")
        for row in chosen:
            pin(row[path_field], row[hash_field])
    baseline = ROOT / "results/boundary_temporal_attention_screen_20260908"
    identity = json.loads((baseline / "frozen_screen.json").read_text())
    pin(baseline / "frozen_screen.json")
    pin(baseline / "summary.json")
    for day in DATES:
        check_completed(baseline / "dates" / day, identity)
        pin(baseline / "dates" / day / "completed.json")
        for symbol in SYMBOLS:
            for name in BASELINES:
                for policy in ("registered", "forecast_3600"):
                    pin(baseline / "dates" / day / "predictions" / f"{symbol}_{name}_{policy}.npz")
        pin(ROOT / parent["newton_run"] / "dates" / day / "combined_hgb.joblib")
        neural = ROOT / "results/boundary_basis_screen_20260907/dates" / day / "combined_neural"
        for path in neural.rglob("*"):
            if path.is_file():
                pin(path)
    new_files = [Path(__file__).resolve(), ROOT / "scripts/run_boundary_okx_counted_screen.py",
        ROOT / "scripts/assess_boundary_okx_counted_screen.py",
        docs / "boundary_okx_counted_model_design_20260909.md",
        ROOT / "tests/test_boundary_okx_screen_inputs.py", ROOT / "tests/test_boundary_okx_quantity_scale.py",
        ROOT / "src/lob_forge/boundary_okx_inputs.py", ROOT / "src/lob_forge/boundary_okx_quantity_scale.py",
        ROOT / "results/boundary_expert_feedback_screen_20260908/summary.json",
        docs / "boundary_expert_feedback_evidence_20260908.json"]
    for path in new_files:
        pin(path)
    names = [*BASELINES, *MODEL_SPECS, *BLENDS]
    if len(MODEL_SPECS) != 22 or len(BLENDS) != 32 or len(names) != 62 or len(set(names)) != 62:
        raise ValueError("The complete fixed counted-model family changed")
    evidence = {"status": "complete_exposed_development_source_and_feature_preparation",
        "feature_manifest": str(feature_path.relative_to(ROOT)), "feature_manifest_sha256": sha256_file(feature_path),
        "feature_protocol_sha256": sha256_file(docs / prerequisites[2]), "source_count": 28,
        "shape_feature_groups": 84, "market_model_fits": 0, "assessment_labels_decoded": False,
        "independent_confirmation_data_opened": False, "all_pinned_inputs_verified": True,
        "source_messages": sum(r["book_checks"]["messages"] for r in source_records),
        "source_observation_bytes": sum(r["observation_bytes"] for r in source_records),
        "decision_rows": sum(r["decision_rows"] for r in sessions), "sessions": source_evidence,
        "scope": "Publisher-time reconstruction with strict delayed sampling. Missing source arrival clocks and sequence identifiers prevent certification of actual live feed availability or packet-loss completeness."}
    write_json(evidence_path, evidence)
    pin(evidence_path)
    protocol = {"study": "boundary_okx_counted_screen_20260909", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "development_only_on_exposed_dates", "dates": DATES, "symbols": list(SYMBOLS),
        "model_specs": MODEL_SPECS, "baselines": list(BASELINES), "blends": {k: list(v) for k, v in BLENDS.items()},
        "trainable_model_fits": 66, "post_fit_panels": 744, "primary_policy": "forecast_3600",
        "decision_policies": ["registered", "forecast_3600"], "neural_config": dict(CONFIG),
        "definition": "docs/research/boundary_okx_counted_model_design_20260909.md",
        "feature_manifest": str(feature_path.relative_to(ROOT)),
        **{k: parent[k] for k in ("manifest", "depth_manifest", "spot_manifest", "newton_run", "prior_run")},
        "basis_run": "results/boundary_basis_screen_20260907", "baseline_run": str(baseline.relative_to(ROOT)),
        "preparation_evidence": str(evidence_path.relative_to(ROOT)),
        "limits": {"worker_rss_bytes": 8 * 1024**3, "rss_plus_driver_bytes": 8 * 1024**3,
                   "memory_sample_seconds": .2, "per_fit_wall_seconds": 900},
        "future_prefix_absolute_tolerance": 0, "query_partition_absolute_tolerance": 2e-6,
        "development_advancement": {"reference": "combined_instant_hgb_old_neural", "minimum_mean_gain_pp": .5,
            "both_assets_improve": True, "maximum_log_loss_ratio": 1.01},
        "confirmation": "Unchanged independent twenty-date, five-point and research-budget gates; no confirmation round consumed here.",
        "input_hashes": {path: expected[path] for path in sorted(paths)}}
    write_json(final, protocol)
    print(json.dumps({"protocol": str(final.relative_to(ROOT)), "sha256": sha256_file(final),
        "input_hashes": len(paths), "evidence_sha256": sha256_file(evidence_path), "fits": 66, "panels": 744}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    freeze()
