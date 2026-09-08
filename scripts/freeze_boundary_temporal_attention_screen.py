"""Freeze the fixed attention market family after its prerequisites pass."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_temporal_attention import CONFIG
from benchmark_boundary_temporal_attention import choose_backend
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tabicl_screen import verify_artifacts
from run_boundary_temporal_attention_screen import BASELINES, MODEL_SPECS, BLENDS

ROOT = Path(__file__).resolve().parents[1]


def freeze():
    docs = ROOT / "docs/research"
    definition_path = docs / "boundary_temporal_attention_definition_20260908.json"
    preflight_path = docs / "boundary_temporal_attention_preflight_20260908.json"
    parent_path = docs / "boundary_proper_score_screen_20260908.json"
    evidence_path = docs / "boundary_temporal_attention_preflight_evidence_20260908.json"
    final_path = docs / "boundary_temporal_attention_screen_20260908.json"
    if evidence_path.exists() or final_path.exists():
        raise ValueError("Preserve any existing attention preflight evidence or market protocol")
    definition = json.loads(definition_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    parent = json.loads(parent_path.read_text())
    preflight_summary_path = ROOT / "data/research/boundary_temporal_attention_preflight_20260908/summary.json"
    summary = json.loads(preflight_summary_path.read_text())
    if summary["protocol_sha256"] != sha256_file(preflight_path):
        raise ValueError("Original attention preflight identity changed")
    verify_artifacts(preflight_summary_path.parent, summary)
    chosen, feasible = choose_backend(summary["records"])
    if chosen is None or chosen != summary["selected_backend"] or feasible != summary["feasible_backend_summed_median_block_seconds"]:
        raise ValueError("The registered common-backend selector must pass unchanged")
    paths = set()
    for source_protocol in (parent, preflight):
        for path, digest in source_protocol["input_hashes"].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError(f"Frozen attention prerequisite input changed: {path}")
            paths.add(path)
    baseline = ROOT / "results/boundary_proper_score_screen_20260908"
    broad = ROOT / "results/boundary_broad_accuracy_screen_20260908"
    for folder, fits, panels in ((baseline, 24, 3072), (broad, 40, 560)):
        data = json.loads((folder / "summary.json").read_text())
        identity = json.loads((folder / "frozen_screen.json").read_text())
        if data["identity"] != identity or data["trainable_model_fits"] != fits or len(data["records"]) != panels:
            raise ValueError("The original complete prerequisite family is required")
        for day in sorted({r["date"] for r in data["records"]}):
            check_completed(folder / "dates" / day, identity)
            paths.add(str((folder / "dates" / day / "completed.json").relative_to(ROOT)))
        paths.update(str((folder / name).relative_to(ROOT)) for name in ("summary.json", "frozen_screen.json"))
    replay_evidence_path = docs / "boundary_temporal_attention_input_evidence_20260908.json"
    replay_evidence = json.loads(replay_evidence_path.read_text())
    replay_path = ROOT / replay_evidence["summary"]
    if sha256_file(replay_path) != replay_evidence["summary_sha256"]:
        raise ValueError("Original temporal input-replay evidence changed")
    verify_artifacts(replay_path.parent, json.loads(replay_path.read_text()))
    selected = [r for r in summary["records"] if r["backend"] == chosen]
    evidence = {"status": "synthetic_runtime_only", "market_model_fits": 0, "market_assessment_metrics_computed": False,
        "protocol": str(preflight_path.relative_to(ROOT)), "protocol_sha256": sha256_file(preflight_path),
        "summary": str(preflight_summary_path.relative_to(ROOT)), "summary_sha256": sha256_file(preflight_summary_path),
        "selected_backend": chosen, "all_sixteen_attempts_preserved": True, "selected_backend_eight_cases_passed": True,
        "feasible_backend_summed_median_block_seconds": feasible, "records": summary["records"],
        "maximum_selected_backend_projected_seconds": max(r["projected_six_epoch_updates_and_validation_seconds"] for r in selected),
        "maximum_selected_backend_sampled_rss_bytes": max(r["memory"]["maximum_sampled_rss_bytes"] for r in selected),
        "scope": "Fixed-weight arithmetic and independently warmed forecast checks plus measured synthetic blocks. The extrapolation excludes real historical-grid materialization, paging and other market operations; it is not a measured full market fit or memory guarantee."}
    names = [*BASELINES, *MODEL_SPECS, *BLENDS]
    if (len(BASELINES) != 256 or len(MODEL_SPECS) != 8 or len(BLENDS) != 16 or len(names) != len(set(names))
        or len(names) != definition["forecast_sources"] or preflight["config"] != CONFIG):
        raise ValueError("The original attention family, architecture or optimization changed")
    paths.update(str(p.relative_to(ROOT)) for p in (definition_path, preflight_path, parent_path,
        preflight_summary_path, replay_evidence_path, replay_path, Path(__file__).resolve(),
        ROOT / "scripts/run_boundary_temporal_attention_screen.py",
        docs / "boundary_temporal_attention_input_replay_20260908.json",
        ROOT / "results/boundary_expert_feedback_screen_20260908/summary.json",
        docs / "boundary_expert_feedback_evidence_20260908.json"))
    write_json(evidence_path, evidence)
    paths.add(str(evidence_path.relative_to(ROOT)))
    protocol = {**definition, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_status": "complete_original_input_replay_and_backend_preflight_before_market_fits",
        "definition": str(definition_path.relative_to(ROOT)), "definition_sha256": sha256_file(definition_path),
        "baseline_run": str(baseline.relative_to(ROOT)),
        **{key: parent[key] for key in ("manifest", "memory_run", "context_run", "basis_run", "prior_run")},
        "config": dict(CONFIG), "selected_backend": chosen, "retained_sources": list(BASELINES),
        "model_cases": list(MODEL_SPECS), "blend_members": {name: list(pair) for name, pair in BLENDS.items()},
        "preflight_protocol": str(preflight_path.relative_to(ROOT)), "preflight_summary": str(preflight_summary_path.relative_to(ROOT)),
        "preflight_evidence": str(evidence_path.relative_to(ROOT)), "input_replay_evidence": str(replay_evidence_path.relative_to(ROOT)),
        "completed_broad_audit_summary": str((broad / "summary.json").relative_to(ROOT)),
        "additional_completed_development_comparison_runs": ["results/boundary_expert_feedback_screen_20260908", str(broad.relative_to(ROOT))],
        "operational_limits": {"worker_seconds": 3600, "worker_rss_bytes": 10 * 1024 ** 3,
            "rss_plus_driver_bytes": 12 * 1024 ** 3, "memory_sample_seconds": .25},
        "query_probability_absolute_tolerance": 2e-5,
        "input_hashes": {path: sha256_file(ROOT / path) for path in sorted(paths)}}
    write_json(final_path, protocol)
    print(json.dumps({"protocol": str(final_path.relative_to(ROOT)), "sha256": sha256_file(final_path),
        "preflight_evidence_sha256": sha256_file(evidence_path), "input_hashes": len(paths), "backend": chosen}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        print("Pass --freeze only after the registered attention preflight and earlier studies complete; prioritize a ready fresh-confirmation candidate first.")
