"""Freeze attention runtime checks only after the three earlier studies finish."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from benchmark_boundary_temporal_attention import PREREQUISITES
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]


def freeze():
    definition_path = ROOT / "docs/research/boundary_temporal_attention_preflight_definition_20260908.json"
    definition = json.loads(definition_path.read_text())
    final_path = ROOT / "docs/research/boundary_temporal_attention_preflight_20260908.json"
    if final_path.exists():
        raise ValueError("Preserve any previous frozen attention preflight")
    # This checks completion, not whether a fresh-confirmation candidate should
    # take priority. The registered research decision still precedes execution.
    paths = {str(definition_path.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT))}
    for path, digest in definition["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Prospectively defined attention benchmark input changed: {path}")
        paths.add(path)
    completed = []
    for source, fits, panels in zip(PREREQUISITES, (24, 24, 40), (2784, 3072, 560)):
        summary_path = ROOT / source
        if not summary_path.exists():
            raise ValueError(f"Required earlier study has not completed: {source}")
        summary = json.loads(summary_path.read_text())
        identity_path = summary_path.parent / "frozen_screen.json"
        identity = json.loads(identity_path.read_text())
        if (summary["trainable_model_fits"] != fits or len(summary["records"]) != panels
            or summary["identity"] != identity):
            raise ValueError("Incomplete or inconsistent prerequisite study")
        dates = sorted({record["date"] for record in summary["records"]})
        for day in dates:
            folder = summary_path.parent / "dates" / day
            check_completed(folder, identity)
            paths.add(str((folder / "completed.json").relative_to(ROOT)))
        paths.update((source, str(identity_path.relative_to(ROOT))))
        completed.append({"summary": source, "sha256": sha256_file(summary_path), "fits": fits, "panels": panels, "dates": dates})
    # Freeze the local source closure without including unrelated backup names.
    for directory in (ROOT / "src/lob_forge", ROOT / "scripts"):
        paths.update(str(p.relative_to(ROOT)) for p in directory.glob("*.py") if p.stem.isidentifier())
    # Retain the already located optional memory-sampler runtime binaries.
    previous = json.loads((ROOT / "docs/research/boundary_proper_score_preflight_20260908.json").read_text())
    for path, digest in previous["input_hashes"].items():
        if "psutil" in path:
            if sha256_file(ROOT / path) != digest:
                raise ValueError("The original optional memory-sampler runtime changed")
            paths.add(path)
    protocol = {**definition, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_status": "source_frozen_after_required_studies_before_any_attention_runtime_or_market_fit",
        "operational_definition": str(definition_path.relative_to(ROOT)), "operational_definition_sha256": sha256_file(definition_path),
        "completed_prerequisite_summaries": PREREQUISITES, "prerequisite_evidence": completed,
        "input_hashes": {path: sha256_file(ROOT / path) for path in sorted(paths)}}
    write_json(final_path, protocol)
    print(json.dumps({"protocol": str(final_path.relative_to(ROOT)), "sha256": sha256_file(final_path), "input_hashes": len(paths)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        print("Pass --freeze only after the earlier studies complete and the fresh-confirmation priority is reviewed.")
