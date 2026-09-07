"""Apply the prespecified program-level gate after the complete confirmation."""

from __future__ import annotations

import json
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_research_spending import research_round_interval

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol_path = ROOT / "docs/research/boundary_research_error_budget_20260907.json"
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Prespecified research gate implementation changed")
    summary_path = ROOT / "results/boundary_pooled_tree_confirmation_20260907/summary.json"
    results = json.loads(summary_path.read_text())
    if sorted(results["procedures"]) != sorted(protocol["round_one_procedures"]):
        raise ValueError("Registered procedure family changed")
    evidence = {}
    for name, result in results["procedures"].items():
        deltas = [row["original_reference"] for row in result["paired_date_balanced_accuracy_deltas"]]
        interval = research_round_interval(deltas, round_number=1, procedures=2)
        block = research_round_interval(deltas, round_number=1, procedures=2, block_length=3)
        evidence[name] = {
            "interval": interval, "three_day_sensitivity": block,
            "original_gates": result["gates"],
            "research_discovery_gate_passed": result["substantial_predictive_gain_confirmed_with_family_control"] and interval["lower"] > 0,
        }
    output = ROOT / "docs/research/boundary_research_gate_result_20260907.json"
    output.write_text(json.dumps({
        "protocol_sha256": sha256_file(protocol_path), "confirmation_summary_sha256": sha256_file(summary_path),
        "procedures": evidence, "substantial_predictive_gain_confirmed": any(v["research_discovery_gate_passed"] for v in evidence.values()),
        "economic_performance_confirmed": False,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"research_gate_assessed {output}")


if __name__ == "__main__":
    main()
