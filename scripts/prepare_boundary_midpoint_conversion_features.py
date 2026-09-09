"""Prepare the fixed side-aware conversion window with isolated daily workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from lob_forge.binance_vision import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve each full side-aware conversion preparation attempt")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("The original full-window side-aware prerequisite changed")
    if len(protocol["days"]) != 14 or len({r["date"] for r in protocol["days"]}) != 14:
        raise ValueError("All fourteen registered exposed source dates are required")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    write(output / "frozen_preparation.json", identity)
    started, sessions = time.monotonic(), []
    for day in protocol["days"]:
        reused = "reuse_summary" in day
        if reused:
            summary_path = ROOT / day["reuse_summary"]
            if sha256_file(summary_path) != day["reuse_summary_sha256"]:
                raise ValueError("The original side-aware training preflight changed")
        else:
            daily_path = output / "definitions" / f"{day['date']}.json"
            daily = {**protocol["daily_settings"], **day,
                "input_hashes": {**protocol["input_hashes"], str(protocol_path.relative_to(ROOT)): sha256_file(protocol_path)}}
            write(daily_path, daily)
            folder = output / "dates" / day["date"]
            log_path = output / "logs" / f"{day['date']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            wall_start, returncode, failure = time.monotonic(), None, None
            try:
                with log_path.open("w") as stream:
                    result = subprocess.run([sys.executable, str(ROOT / protocol["daily_runner"]), "--protocol", str(daily_path),
                        "--output", str(folder)], cwd=ROOT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["per_day_wall_seconds"], check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                failure = "registered_feature_preparation_wall_time_exceeded"
            write(output / "supervision" / f"{day['date']}.json", {"returncode": returncode, "failure": failure,
                "seconds": time.monotonic()-wall_start, "log_sha256": sha256_file(log_path)})
            if returncode != 0 or failure is not None:
                raise ValueError(f"Preserve failed side-aware feature preparation: {day['date']}")
            summary_path = folder / "summary.json"
            if json.loads(summary_path.read_text())["protocol_sha256"] != sha256_file(daily_path):
                raise ValueError("Daily side-aware preparation identity changed")
        summary = json.loads(summary_path.read_text())
        if (not summary["complete"] or summary["model_fits"] != 0 or summary["target_labels_decoded"]
            or summary["predictive_metrics_computed"] or summary["independent_confirmation_data_opened"]
            or not summary["both_variants_share_exact_conversion_sides"]):
            raise ValueError("Every day must pass its original checks without predictive assessments")
        for path, digest in summary["artifact_hashes"].items():
            if sha256_file(summary_path.parent / path) != digest:
                raise ValueError("A persisted daily side-aware artifact changed")
        records = summary["records"]
        if (len(records) != 4 or {(r["variant"], r["delay_ms"]) for r in records}
            != {(v, d) for v in ("raw_side", "midpoint_side") for d in (100, 500)}
            or len({r["rows"] for r in records}) != 1):
            raise ValueError("Each day requires the same original rows and all four fixed groups")
        sessions.append({"symbol": "BTCUSDT", "date": day["date"], "decision_rows": records[0]["rows"],
            "canonical": day["canonical"], "canonical_sha256": protocol["input_hashes"][day["canonical"]],
            "groups": {f"{r['variant']}_{r['delay_ms']}": r for r in records},
            "source_audits": summary["source_audits"], "memory": summary["memory"], "seconds": summary["seconds"],
            "preparation_summary": str(summary_path.relative_to(ROOT)), "preparation_summary_sha256": sha256_file(summary_path),
            "reused_training_preflight": reused})
        write(output / "progress.json", {"complete_days": len(sessions), "expected_days": 14, "latest": day["date"],
            "model_fits": 0, "predictive_metrics_computed": False})
        print(f"side_aware_feature_days={len(sessions)}/14 {day['date']} reused={reused}", flush=True)
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("A full-window side-aware prerequisite changed during preparation")
    write(output / "feature_manifest.json", {"identity": identity, "complete": True, "sessions": sessions,
        "seconds": time.monotonic()-started, "model_fits": 0, "target_labels_decoded": False,
        "predictive_metrics_computed": False, "independent_confirmation_data_opened": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol.resolve(), args.output.resolve())
