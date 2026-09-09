"""Prepare the frozen alternative-quote development window, one day at a time."""

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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve each previous complete-window feature attempt")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Frozen complete-window feature prerequisite changed")
    if len(protocol["days"]) != 14 or len({r["date"] for r in protocol["days"]}) != 14:
        raise ValueError("All fourteen registered development source days are required")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    write(output / "frozen_preparation.json", identity)
    begun, sessions, evidence = time.monotonic(), [], []
    for day in protocol["days"]:
        if "reuse_summary" in day:
            summary_path = ROOT / day["reuse_summary"]
            if sha256_file(summary_path) != day["reuse_summary_sha256"]:
                raise ValueError("The original reused preflight changed")
            reused = True
        else:
            daily_path = output / "definitions" / f"{day['date']}.json"
            daily = {**protocol["daily_settings"], **day,
                "input_hashes": {**protocol["input_hashes"], str(protocol_path.relative_to(ROOT)): sha256_file(protocol_path)}}
            write(daily_path, daily)
            folder = output / "dates" / day["date"]
            log = output / "logs" / f"{day['date']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            started, returncode, failure = time.monotonic(), None, None
            try:
                with log.open("w") as stream:
                    result = subprocess.run([sys.executable, str(ROOT / protocol["daily_runner"]),
                        "--protocol", str(daily_path), "--output", str(folder)], cwd=ROOT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["per_day_wall_seconds"], check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                failure = "registered_data_preparation_wall_time_exceeded"
            write(output / "supervision" / f"{day['date']}.json", {"returncode": returncode, "failure": failure,
                "seconds": time.monotonic() - started, "log_sha256": sha256_file(log)})
            if returncode != 0 or failure is not None:
                raise ValueError(f"Preserve the failed data preparation: {day['date']}")
            summary_path, reused = folder / "summary.json", False
            if json.loads(summary_path.read_text())["protocol_sha256"] != sha256_file(daily_path):
                raise ValueError("Daily feature preparation identity changed")
        summary = json.loads(summary_path.read_text())
        if (not summary["complete"] or len(summary["records"]) != 4 or summary["model_fits"] != 0
            or summary["target_labels_decoded"] or summary["predictive_metrics_computed"]
            or summary["independent_confirmation_data_opened"]):
            raise ValueError("Every source day must pass the complete forecast-free checks")
        for path, digest in summary["artifact_hashes"].items():
            if sha256_file(summary_path.parent / path) != digest:
                raise ValueError("A prepared feature artifact changed")
        for symbol, native_symbol, conversion_symbol in protocol["daily_settings"]["pairs"]:
            records = [r for r in summary["records"] if r["symbol"] == symbol]
            if len(records) != 2 or {r["delay_ms"] for r in records} != {100, 500} or len({r["rows"] for r in records}) != 1:
                raise ValueError("Both fixed source delays are required for each target asset")
            canonical = day["canonical_sources"][symbol]
            sessions.append({"symbol": symbol, "date": day["date"], "native_symbol": native_symbol,
                "conversion_symbol": conversion_symbol, "original_features_path": canonical,
                "original_features_sha256": protocol["input_hashes"][canonical],
                "decision_rows": records[0]["rows"],
                "groups": {str(r["delay_ms"]): {**r, "features_path": r["feature_path"]} for r in records},
                "preparation_summary": str(summary_path.relative_to(ROOT)), "preparation_summary_sha256": sha256_file(summary_path),
                "reused_verified_training_preflight": reused})
        evidence.append({"date": day["date"], "reused": reused, "source_audits": summary["source_audits"],
                         "memory": summary["memory"], "seconds": summary["seconds"]})
        write(output / "progress.json", {"complete_days": len(evidence), "expected_days": 14,
            "latest": day["date"], "model_fits": 0, "predictive_metrics_computed": False})
        print(f"quote_feature_days={len(evidence)}/14 {day['date']} reused={reused}", flush=True)
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Complete-window feature input changed during execution")
    write(output / "feature_manifest.json", {"identity": identity, "complete": True, "sessions": sessions,
        "day_evidence": evidence, "seconds": time.monotonic() - begun, "model_fits": 0,
        "target_labels_decoded": False, "predictive_metrics_computed": False, "independent_confirmation_data_opened": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol.resolve(), args.output.resolve())
