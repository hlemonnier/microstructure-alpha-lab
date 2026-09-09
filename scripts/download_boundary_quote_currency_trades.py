"""Acquire exactly the frozen alternative-pair archives without decoding them."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lob_forge.binance_vision import archive_key, sha256_file, url_for_key

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def acquire(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve every earlier archive acquisition attempt")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Frozen archive acquisition input changed")
    inventory = json.loads((ROOT / protocol["inventory"]).read_text())
    expected_keys = {(s, d) for s in protocol["symbols"] for d in protocol["dates"]}
    if (not inventory["complete"] or len(inventory["files"]) != 42 or len(expected_keys) != 42
        or {(r["symbol"], r["date"]) for r in inventory["files"]} != expected_keys
        or sum(r["expected_bytes"] for r in inventory["files"]) != inventory["total_expected_bytes"]
        or inventory["total_expected_bytes"] > protocol["maximum_total_archive_bytes"]):
        raise ValueError("Exactly the complete bounded archive inventory is required")
    for row in inventory["files"]:
        key = archive_key(market="spot", frequency="daily", dataset="aggTrades", symbol=row["symbol"], date_value=row["date"])
        if (row["url"] != url_for_key(key) or row["filename"] != Path(key).name
            or row["expected_bytes"] > protocol["maximum_single_archive_bytes"]):
            raise ValueError("An archive URL, name or size differs from the fixed inventory")
    output.mkdir(parents=True)
    write(output / "frozen_acquisition.json", {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]})
    begun, records = time.monotonic(), []

    def one(row):
        folder = output / row["symbol"] / row["date"]
        folder.mkdir(parents=True)
        partial = folder / (row["filename"] + ".part")
        record = {**{k: v for k, v in row.items() if k != "artifact_hashes"}, "status": "failed",
                  "inventory_artifact_hashes": row["artifact_hashes"], "archive_or_market_content_decoded": False}
        started = time.monotonic()
        try:
            with (folder / "curl.stderr").open("w") as errors:
                response = subprocess.run(["curl", "--proto", "=https", "--silent", "--show-error", "--fail-with-body",
                    "--connect-timeout", "10", "--max-time", str(protocol["request_seconds"]), "--retry", "0",
                    "--max-filesize", str(protocol["maximum_single_archive_bytes"]),
                    "--dump-header", str(folder / "response_headers.txt"), "--output", str(partial), row["url"]],
                    stdout=subprocess.DEVNULL, stderr=errors, timeout=protocol["request_seconds"] + 5, check=False)
            if response.returncode != 0:
                raise ValueError(f"Preserved archive request failure: curl {response.returncode}")
            digest = sha256_file(partial)
            if partial.stat().st_size != row["expected_bytes"] or digest != row["official_sha256"]:
                raise ValueError("Archive bytes and SHA256 must match the previously frozen official inventory")
            target = folder / row["filename"]
            partial.rename(target)
            record.update(status="verified", path=str(target.relative_to(ROOT)), sha256=digest, bytes=target.stat().st_size)
        except Exception as exc:
            record.update(exception=type(exc).__name__, reason=str(exc))
        record["seconds"] = time.monotonic() - started
        record["acquisition_artifact_hashes"] = {str(p.relative_to(folder)): sha256_file(p) for p in folder.iterdir() if p.is_file()}
        write(folder / "record.json", record)
        return record

    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(one, r) for r in inventory["files"]]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            write(output / "progress.json", {"finished": len(records), "expected": 42, "latest": [row["symbol"], row["date"], row["status"]]})
            print(f"quote_trade_archives={len(records)}/42 {row['status']} {row['symbol']}/{row['date']}", flush=True)
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Archive acquisition input changed during execution")
    complete = all(r["status"] == "verified" for r in records)
    write(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "complete": complete,
        "files": sorted(records, key=lambda r: (r["date"], r["symbol"])), "file_count": len(records),
        "verified_bytes": sum(r.get("bytes", 0) for r in records), "seconds": time.monotonic() - begun,
        "archive_or_market_content_decoded": False, "model_fits": 0, "assessment_metrics_computed": False,
        "independent_confirmation_data_opened": False})
    if not complete:
        raise ValueError("Preserve every failed and completed archive; no implicit resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.protocol.resolve(), args.output.resolve())
