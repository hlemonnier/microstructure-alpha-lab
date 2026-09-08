"""Download a bounded, public first-of-month Binance native-depth interval."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_tardis_depth import capture_nanoseconds
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def download_slice(protocol, output, identity, offset):
    folder = output / f"minute_{offset:04d}"
    completed = folder / "completed.json"
    if completed.exists():
        record = json.loads(completed.read_text())
        if record["identity"] != identity or any(sha256_file(ROOT / p) != h for p, h in record["artifact_hashes"].items()):
            raise ValueError("Completed public source bytes changed")
        return record
    folder.mkdir(parents=True, exist_ok=True)
    params = {"from": protocol["date"], "offset": offset, "sliceSize": protocol["slice_minutes"],
              "compression": "gzip", "filters": json.dumps(protocol["filters"], separators=(",", ":"))}
    url = protocol["endpoint"] + "?" + urlencode(params)
    begin = time.monotonic()
    # The protocol permits two attempts per slice, including across resumes.
    attempts = list(folder.glob("attempt_*.json"))
    for attempt in range(len(attempts) + 1, protocol["attempts_per_slice"] + 1):
        payload, headers = folder / f"attempt_{attempt}.payload", folder / f"attempt_{attempt}.headers"
        attempt_file = folder / f"attempt_{attempt}.json"
        write_json(attempt_file, {"offset": offset, "url": url, "state": "started", "model_fits": 0})
        result = subprocess.run(["curl", "--http1.1", "--fail-with-body", "--silent", "--show-error", "--header", "Accept-Encoding: gzip",
            "--connect-timeout", "20", "--max-time", str(protocol["max_response_seconds"]), "--max-filesize", str(protocol["max_response_bytes"]),
            "--dump-header", str(headers), "--output", str(payload), "--write-out", "%{http_code}", url], capture_output=True, text=True)
        status = int(result.stdout) if result.stdout.isdigit() else None
        record = {"offset": offset, "url": url, "state": "response", "attempt": attempt, "exit_code": result.returncode,
                  "http_status": status, "stderr": result.stderr, "bytes": payload.stat().st_size if payload.exists() else 0, "model_fits": 0}
        write_json(attempt_file, record)
        if result.returncode == 0 and status == 200:
            if not payload.stat().st_size or payload.stat().st_size > protocol["max_response_bytes"]:
                raise ValueError("Unexpected public response size")
            with payload.open("rb") as stream:
                compressed = stream.read(2) == b"\x1f\x8b"
            opener = gzip.open if compressed else open
            with opener(payload, "rt", encoding="utf-8") as stream:
                first = next(line for line in stream if line.strip())
            stamp, native = first.rstrip("\n").split(" ", 1)
            interval_start = capture_nanoseconds(protocol["date"] + "T00:00:00.000000000Z") + offset * 60_000_000_000
            if not interval_start <= capture_nanoseconds(stamp) < interval_start + protocol["slice_minutes"] * 60_000_000_000:
                raise ValueError("Public response does not belong to its requested capture-time interval")
            if not isinstance(json.loads(native).get("stream"), str):
                raise ValueError("Response does not contain native market-data messages")
            record.update({"identity": identity, "payload_path": str(payload.relative_to(ROOT)), "format": "gzip_ndjson" if compressed else "plain_ndjson",
                "first_capture": stamp, "seconds": time.monotonic() - begin,
                "artifact_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in (payload, headers, attempt_file)}})
            write_json(completed, record)
            return record
        if status in (401, 403) or status is not None and 400 <= status < 500 and status != 429:
            raise RuntimeError(f"Public endpoint rejected slice {offset}: HTTP {status}; no authentication workaround")
        if attempt < protocol["attempts_per_slice"]:
            time.sleep(30 if status == 429 else 2)
    raise RuntimeError(f"Bounded attempts exhausted for public slice {offset}")


def download(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Frozen public acquisition input changed: {name}")
    if (protocol["date"].split("-")[-1] != "01" or protocol["endpoint"] != "https://api.tardis.dev/v1/data-feeds/binance-futures"
        or not 1 <= protocol["slice_minutes"] <= 10 or protocol["concurrency"] > 4):
        raise ValueError("This runner only supports bounded documented public first-day access")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    frozen = output / "frozen_acquisition.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing public acquisition after source changes")
    write_json(frozen, identity)
    offsets = iter(protocol["minute_offsets"])
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=protocol["concurrency"]) as pool:
        pending = {pool.submit(download_slice, protocol, output, identity, next(offsets)) for _ in range(min(protocol["concurrency"], len(protocol["minute_offsets"])))}
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                records.append(future.result())
                print(f"tardis_slices={len(records)}/{len(protocol['minute_offsets'])} offset={records[-1]['offset']} bytes={records[-1]['bytes']}", flush=True)
                offset = next(offsets, None)
                if offset is not None:
                    pending.add(pool.submit(download_slice, protocol, output, identity, offset))
    write_json(output / "raw_manifest.json", {"identity": identity, "evidence_status": protocol["evidence_status"], "date": protocol["date"],
        "model_fits": 0, "records": sorted(records, key=lambda r: r["offset"]), "total_bytes": sum(r["bytes"] for r in records)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --download for the bounded public acquisition.")
