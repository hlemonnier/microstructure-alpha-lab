"""Download a fixed public OKX development inventory with explicit integrity checks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def acquire(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve any existing acquisition attempt")
    for path, checksum in protocol["input_hashes"].items():
        if hashes(ROOT / path)[0] != checksum:
            raise ValueError(f"Frozen public acquisition input changed: {path}")
    inventory = json.loads((ROOT / protocol["inventory"]).read_text())
    if (inventory["file_count"] != 28 or len(inventory["files"]) != 28
        or inventory["total_expected_compressed_bytes"] > protocol["maximum_total_archive_bytes"]
        or protocol["workers"] != 2):
        raise ValueError("Only the bounded original inventory is authorized")
    if len({r["filename"] for r in inventory["files"]}) != 28:
        raise ValueError("Duplicate public archive")
    for row in inventory["files"]:
        expected = f"{row['symbol']}-L2orderbook-400lv-{row['date']}.tar.gz"
        url = f"https://static.okx.com/cdn/okx/match/orderbook/L2/400lv/daily/{row['date'].replace('-', '')}/{expected}"
        if (row["url"] != url or row["filename"] != expected or row["date"] not in protocol["dates"]
            or row["symbol"] not in protocol["symbols"] or row["expected_bytes"] > protocol["maximum_single_archive_bytes"]):
            raise ValueError("Unregistered archive URL, asset, date or size")
    output.mkdir(parents=True)
    begun = time.monotonic()

    def one(row):
        folder = output / row["filename"].removesuffix(".tar.gz")
        folder.mkdir()
        started = time.monotonic()
        try:
            if row["filename"] in protocol["reuse"]:
                reuse = protocol["reuse"][row["filename"]]
                target = ROOT / reuse["path"]
                sha, md5 = hashes(target)
                if sha != reuse["sha256"]:
                    raise ValueError("The original acquired sample changed")
                reused = True
            else:
                target = folder / row["filename"]
                partial = folder / (row["filename"] + ".part")
                with (folder / "curl.stderr.log").open("w") as log:
                    subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--connect-timeout", "10", "--max-time", str(protocol["per_request_seconds"]),
                        "--retry", str(protocol["transport_retries"]), "--retry-all-errors", "--retry-delay", "1",
                        "--max-filesize", str(protocol["maximum_single_archive_bytes"]),
                        "--dump-header", str(folder / "response_headers.txt"), "--output", str(partial), row["url"]],
                        check=True, stdout=subprocess.DEVNULL, stderr=log, timeout=protocol["per_file_process_seconds"])
                sha, md5 = hashes(partial)
                if partial.stat().st_size != row["expected_bytes"] or md5 != row["official_content_md5_base64"]:
                    raise ValueError("Provider byte length or HTTP MD5 disagrees with the downloaded body")
                partial.rename(target)
                reused = False
            if target.stat().st_size != row["expected_bytes"] or md5 != row["official_content_md5_base64"]:
                raise ValueError("Archive integrity mismatch")
            record = {**row, "status": "verified", "path": str(target.relative_to(ROOT)), "sha256": sha,
                "md5_base64": md5, "reused_original_sample": reused, "new_downloaded_bytes": 0 if reused else target.stat().st_size,
                "gzip_tar_or_market_content_opened_by_this_step": False, "seconds": time.monotonic() - started}
        except Exception as exc:
            record = {**row, "status": "failed", "exception": type(exc).__name__, "reason": str(exc),
                "seconds": time.monotonic() - started, "market_content_opened_by_this_step": False}
        record["local_artifact_hashes"] = {str(p.relative_to(output)): hashes(p)[0] for p in folder.iterdir() if p.is_file() and p.suffix != ".gz"}
        write(folder / "record.json", record)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as executor:
        pending = [executor.submit(one, row) for row in inventory["files"]]
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            write(output / "progress.json", {"records": records, "market_model_fits": 0})
            print(f"okx_archives={len(records)}/28 status={record['status']} file={record['filename']}", flush=True)
    complete = all(r["status"] == "verified" for r in records)
    summary = {"protocol_sha256": hashes(protocol_path)[0], "files": sorted(records, key=lambda r: (r["date"], r["symbol"])),
        "complete": complete, "file_count": len(records), "new_downloaded_bytes": sum(r.get("new_downloaded_bytes", 0) for r in records),
        "total_verified_archive_bytes": sum(r["expected_bytes"] for r in records if r["status"] == "verified"),
        "market_model_fits": 0, "assessment_metrics_computed": False, "independent_confirmation_data_opened": False,
        "gzip_tar_or_market_content_opened_by_this_step": False, "seconds": time.monotonic() - begun,
        "artifact_hashes": {str(p.relative_to(output)): hashes(p)[0] for p in output.glob("*/record.json")}}
    write(output / "summary.json", summary)
    if not complete:
        raise ValueError("Preserve incomplete acquisition and every successful/failed artifact; no implicit resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.protocol.resolve(), args.output.resolve())
