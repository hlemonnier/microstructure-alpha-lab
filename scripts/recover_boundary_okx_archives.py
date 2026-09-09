"""Resume frozen failed public downloads into new, separately audited files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def recover(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve every prior recovery attempt")
    for path, checksum in protocol["input_hashes"].items():
        if hashes(ROOT / path)[0] != checksum:
            raise ValueError(f"Frozen recovery input changed: {path}")
    output.mkdir(parents=True)
    results, started = [], time.monotonic()
    for row in protocol["files"]:
        original = json.loads((ROOT / row["failure_record"]).read_text())
        if original["status"] != "failed" or original["filename"] != row["filename"]:
            raise ValueError("Recovery requires its exact preserved failed attempt")
        expected = f"https://static.okx.com/cdn/okx/match/orderbook/L2/400lv/daily/{original['date'].replace('-', '')}/{original['filename']}"
        if original["url"] != expected or not "2023-05-29" <= original["date"] <= "2023-06-11":
            raise ValueError("Only the original official development source may be resumed")
        folder = output / row["filename"].removesuffix(".tar.gz")
        folder.mkdir()
        partial = folder / (row["filename"] + ".part")
        begin = time.monotonic()
        try:
            shutil.copyfile(ROOT / row["partial_source"], partial)
            if hashes(partial)[0] != row["partial_sha256"] or not 0 < partial.stat().st_size < original["expected_bytes"]:
                raise ValueError("Frozen original prefix must be exact and incomplete")
            prefix_bytes = partial.stat().st_size
            with (folder / "curl.stderr.log").open("w") as log:
                subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--proto", "=https",
                    "--proto-redir", "=https", "--connect-timeout", "10", "--max-time", str(protocol["request_seconds"]),
                    "--retry", "0", "--continue-at", "-", "--max-filesize", str(original["expected_bytes"]),
                    "--dump-header", str(folder / "response_headers.txt"), "--output", str(partial), original["url"]],
                    check=True, stderr=log, stdout=subprocess.DEVNULL, timeout=protocol["request_seconds"] + 20)
            sha, md5 = hashes(partial)
            if partial.stat().st_size != original["expected_bytes"] or md5 != original["official_content_md5_base64"]:
                raise ValueError("Resumed body differs from original provider byte count or MD5")
            destination = folder / row["filename"]
            partial.rename(destination)
            result = {**original, "status": "verified", "path": str(destination.relative_to(ROOT)),
                "sha256": sha, "md5_base64": md5, "resumed_prefix_bytes": prefix_bytes,
                "new_downloaded_bytes": original["expected_bytes"] - prefix_bytes,
                "failure_record": row["failure_record"], "failure_record_sha256": hashes(ROOT / row["failure_record"])[0]}
            # Transport failure details belong to the separately linked attempt.
            result.pop("exception", None)
            result.pop("reason", None)
            result.pop("local_artifact_hashes", None)
        except Exception as exc:
            result = {"filename": row["filename"], "status": "failed", "exception": type(exc).__name__, "reason": str(exc)}
        result.update(seconds=time.monotonic() - begin, gzip_tar_or_market_content_opened_by_this_step=False)
        result["artifact_hashes"] = {p.name: hashes(p)[0] for p in folder.iterdir() if p.is_file()}
        write(folder / "record.json", result)
        results.append(result)
        print(f"recovered={len(results)}/{len(protocol['files'])} status={result['status']} {row['filename']}", flush=True)
    complete = all(r["status"] == "verified" for r in results)
    write(output / "summary.json", {"protocol_sha256": hashes(protocol_path)[0], "files": results,
        "complete": complete, "seconds": time.monotonic() - started, "market_model_fits": 0,
        "independent_confirmation_data_opened": False, "gzip_tar_or_market_content_opened_by_this_step": False})
    if not complete:
        raise ValueError("Preserve incomplete recovery; no implicit retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.protocol.resolve(), args.output.resolve())
