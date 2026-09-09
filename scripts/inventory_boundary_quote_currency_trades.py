"""Read only public aggregate-trade headers and official checksums."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lob_forge.binance_vision import archive_key, sha256_file, url_for_key

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def inventory(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve all previous source-inventory attempts")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Source-inventory prerequisite changed")
    output.mkdir(parents=True)
    begun, records = time.monotonic(), []

    def one(symbol, day):
        folder = output / symbol / day
        folder.mkdir(parents=True)
        key = archive_key(market="spot", frequency="daily", dataset="aggTrades", symbol=symbol, date_value=day)
        url = url_for_key(key)
        record = {"symbol": symbol, "date": day, "filename": Path(key).name, "url": url, "status": "failed"}
        try:
            for head in (True, False):
                name = "archive_headers.txt" if head else "official_checksum.txt"
                target = folder / name
                command = ["curl", "--proto", "=https", "--silent", "--show-error", "--fail-with-body",
                    "--max-time", str(protocol["request_seconds"]), "--retry", "0", "--output", str(target)]
                if head:
                    command.append("--head")
                else:
                    command.extend(["--max-filesize", "4096", "--dump-header", str(folder / "checksum_headers.txt")])
                command.append(url if head else url + ".CHECKSUM")
                with (folder / (name + ".stderr")).open("w") as errors:
                    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors,
                                            timeout=protocol["request_seconds"] + 5, check=False)
                if result.returncode != 0:
                    raise ValueError(f"Preserved metadata request failure: curl {result.returncode}")
            raw = (folder / "archive_headers.txt").read_text()
            blocks = [b for b in re.split(r"\n\s*\n", raw) if b.startswith("HTTP/")]
            if not blocks or blocks[-1].splitlines()[0].split()[1] != "200":
                raise ValueError("The exact archive HEAD request must return 200")
            headers = dict(line.split(":", 1) for line in blocks[-1].splitlines()[1:] if ":" in line)
            headers = {k.lower().strip(): v.strip() for k, v in headers.items()}
            size = int(headers["content-length"])
            fields = (folder / "official_checksum.txt").read_text().strip().split()
            if size <= 0 or len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) or fields[1].lstrip("*") != Path(key).name:
                raise ValueError("Positive advertised bytes and an exact official SHA256 filename are required")
            record.update(status="available", expected_bytes=size, official_sha256=fields[0].lower(),
                last_modified=headers.get("last-modified"), etag=headers.get("etag"),
                archive_body_downloaded=False, market_content_decoded=False)
        except Exception as exc:
            record.update(exception=type(exc).__name__, reason=str(exc))
        record["artifact_hashes"] = {str(p.relative_to(folder)): sha256_file(p) for p in folder.iterdir() if p.is_file()}
        write(folder / "record.json", record)
        return record

    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(one, symbol, day) for symbol in protocol["symbols"] for day in protocol["dates"]]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(f"quote_trade_metadata={len(records)}/{len(futures)} {row['status']} {row['symbol']}/{row['date']}", flush=True)
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Source-inventory prerequisite changed during execution")
    write(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path),
        "complete": all(r["status"] == "available" for r in records), "file_count": len(records),
        "files": sorted(records, key=lambda r: (r["date"], r["symbol"])),
        "total_expected_bytes": sum(r.get("expected_bytes", 0) for r in records),
        "seconds": time.monotonic() - begun, "archive_bodies_downloaded": 0,
        "market_content_decoded": False, "model_fits": 0, "assessment_metrics_computed": False,
        "independent_confirmation_data_opened": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory(args.protocol.resolve(), args.output.resolve())
