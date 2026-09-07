"""Inspect public archive metadata without downloading or opening market observations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from lob_forge.binance_vision import archive_key, url_for_key

ROOT = Path(__file__).resolve().parents[1]


def sources():
    for offset in range(20):
        day = (date(2023, 5, 24) + timedelta(days=offset)).isoformat()
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for dataset in ["bookTicker", "aggTrades"]:
                key = archive_key(
                    market="futures/um", frequency="daily", dataset=dataset, symbol=symbol, date_value=day
                )
                yield {
                    "symbol": symbol,
                    "session_date": day,
                    "dataset": dataset,
                    "key": key,
                    "url": url_for_key(key),
                    "checksum_url": url_for_key(key + ".CHECKSUM"),
                }


def probe(entry):
    try:
        request = urllib.request.Request(entry["url"], method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = {"status": response.status, "bytes": int(response.headers["Content-Length"])}
        with urllib.request.urlopen(entry["checksum_url"], timeout=30) as response:
            content = response.read(4096).decode("ascii").strip()
        checksum = content.split()[0]
        if len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum):
            raise ValueError("Invalid official SHA-256 checksum")
        return {**entry, **metadata, "provider_checksum_sha256": checksum.lower()}
    except (ValueError, OSError, urllib.error.URLError) as error:
        return {**entry, "status": getattr(error, "code", None), "error": str(error)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/research/boundary_confirmation_source_plan_20260907.json"
    )
    args = parser.parse_args()
    entries = list(sources())
    if args.probe:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            entries = list(executor.map(probe, entries))
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_only": True,
            "market_observations_downloaded_or_inspected": False,
            "archives": entries,
            "successful_archives": sum(e.get("status") == 200 and "provider_checksum_sha256" in e for e in entries),
            "total_compressed_bytes": sum(e.get("bytes", 0) for e in entries),
            "evaluation_dates": "May 24 through June 12, 2023; twenty consecutive dates, both assets",
            "execution_requires": "A separately frozen model, selection, training, evaluation and resource protocol before downloading observations.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "archives"}, indent=2))
    else:
        print(json.dumps({"planned_archives": len(entries), "requires": "--probe for metadata only"}))
