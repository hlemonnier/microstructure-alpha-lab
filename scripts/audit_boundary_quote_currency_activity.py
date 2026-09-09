"""Inventory alternative spot quote currencies without fitting or scoring models."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from lob_forge.binance_vision import archive_key, sha256_file, url_for_key

ROOT = Path(__file__).resolve().parents[1]


def write(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def read_activity(path, symbol, day, maximum_member_bytes):
    expected = f"{symbol}-1m-{day}.csv"
    start = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
    base, quote, buy_base, buy_quote = (Decimal(0) for _ in range(4))
    trades, active_minutes, rows, closes = 0, 0, 0, []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != expected or members[0].file_size > maximum_member_bytes:
            raise ValueError("Exactly the bounded expected one-minute candle member is required")
        with archive.open(members[0]) as raw:
            for i, row in enumerate(csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))):
                if len(row) != 12 or i >= 1440 or int(row[0]) != start + i * 60000 or int(row[6]) != start + (i+1)*60000 - 1:
                    raise ValueError("A complete 1440-minute millisecond clock and exact candle schema are required")
                values = [Decimal(row[j]) for j in (1, 2, 3, 4, 5, 7, 9, 10)]
                opening, high, low, closing, volume, quote_volume, taker_base, taker_quote = values
                if (not all(v.is_finite() for v in values) or min(opening, high, low, closing) <= 0
                    or low > min(opening, closing) or high < max(opening, closing)
                    or min(volume, quote_volume, taker_base, taker_quote) < 0
                    or taker_base > volume or taker_quote > quote_volume or not re.fullmatch(r"\d+", row[8])):
                    raise ValueError("Finite price ranges and consistent nonnegative traded quantities required")
                count = int(row[8])
                if (count == 0) != (volume == 0):
                    raise ValueError("Reported activity count and base quantity must agree on empty candles")
                base += volume
                quote += quote_volume
                buy_base += taker_base
                buy_quote += taker_quote
                trades += count
                active_minutes += count > 0
                rows += 1
                closes.append(closing)
    if rows != 1440:
        raise ValueError("Every calendar minute must be retained in the source inventory")
    return {"rows": rows, "base_quantity": str(base), "quote_quantity": str(quote),
        "taker_buy_base_quantity": str(buy_base), "taker_buy_quote_quantity": str(buy_quote),
        "reported_trades": trades, "active_minutes": active_minutes,
        "minimum_minute_close": str(min(closes)), "maximum_minute_close": str(max(closes)),
        "full_member_crc_verified": True}


def audit(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve each inventory attempt; no implicit resume or retry")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Frozen activity-inventory input changed: {path}")
    output.mkdir(parents=True)
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    write(output / "frozen_inventory.json", identity)
    begun = time.monotonic()

    def one(symbol, day):
        folder = output / symbol / day
        folder.mkdir(parents=True)
        key = archive_key(market="spot", frequency="daily", dataset="klines", symbol=symbol, date_value=day, interval="1m")
        url = url_for_key(key)
        record = {"symbol": symbol, "date": day, **protocol["pairs"][symbol], "url": url, "status": "failed"}
        try:
            destinations = []
            for suffix in (".CHECKSUM", ""):
                target = folder / (Path(key).name + suffix)
                headers, errors = target.with_name(target.name + ".headers"), target.with_name(target.name + ".stderr")
                with errors.open("w") as stderr:
                    process = subprocess.run(["curl", "--proto", "=https", "--silent", "--show-error", "--fail-with-body",
                        "--max-time", str(protocol["request_seconds"]), "--retry", "0", "--max-filesize", str(protocol["maximum_response_bytes"]),
                        "--dump-header", str(headers), "--output", str(target), url + suffix],
                        stdout=subprocess.DEVNULL, stderr=stderr, timeout=protocol["request_seconds"] + 5, check=False)
                if process.returncode != 0:
                    raise ValueError(f"Preserved source request failure, curl code {process.returncode}")
                if target.stat().st_size > protocol["maximum_response_bytes"]:
                    raise ValueError("The source response exceeded its registered bound")
                destinations.append(target)
            checksum, archive = destinations
            fields = checksum.read_text().strip().split()
            if len(fields) != 2 or not re.fullmatch(r"[a-fA-F0-9]{64}", fields[0]) or fields[1].lstrip("*") != archive.name:
                raise ValueError("An official checksum naming the exact requested archive is required")
            digest = sha256_file(archive)
            if digest != fields[0].lower():
                raise ValueError("The source body must match the official SHA256 checksum")
            activity = read_activity(archive, symbol, day, protocol["maximum_member_bytes"])
            record.update(status="complete", archive=str(archive.relative_to(ROOT)), archive_sha256=digest,
                archive_bytes=archive.stat().st_size, checksum=str(checksum.relative_to(ROOT)), checksum_sha256=sha256_file(checksum), activity=activity)
        except Exception as exc:
            record.update(exception=type(exc).__name__, reason=str(exc))
        record["artifact_hashes"] = {str(p.relative_to(folder)): sha256_file(p) for p in folder.iterdir() if p.is_file()}
        write(folder / "record.json", record)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        pending = [pool.submit(one, symbol, day) for symbol in protocol["pairs"] for day in protocol["dates"]]
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print(f"activity_source={len(records)}/{len(pending)} {record['status']} {record['symbol']}/{record['date']}", flush=True)
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Activity-inventory implementation changed during its run")
    write(output / "summary.json", {"identity": identity, "inventory_complete": True,
        "all_sources_available": all(r["status"] == "complete" for r in records),
        "records": sorted(records, key=lambda r: (r["date"], r["symbol"])), "seconds": time.monotonic() - begun,
        "model_fits": 0, "target_labels_decoded": False, "predictive_metrics_computed": False,
        "independent_confirmation_data_opened": False,
        "scope": "Reported source activity in the four historical training dates only. Candle closes are used for inventory validation, never as five-second predictors. Volume comparisons use the same base asset; quote currencies are not equated."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.protocol.resolve(), args.output.resolve())
