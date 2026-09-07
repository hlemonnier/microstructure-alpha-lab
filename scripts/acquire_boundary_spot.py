"""Acquire a frozen, bounded set of public spot trade archives with official hashes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import archive_key, download_key, fetch_checksum, sha256_file, url_for_key
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_spot_data import read_spot_trades
from prepare_boundary_event_clock import write_json

ROOT = Path(__file__).resolve().parents[1]


def acquire(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError("Registered spot acquisition code changed")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    frozen = output / "frozen_acquisition.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing spot acquisition after source changes")
    write_json(frozen, identity)
    records, total_bytes = [], 0
    for symbol in protocol["symbols"]:
        for day in protocol["dates"]:
            key = archive_key(market="spot", frequency="daily", dataset="aggTrades", symbol=symbol, date_value=day)
            metadata = output / "metadata" / symbol / f"{day}.json"
            if metadata.exists():
                record = json.loads(metadata.read_text())
                if record["identity"] != identity or sha256_file(ROOT / record["local_path"]) != record["sha256"]:
                    raise ValueError("Cached spot archive changed")
            else:
                started = time.monotonic()
                request = urllib.request.Request(url_for_key(key), method="HEAD")
                with urllib.request.urlopen(request, timeout=30) as response:
                    size = int(response.headers["Content-Length"])
                    last_modified = response.headers.get("Last-Modified")
                if size <= 0 or total_bytes + size > protocol["maximum_total_compressed_bytes"]:
                    raise ValueError("Registered spot acquisition size bound exceeded")
                expected = fetch_checksum(key)
                if not expected or len(expected) != 64:
                    raise ValueError("An official spot SHA256 is mandatory")
                path = download_key(key, root=output / "archives", verify_checksum=True)
                actual = sha256_file(path)
                if actual != expected or path.stat().st_size != size:
                    raise ValueError("Spot archive must match official SHA256 and advertised bytes")
                trades = read_spot_trades(path, day_start_ms=utc_ms(day))
                timestamps = trades.transact_time.to_numpy()
                if timestamps[0] - utc_ms(day) > 300000 or utc_ms(day) + 86400000 - timestamps[-1] > 300000:
                    raise ValueError("Spot trade coverage must reach both day boundaries within five minutes")
                record = {"symbol": symbol, "session_date": day, "identity": identity, "key": key, "url": url_for_key(key),
                          "local_path": str(path.relative_to(ROOT)), "bytes": size, "sha256": actual, "provider_checksum_sha256": expected,
                          "last_modified": last_modified, "acquired_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(trades),
                          "first_trade_time_ms": int(timestamps[0]), "last_trade_time_ms": int(timestamps[-1]),
                          "maximum_intertrade_gap_ms": int(np.diff(timestamps).max(initial=0)),
                          "aggregate_id_gaps": int((np.diff(trades.agg_trade_id.to_numpy()) != 1).sum()),
                          "timestamp_unit": "milliseconds", "schema": "eight_column_headerless_spot_aggTrades", "seconds": time.monotonic() - started}
                write_json(metadata, record)
                del trades
            total_bytes += record["bytes"]
            if total_bytes > protocol["maximum_total_compressed_bytes"]:
                raise ValueError("Registered total size bound exceeded")
            records.append(record)
            print(f"spot_acquired={len(records)}/{len(protocol['dates'])*len(protocol['symbols'])} {symbol}/{day} bytes={record['bytes']}", flush=True)
    write_json(output / "source_manifest.json", {"identity": identity, "sessions": records, "compressed_bytes": total_bytes,
               "evidence_status": "new_observation_source_on_exposed_dates", "forecast_only": True})
    print(f"spot_acquisition_complete {output / 'source_manifest.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_spot_acquisition_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_spot_raw_20260907")
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if args.acquire:
        acquire(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --acquire for the fixed set of public spot archives.")
