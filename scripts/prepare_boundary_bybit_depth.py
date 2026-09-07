"""Acquire bounded public Bybit archives and prepare atomic depth sidecars."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_bybit_depth import bybit_archive_messages, sample_bybit_depth
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.l2_ingest import build_historical_l2_manifest, fetch_bybit_historical_download_details
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered Bybit preparation input changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {k: version(k) for k in ("numpy", "pandas", "pyarrow")}}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing depth artifacts after an identity change")
    write_json(frozen, identity)
    original_manifest = json.loads((ROOT / protocol["source_manifest"]).read_text())
    original = {(r["symbol"], r["session_date"]): r for r in original_manifest["sessions"]}
    entries = build_historical_l2_manifest(source_id="bybit", symbols=protocol["symbols"],
        start=protocol["dates"][0], end=protocol["dates"][-1])
    if {e.session_date for e in entries} != set(protocol["dates"]):
        raise ValueError("The registered preparation requires a complete contiguous date range")
    resolved, total_bytes = [], 0
    for entry in entries:
        destination = output / "provider_metadata" / entry.symbol / f"{entry.session_date}.json"
        if destination.exists():
            result = json.loads(destination.read_text())
        else:
            body = fetch_bybit_historical_download_details(entry, 30)
            result = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "body": body}
            write_json(destination, result)
            time.sleep(0.5)
        rows = result["body"].get("list", [])
        matches = [r for r in rows if r.get("symbol") == entry.symbol and r.get("date") == entry.session_date
                   and r.get("filename") == f"{entry.session_date}_{entry.symbol}_ob500.data.zip"]
        if len(matches) != 1:
            raise ValueError(f"Exactly one registered 500-level archive required: {entry.symbol}/{entry.session_date}")
        record = matches[0]
        parsed = urlparse(record["url"])
        size = int(record["size"])
        if parsed.scheme != "https" or parsed.hostname != "quote-saver.bycsi.com" or size <= 0:
            raise ValueError("Positive byte count and official HTTPS archive host required")
        total_bytes += size
        resolved.append({"symbol": entry.symbol, "session_date": entry.session_date, "provider": record,
            "provider_metadata_path": str(destination.relative_to(ROOT)), "provider_metadata_sha256": sha256_file(destination)})
    if total_bytes > protocol["max_final_archive_bytes"]:
        raise ValueError(f"Registered archive budget exceeded: {total_bytes}")
    write_json(output / "resolved_sources.json", {"identity": identity, "total_archive_bytes": total_bytes, "sessions": resolved})
    print(f"bybit_sources_resolved={len(resolved)} archive_bytes={total_bytes}", flush=True)
    prepared = []
    for entry in resolved:
        begin = time.monotonic()
        symbol, day, provider = entry["symbol"], entry["session_date"], entry["provider"]
        metadata_path = output / "observations" / symbol / f"{day}.json"
        if metadata_path.exists():
            cached = json.loads(metadata_path.read_text())
            if cached["identity"] != identity or sha256_file(ROOT / cached["observation_path"]) != cached["observation_sha256"] or sha256_file(ROOT / cached["archive_path"]) != cached["archive_sha256"]:
                raise ValueError("Cached Bybit depth observations changed")
            prepared.append(cached)
            continue
        reused = protocol.get("verified_existing_archives", {}).get(f"{symbol}/{day}")
        archive = ROOT / reused["path"] if reused else output / "archives" / symbol / provider["filename"]
        if reused and sha256_file(archive) != reused["sha256"]:
            raise ValueError("Previously verified source archive changed")
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_name(archive.name + ".partial")
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--proto", "=https", "--proto-redir", "=https",
                "--max-time", "300", "--max-filesize", str(int(provider["size"])), "--output", str(partial), provider["url"]], check=True)
            if partial.stat().st_size != int(provider["size"]):
                raise ValueError("Archive byte count differs from the official resolver")
            partial.replace(archive)
        if archive.stat().st_size != int(provider["size"]):
            raise ValueError("Existing archive byte count differs from provider metadata")
        reference = original[symbol, day]
        reference_path = ROOT / reference["features_path"]
        if sha256_file(reference_path) != reference["sha256"]:
            raise ValueError("Original Binance decision source changed")
        clock = pd.read_parquet(reference_path, columns=["decision_time"]).decision_time.to_numpy()
        values, checks = sample_bybit_depth(bybit_archive_messages(archive), clock, symbol,
            depth=protocol["depth"], delays_ms=tuple(protocol["added_publisher_delays_ms"]))
        if checks["first_publisher_time"] > utc_ms(day) + 300000 or checks["last_publisher_time"] < utc_ms(day) + 86400000 - 300000:
            raise ValueError("Required full-day depth coverage is missing")
        destination = output / "observations" / symbol / f"{day}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **values)
        partial.replace(destination)
        with np.load(destination, allow_pickle=False) as restored:
            for key, value in values.items():
                np.testing.assert_array_equal(restored[key], value)
        record = {**entry, "identity": identity, "archive_path": str(archive.relative_to(ROOT)), "archive_sha256": sha256_file(archive),
            "archive_bytes": archive.stat().st_size, "official_byte_count_parity": True, "zip_crc_verified_by_full_member_read": True,
            "observation_path": str(destination.relative_to(ROOT)), "observation_sha256": sha256_file(destination),
            "original_features_path": reference["features_path"], "original_features_sha256": reference["sha256"],
            "only_decision_time_column_read": True, "rows": len(clock), "complete_depth_rows": values["available"].sum(axis=0).tolist(),
            "save_reload_parity": True, "checks": checks, "seconds": time.monotonic() - begin}
        write_json(metadata_path, record)
        prepared.append(record)
        write_json(output / "progress.json", {"prepared_sessions": len(prepared), "planned_sessions": len(resolved), "latest_session": [symbol, day]})
        print(f"bybit_depth_prepared={len(prepared)}/{len(resolved)} {symbol}/{day} rows={len(clock)} seconds={record['seconds']:.1f}", flush=True)
        del values
        gc.collect()
    write_json(output / "depth_manifest.json", {"identity": identity, "sessions": prepared,
        "evidence_status": "development_only_on_exposed_dates", "forecast_only": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_bybit_depth_data_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_bybit_depth_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare to acquire and validate the fixed exposed-date depth observations.")
