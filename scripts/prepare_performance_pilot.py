#!/usr/bin/env python3
"""Prepare the preregistered eight-session performance pilot without reading its outcomes.

The default invocation only validates and prints the acquisition plan. ``--prepare``
downloads complete checksum-verified public archives, retains the immutable raw
two-hour-and-30-second UTC slice in that registration, and calls the current feature builder.
No dates, windows, labels, or model choices are selected from observed performance.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.binance_vision import archive_key, download_key, fetch_checksum, sha256_file, url_for_key
from lob_forge.experiments import (
    default_min_tick_for_symbol,
    expected_feature_build_config,
    feature_build_is_current,
    write_feature_build_marker,
)
from lob_forge.features import (
    AGG_TRADE_COLUMNS,
    BOOK_TICKER_COLUMNS,
    build_quote_trade_dataset,
    iter_quote_events,
    iter_zip_dict_rows,
)
from lob_forge.memory_guard import apply_process_memory_limit

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
DATES = tuple(f"2023-05-{day:02d}" for day in range(16, 24))
DATASETS = ("bookTicker", "aggTrades")
PREFIX_DURATION_MS = 7_230_000
MAX_COMPRESSED_BYTES = 2_000_000_000
PREPARATION_VERSION = 1


def resident_memory_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def start_resident_memory_watchdog(limit_bytes: int) -> threading.Event:
    """Abort within one second of observing excessive RSS when macOS denies rlimits.

    This is a monitored resident-memory budget, not a claimed hard address-space
    limit. Partial files never carry a completed dataset manifest.
    """
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(1.0):
            observed = resident_memory_bytes()
            if observed > limit_bytes:
                print(f"resident_memory_budget_exceeded bytes={observed} limit={limit_bytes}",
                      file=sys.stderr, flush=True)
                os._exit(70)

    threading.Thread(target=monitor, name="preparation-memory-watchdog", daemon=True).start()
    return stop


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def preparation_code_hash() -> str:
    """Bind preprocessing only; independently frozen model experiments may evolve."""
    paths = [Path(__file__).resolve(), *[
        ROOT / "src" / "lob_forge" / name
        for name in ("features.py", "experiments.py", "binance_vision.py", "memory_guard.py")
    ]]
    digest = hashlib.sha256()
    for path in sorted(paths):
        # User backup files are not importable model code.
        if not path.stem.isidentifier():
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validated_archives(plan: dict) -> list[dict]:
    expected = {(symbol, day, dataset) for symbol in SYMBOLS for day in DATES for dataset in DATASETS}
    entries = plan.get("archives", [])
    observed: set[tuple[str, str, str]] = set()
    total = 0
    for entry in entries:
        identity = (entry["symbol"], entry["session_date"], entry["dataset"])
        if identity not in expected or identity in observed:
            raise ValueError(f"unexpected or duplicate archive: {identity}")
        observed.add(identity)
        key = archive_key(
            market="futures/um", frequency="daily", dataset=identity[2], symbol=identity[0], date_value=identity[1]
        )
        if entry.get("key") != key or entry.get("url") != url_for_key(key):
            raise ValueError("archive source must be the canonical Binance USD-M public URL")
        if entry.get("checksum_url") != url_for_key(key + ".CHECKSUM"):
            raise ValueError("archive checksum URL differs from canonical provider URL")
        if entry.get("status") != 200 or not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise ValueError("each archive requires successful metadata and a positive byte size")
        total += entry["bytes"]
    if observed != expected:
        raise ValueError("plan must contain exactly the 32 preregistered symbol/date/dataset archives")
    if total > MAX_COMPRESSED_BYTES or plan.get("total_compressed_bytes") != total:
        raise ValueError("archive plan exceeds the 2 GB compressed cap or has an inconsistent total")
    return sorted(entries, key=lambda e: (e["symbol"], e["session_date"], e["dataset"]))


def utc_offset_ms(value: str) -> int:
    parsed = datetime.strptime(value, "%H:%M:%S")
    return ((parsed.hour * 60 + parsed.minute) * 60 + parsed.second) * 1000


def bind_acquisition(output_root: Path, protocol: Path, plan: Path) -> dict:
    if not protocol.is_file() or not isinstance(json.loads(protocol.read_text()), dict):
        raise ValueError("a written JSON preregistration is required before acquisition")
    registered = json.loads(protocol.read_text())
    data = registered.get("data", {})
    features = registered.get("features", {})
    required_data = {
        "symbols": list(SYMBOLS), "start_date": DATES[0], "end_date": DATES[-1],
        "max_compressed_download_bytes": MAX_COMPRESSED_BYTES,
        "max_parallel_downloads": 2, "require_official_sha256": True,
    }
    required_features = {
        "bucket_ms": 1000, "horizon_ms": 5000, "execution_latency_ms": 100, "with_book_depth": False,
    }
    if any(data.get(key) != value for key, value in required_data.items()):
        raise ValueError("preregistered data scope differs from this fixed preparation protocol")
    if any(features.get(key) != value for key, value in required_features.items()):
        raise ValueError("preregistered feature settings differ from this fixed preparation protocol")
    if utc_offset_ms(data["slice_end_utc"]) - utc_offset_ms(data["slice_start_utc"]) != PREFIX_DURATION_MS:
        raise ValueError("preregistered slice must span exactly two hours and 30 seconds inside one UTC day")
    identity = {
        "preparation_version": PREPARATION_VERSION,
        "protocol_sha256": sha256_file(protocol),
        "acquisition_plan_sha256": sha256_file(plan),
        "prefix_start_utc": data["slice_start_utc"],
        "prefix_end_exclusive_utc": data["slice_end_utc"],
        "symbols": list(SYMBOLS),
        "dates": list(DATES),
        "bucket_ms": 1000,
        "horizon_ms": 5000,
        "execution_latency_ms": 100,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "acquisition_lock.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("preregistration or acquisition scope changed after acquisition was started")
    else:
        # Exclusive creation prevents two processes silently replacing the registration.
        with path.open("x") as handle:
            handle.write(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity


def acquire_original(entry: dict, raw_root: Path, source_record: Path) -> tuple[Path, dict]:
    destination = raw_root / entry["key"]
    if source_record.exists() and destination.exists():
        cached = json.loads(source_record.read_text())
        if (
            cached.get("url") == entry["url"]
            and cached.get("bytes") == entry["bytes"] == destination.stat().st_size
            and cached.get("provider_checksum_sha256") == cached.get("sha256") == sha256_file(destination)
        ):
            return destination, cached
    expected = fetch_checksum(entry["key"])
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError(f"provider SHA-256 is missing or invalid for {entry['key']}")
    destination = download_key(entry["key"], root=raw_root, verify_checksum=True)
    actual = sha256_file(destination)
    if actual.lower() != expected.lower() or destination.stat().st_size != entry["bytes"]:
        raise ValueError(f"download differs from pinned metadata/provider checksum: {entry['key']}")
    record = {
        "key": entry["key"],
        "url": entry["url"],
        "checksum_url": entry["checksum_url"],
        "provider_checksum_sha256": expected.lower(),
        "sha256": actual,
        "bytes": destination.stat().st_size,
        "local_path": str(destination.resolve()),
    }
    atomic_json(source_record, record)
    return destination, record


def extract_prefix(source: Path, destination: Path, *, dataset: str, start_ms: int, end_ms: int) -> dict:
    """Stream and preserve all source columns; fixed ZIP metadata makes bytes reproducible."""
    if dataset not in DATASETS or not 0 <= start_ms < end_ms:
        raise ValueError("invalid prefix dataset or half-open time interval")
    columns = BOOK_TICKER_COLUMNS if dataset == "bookTicker" else AGG_TRADE_COLUMNS
    time_column = "event_time" if dataset == "bookTicker" else "transact_time"
    id_column = "update_id" if dataset == "bookTicker" else "agg_trade_id"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    first_time = last_time = after_end = None
    previous_key = None
    count = 0
    with zipfile.ZipFile(source) as archive:
        if len([name for name in archive.namelist() if name.endswith(".csv")]) != 1:
            raise ValueError("source archive must contain exactly one CSV")
    try:
        info = zipfile.ZipInfo(dataset + ".csv", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            with archive.open(info, "w", force_zip64=True) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
                    writer.writeheader()
                    for row in iter_zip_dict_rows(source, columns):
                        time_ms = int(row[time_column])
                        key = (time_ms, int(row[id_column]))
                        if previous_key is not None and (key <= previous_key or key[1] <= previous_key[1]):
                            raise ValueError(f"{dataset} prefix source has nonmonotonic time/IDs or duplicates")
                        previous_key = key
                        if time_ms >= end_ms:
                            after_end = time_ms
                            break
                        if time_ms < start_ms:
                            continue
                        writer.writerow(row)
                        count += 1
                        first_time = time_ms if first_time is None else first_time
                        last_time = time_ms
        if count == 0 or after_end is None:
            raise ValueError("source does not demonstrate a nonempty prefix and coverage through its end")
        if dataset == "bookTicker":
            # Exercise the same finite-price, uncrossed-book, timestamp and ID contract as the builder.
            for _ in iter_quote_events(temporary):
                pass
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination.resolve()), "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size, "rows": count,
        "start_inclusive_ms": start_ms, "end_exclusive_ms": end_ms,
        "first_event_time_ms": first_time, "last_event_time_ms": last_time,
        "first_excluded_event_time_ms": after_end,
    }


def feature_metadata(path: Path) -> dict:
    first = last = future_max = None
    count = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            decision = int(row["decision_time"])
            if last is not None and decision <= last:
                raise ValueError("feature decisions must be strictly increasing")
            count += 1
            first = decision if first is None else first
            last = decision
            future_max = max(future_max or 0, int(row["future_event_time"]))
    if count == 0:
        raise ValueError("feature builder produced no decisions")
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path), "rows": count,
        "first_decision_time_ms": first, "last_decision_time_ms": last,
        "maximum_label_endpoint_ms": future_max,
    }


def prepare_session(entries: list[dict], *, output_root: Path, raw_root: Path, identity: dict, code_hash: str) -> dict:
    symbol, day = entries[0]["symbol"], entries[0]["session_date"]
    output = output_root / symbol / day
    output.mkdir(parents=True, exist_ok=True)
    start = (int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
             + utc_offset_ms(identity["prefix_start_utc"]))
    end = start + PREFIX_DURATION_MS
    sources, prefixes = {}, {}
    prefix_manifest = output / "prefix_manifest.json"
    cached = json.loads(prefix_manifest.read_text()) if prefix_manifest.exists() else {}
    for entry in entries:
        dataset = entry["dataset"]
        original, sources[dataset] = acquire_original(entry, raw_root, output / f"{dataset}.source.json")
        prefix = output / f"{dataset}.zip"
        old = cached.get("prefixes", {}).get(dataset, {})
        if (
            cached.get("preparation_code_sha256") == code_hash
            and cached.get("acquisition_identity") == identity
            and cached.get("sources", {}).get(dataset) == sources[dataset]
            and old.get("start_inclusive_ms") == start and old.get("end_exclusive_ms") == end
            and prefix.exists() and old.get("sha256") == sha256_file(prefix)
        ):
            prefixes[dataset] = old
        else:
            prefixes[dataset] = extract_prefix(original, prefix, dataset=dataset, start_ms=start, end_ms=end)
        print(f"prefix_ready symbol={symbol} date={day} dataset={dataset} rows={prefixes[dataset]['rows']}", flush=True)
    prefix_payload = {
        "sources": sources, "prefixes": prefixes, "acquisition_identity": identity,
        "preparation_code_sha256": code_hash,
    }
    atomic_json(prefix_manifest, prefix_payload)
    feature = output / "features.csv"
    marker = feature.with_suffix(".csv.done")
    build_config = expected_feature_build_config(
        symbol=symbol, date_value=day, bucket_ms=1000, horizon_ms=5000, execution_latency_ms=100,
        threshold="half_spread", min_tick=default_min_tick_for_symbol(symbol), large_trade_notional=10_000.0,
        max_quote_buckets=None, with_book_depth=False, execution_quote_resolution="raw",
    )
    inputs = {
        "book_ticker_sha256": prefixes["bookTicker"]["sha256"],
        "agg_trades_sha256": prefixes["aggTrades"]["sha256"], "book_depth_sha256": None,
    }
    if not feature_build_is_current(feature, marker, build_config=build_config, input_hashes=inputs):
        temporary = feature.with_name("features.csv.partial")
        build_quote_trade_dataset(
            book_ticker_zip=output / "bookTicker.zip", agg_trades_zip=output / "aggTrades.zip",
            output_csv=temporary, bucket_ms=1000, horizon_ms=5000, execution_latency_ms=100,
            min_tick=default_min_tick_for_symbol(symbol), execution_quote_resolution="raw",
            max_feature_build_memory_gb=2.0,
        )
        temporary.replace(feature)
        write_feature_build_marker(feature, marker, build_config=build_config, input_hashes=inputs)
    result = {
        "symbol": symbol, "session_date": day, **prefix_payload,
        "feature": feature_metadata(feature), "feature_build_config": build_config,
        "feature_marker_sha256": sha256_file(marker),
        "features_path": str(feature.resolve()),
        "book_ticker_path": str((output / "bookTicker.zip").resolve()),
        "agg_trades_path": str((output / "aggTrades.zip").resolve()),
    }
    result["feature_rows"] = result["feature"]["rows"]
    atomic_json(output / "session_manifest.json", result)
    print(f"features_ready symbol={symbol} date={day} rows={result['feature']['rows']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/research/performance_pilot_20260907")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    entries = validated_archives(plan)
    if not args.protocol.is_file():
        raise ValueError("preregistration must exist before preparation")
    print(json.dumps({"archives": len(entries), "compressed_bytes": sum(e["bytes"] for e in entries),
                      "workers": args.workers, "prepare": args.prepare, "protocol_sha256": sha256_file(args.protocol)}), flush=True)
    if not args.prepare:
        return 0
    memory_limit = apply_process_memory_limit(4.0)
    memory_stop = start_resident_memory_watchdog(4_000_000_000)
    print(f"memory_guard resident_watchdog_bytes=4000000000 os_resource_limit_gb={memory_limit}", flush=True)
    identity = bind_acquisition(args.output_root, args.protocol, args.plan)
    code_hash = preparation_code_hash()
    # Preserve a self-contained plan copy; never rely on the initial /tmp metadata path.
    plan_copy = args.output_root / "acquisition_plan.json"
    if args.plan.resolve() != plan_copy.resolve():
        if plan_copy.exists() and sha256_file(plan_copy) != sha256_file(args.plan):
            raise ValueError("output already contains a different acquisition plan")
        if not plan_copy.exists():
            plan_copy.write_bytes(args.plan.read_bytes())
    groups = [[e for e in entries if e["symbol"] == symbol and e["session_date"] == day]
              for symbol in SYMBOLS for day in DATES]
    sessions, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare_session, group, output_root=args.output_root, raw_root=args.raw_root,
                                   identity=identity, code_hash=code_hash) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            try:
                sessions.append(future.result())
            except Exception as error:
                index = futures.index(future)
                item = {"symbol": groups[index][0]["symbol"], "session_date": groups[index][0]["session_date"],
                        "error": repr(error)}
                failures.append(item)
                print(f"session_failed {json.dumps(item)}", file=sys.stderr, flush=True)
    if failures:
        atomic_json(args.output_root / "preparation_failures.json", {"failures": failures, "complete_sessions": len(sessions),
                                                                   "acquisition_identity": identity})
        raise ValueError(f"{len(failures)} session(s) failed preparation; dataset manifest not created")
    # Dependencies can be edited concurrently in a shared workspace; do not silently mix builds.
    if preparation_code_hash() != code_hash:
        raise ValueError("preparation/source code changed during acquisition; rerun to regenerate consistent artifacts")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "manifest_version": 1, "acquisition_identity": identity,
        "preparation_code_sha256": code_hash, "git_commit": commit,
        "total_compressed_source_bytes": sum(e["bytes"] for e in entries),
        "os_resource_limit_gb": memory_limit,
        "resident_memory_watchdog_limit_bytes": 4_000_000_000,
        "resident_memory_peak_bytes": resident_memory_bytes(), "preparation_workers": args.workers,
        "sessions": sorted(sessions, key=lambda item: (item["symbol"], item["session_date"])),
        "scope": "Historical Binance USD-M BBO and aggregate trades; prefix time window only; no empirical performance assessed during preparation.",
    }
    atomic_json(args.output_root / "dataset_manifest.json", manifest)
    memory_stop.set()
    print(f"dataset_manifest={args.output_root / 'dataset_manifest.json'} sessions={len(sessions)}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LOB_FORGE_DOWNLOAD_VERBOSE", "1")
    raise SystemExit(main())
