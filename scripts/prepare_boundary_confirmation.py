"""Acquire the fixed independent evaluation sources without inspecting model performance."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import resource
import shutil
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import archive_key, download_key, fetch_checksum, sha256_file, url_for_key
from lob_forge.boundary_event_inputs import load_event_inputs, utc_ms
from lob_forge.boundary_events import build_event_dataset

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def acquire(entry):
    destination = ROOT / "data/raw" / entry["key"]
    cached = destination.exists()
    if cached:
        if (
            destination.stat().st_size != entry["bytes"]
            or sha256_file(destination) != entry["provider_checksum_sha256"]
        ):
            raise ValueError("Preserve preexisting raw data: cached archive differs from the frozen source plan")
    else:
        if fetch_checksum(entry["key"]) != entry["provider_checksum_sha256"]:
            raise ValueError("Provider checksum changed after metadata registration")
        destination = download_key(entry["key"], root=ROOT / "data/raw", verify_checksum=True)
        if (
            destination.stat().st_size != entry["bytes"]
            or sha256_file(destination) != entry["provider_checksum_sha256"]
        ):
            raise ValueError("Downloaded source differs from registered size or official checksum")
    return {
        **entry,
        "local_path": str(destination),
        "sha256": entry["provider_checksum_sha256"],
        "cached_at_start": cached,
    }


def start_watchdog(limit, output):
    stop = threading.Event()

    def monitor():
        while not stop.wait(1):
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
            if peak_bytes > limit:
                write_json(
                    output / "failure.json",
                    {"stage": "preparation_memory_budget", "peak_bytes": peak_bytes, "limit_bytes": limit},
                )
                os._exit(70)

    threading.Thread(target=monitor, daemon=True, name="confirmation-preparation-memory").start()
    return stop


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    plan_path = ROOT / protocol["source_plan"]["path"]
    if sha256_file(plan_path) != protocol["source_plan"]["sha256"]:
        raise ValueError("Frozen source metadata changed")
    for name, checksum in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Registered model/feature source changed before acquisition")
    entries = json.loads(plan_path.read_text())["archives"]
    first = date.fromisoformat(protocol["source_plan"]["first_date"])
    days = [(first + timedelta(days=offset)).isoformat() for offset in range(20)]
    expected = {
        (symbol, day, dataset)
        for symbol in ["BTCUSDT", "ETHUSDT"]
        for day in days
        for dataset in ["bookTicker", "aggTrades"]
    }
    observed = {(e["symbol"], e["session_date"], e["dataset"]) for e in entries}
    if observed != expected or len(entries) != len(expected):
        raise ValueError("The source plan must contain exactly the fixed eighty archives")
    for entry in entries:
        key = archive_key(
            market="futures/um",
            frequency="daily",
            dataset=entry["dataset"],
            symbol=entry["symbol"],
            date_value=entry["session_date"],
        )
        if (
            entry["key"] != key
            or entry["url"] != url_for_key(key)
            or entry["checksum_url"] != url_for_key(key + ".CHECKSUM")
        ):
            raise ValueError("Only canonical public Binance source URLs are admitted")
    total = sum(entry["bytes"] for entry in entries)
    if (
        total != protocol["source_plan"]["compressed_bytes"]
        or total > protocol["resources"]["max_new_compressed_bytes"]
    ):
        raise ValueError("Source size differs from the registration or exceeds its budget")
    if shutil.disk_usage(ROOT).free < protocol["resources"]["minimum_free_bytes"]:
        raise ValueError("Insufficient free space for the registered acquisition and artifacts")
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "source_plan_sha256": sha256_file(plan_path),
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/prepare_boundary_confirmation.py",
                "src/lob_forge/boundary_events.py",
                "src/lob_forge/boundary_event_inputs.py",
                "src/lob_forge/binance_vision.py",
            ]
        },
    }
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this preparation and use a new output directory after source changes")
    write_json(frozen, identity)
    stop = start_watchdog(protocol["resources"]["monitored_preparation_rss_bytes"], output)
    originals, sessions = {}, []
    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=protocol["resources"]["max_parallel_downloads"]
        ) as executor:
            for number, source in enumerate(executor.map(acquire, entries), 1):
                originals[source["symbol"], source["session_date"], source["dataset"]] = source
                write_json(
                    output / "acquisition_progress.json",
                    {
                        "verified_archives": number,
                        "planned_archives": 80,
                        "verified_compressed_bytes": sum(s["bytes"] for s in originals.values()),
                        "seconds": time.monotonic() - started,
                    },
                )
                print(
                    f"archives_verified={number}/80 {source['symbol']} {source['session_date']} {source['dataset']}",
                    flush=True,
                )
        for day in days:
            daily_sessions = []
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                metadata = output / symbol / f"{day}.json"
                path = output / symbol / f"{day}.parquet"
                if metadata.exists():
                    record = json.loads(metadata.read_text())
                    if record["identity"] != identity or sha256_file(path) != record["sha256"]:
                        raise ValueError("Prepared confirmation data identity mismatch")
                else:
                    sources = {kind: originals[symbol, day, kind] for kind in ["bookTicker", "aggTrades"]}
                    frame = build_event_dataset(
                        Path(sources["bookTicker"]["local_path"]),
                        Path(sources["aggTrades"]["local_path"]),
                        min_tick=protocol["data_contract"]["min_tick"][symbol],
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".partial")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    record = {
                        "identity": identity,
                        "symbol": symbol,
                        "session_date": day,
                        "features_path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                        "rows": len(frame),
                        "first_decision_time_ms": int(frame.decision_time.min()),
                        "last_decision_time_ms": int(frame.decision_time.max()),
                        "sources": sources,
                    }
                    write_json(metadata, record)
                    del frame
                    gc.collect()
                daily_sessions.append(record)
            daily_manifest = output / "coverage_checks" / f"{day}.json"
            write_json(daily_manifest, {"sessions": daily_sessions})
            features, labels, times = load_event_inputs(ROOT, daily_manifest)
            expected_times = np.arange(utc_ms(day, "12:02:00"), utc_ms(day, "13:59:50"), 1000)
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                actual = times[symbol, day]
                actual = actual[(actual >= expected_times[0]) & (actual <= expected_times[-1])]
                if not np.array_equal(actual, expected_times):
                    raise ValueError(
                        f"Registered noon coverage failed for {symbol} {day}; keep the date and repair the study before fitting"
                    )
            sessions.extend(daily_sessions)
            write_json(
                output / "preparation_progress.json",
                {
                    "prepared_sessions": len(sessions),
                    "planned_sessions": 40,
                    "complete_noon_dates": len(sessions) // 2,
                    "seconds": time.monotonic() - started,
                },
            )
            print(f"sessions_prepared={len(sessions)}/40 coverage_verified={day}", flush=True)
            del features, labels, times
            gc.collect()
        historical_path = ROOT / protocol["prior_development_data"]["manifest"]
        if sha256_file(historical_path) != protocol["prior_development_data"]["sha256"]:
            raise ValueError("Historical training dataset changed")
        historical = json.loads(historical_path.read_text())["sessions"]
        write_json(
            output / "dataset_manifest.json",
            {
                "identity": identity,
                "sessions": sessions,
                "forecast_only": True,
                "confirmation_dates": days,
                "performance_inspected": False,
            },
        )
        write_json(
            output / "combined_dataset_manifest.json",
            {
                "identity": identity,
                "sessions": historical + sessions,
                "development_dates": sorted({s["session_date"] for s in historical}),
                "confirmation_dates": days,
                "performance_inspected": False,
                "forecast_only": True,
            },
        )
        print("confirmation_sources_prepared_no_performance_inspected", flush=True)
    except Exception as error:
        write_json(
            output / "failure.json",
            {"stage": "confirmation_preparation", "error": str(error), "sessions_prepared": len(sessions)},
        )
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_confirmation_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_confirmation_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol, args.output)
    else:
        print("Pass --prepare for the fixed twenty-date checksum-verified confirmation acquisition.")
