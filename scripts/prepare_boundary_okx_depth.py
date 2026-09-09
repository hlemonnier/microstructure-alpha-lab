"""Prepare registered counted-depth sidecars without decoding target labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.boundary_okx_depth import counted_archive_messages, sample_counted_depth

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def worker(protocol_path, output, index):
    protocol = json.loads(protocol_path.read_text())
    record = protocol["sources"][index]
    archive, original = ROOT / record["archive"], ROOT / record["features_path"]
    if digest(archive) != record["archive_sha256"] or digest(original) != record["features_sha256"]:
        raise ValueError("Frozen archive or original decision source changed")
    folder = output / record["symbol"] / record["date"]
    started = time.monotonic()
    clock = pd.read_parquet(original, columns=["decision_time"]).decision_time.to_numpy()
    if len(clock) > protocol["maximum_decision_rows"]:
        raise ValueError("The fixed dense grid exceeds its registered allocation")
    start_ms = int(datetime.fromisoformat(record["date"]).replace(tzinfo=timezone.utc).timestamp()) * 1000
    archive_audit = {}
    source = counted_archive_messages(archive, instrument=record["instrument"], start_ms=start_ms,
        end_ms=start_ms + 86400000, audit=archive_audit, maximum_member_bytes=protocol["maximum_member_bytes"],
        maximum_line_bytes=protocol["maximum_line_bytes"])

    def monitored_source():
        for count, message in enumerate(source, 1):
            if count % 250000 == 0:
                elapsed = time.monotonic() - started
                write(folder / "progress.json", {"messages": count, "seconds": elapsed,
                    "last_publisher_time": message.timestamp_ms, "market_model_fits": 0})
                print(f"source_messages={count} seconds={elapsed:.1f}", flush=True)
            yield message

    values, checks = sample_counted_depth(monitored_source(), clock, record["instrument"],
        depth=protocol["stored_depth"], delays_ms=tuple(protocol["delays_ms"]), capacity=protocol["visible_capacity"],
        maximum_gap_ms=protocol["maximum_publisher_gap_ms"], cadence_ms=protocol["query_cadence_ms"])
    if (checks["first_publisher_time"] > start_ms + 300000
        or checks["last_publisher_time"] < start_ms + 86400000 - 300000):
        raise ValueError("The registered full-day source extent is incomplete")
    if sum(v.nbytes for v in values.values()) > protocol["maximum_array_bytes"]:
        raise ValueError("Prepared arrays exceeded their registered byte budget")
    destination = folder / "observations.npz"
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with np.load(destination, allow_pickle=False) as restored:
        for key, expected in values.items():
            np.testing.assert_array_equal(restored[key], expected)
    complete = {str(depth): (values["known_depths"].min(axis=2) >= depth).sum(axis=0).tolist()
                for depth in (1, 25, 100)}
    result = {**record, "protocol_sha256": digest(protocol_path), "observation_path": str(destination.relative_to(ROOT)),
        "observation_sha256": digest(destination), "observation_bytes": destination.stat().st_size,
        "uncompressed_array_bytes": sum(v.nbytes for v in values.values()), "decision_rows": len(clock),
        "complete_rows_by_depth_and_delay": complete, "archive_audit": archive_audit, "book_checks": checks,
        "save_reload_exact": True, "original_read_columns": ["decision_time"], "assessment_labels_decoded": False,
        "market_model_fits": 0, "seconds": time.monotonic() - started}
    write(folder / "record.json", result)
    del values
    gc.collect()


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve existing preparation attempts; no implicit retry or resume")
    for path, checksum in protocol["input_hashes"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Frozen preparation input changed: {path}")
    if protocol["workers"] not in (1, 2) or protocol["stored_depth"] != 100 or protocol["visible_capacity"] != 400:
        raise ValueError("Only registered one/two-worker top-100 preparation is supported")
    keys = set()
    for record in protocol["sources"]:
        key = record["symbol"], record["date"]
        expected = {"BTCUSDT": "BTC-USDT-SWAP", "ETHUSDT": "ETH-USDT-SWAP"}
        if (key in keys or not "2023-05-29" <= record["date"] <= "2023-06-11"
            or expected.get(record["symbol"]) != record["instrument"]
            or protocol["input_hashes"].get(record["archive"]) != record["archive_sha256"]
            or protocol["input_hashes"].get(record["features_path"]) != record["features_sha256"]):
            raise ValueError("Each exposed development archive and decision grid must be unique and frozen")
        keys.add(key)
    if len(keys) != protocol["expected_sessions"] or not keys:
        raise ValueError("Preparation session inventory differs from its protocol")
    output.mkdir(parents=True)
    identity = {"protocol_sha256": digest(protocol_path), "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow")},
        "input_hashes": protocol["input_hashes"]}
    write(output / "frozen_preparation.json", identity)
    started = time.monotonic()

    def one(index, record):
        folder = output / record["symbol"] / record["date"]
        folder.mkdir(parents=True)
        begin = time.monotonic()
        try:
            with (folder / "worker.log").open("w") as log:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--protocol", str(protocol_path), "--output", str(output), "--worker", str(index)],
                    cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    stdout=log, stderr=subprocess.STDOUT, timeout=protocol["per_file_wall_seconds"], check=False)
            if completed.returncode:
                raise ValueError(f"Source preparation worker exited {completed.returncode}")
            result = {"status": "complete", "record": str((folder / "record.json").relative_to(ROOT)),
                "record_sha256": digest(folder / "record.json")}
        except Exception as exc:
            result = {"status": "failed", "exception": type(exc).__name__, "reason": str(exc)}
        result.update(symbol=record["symbol"], date=record["date"], seconds=time.monotonic() - begin,
                      worker_log_sha256=digest(folder / "worker.log"))
        write(folder / "supervision.json", result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        pending = [pool.submit(one, i, record) for i, record in enumerate(protocol["sources"])]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            write(output / "progress.json", {"completed_sessions": len(results), "planned_sessions": len(keys), "results": results})
            print(f"prepared_sources={len(results)}/{len(keys)} status={result['status']} {result['symbol']}/{result['date']}", flush=True)
    for path, checksum in protocol["input_hashes"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Preparation input changed during execution: {path}")
    complete = all(r["status"] == "complete" for r in results)
    sessions = [json.loads((ROOT / r["record"]).read_text()) for r in results if r["status"] == "complete"]
    write(output / "summary.json", {"identity": identity, "complete": complete,
        "sessions": sorted(sessions, key=lambda r: (r["date"], r["symbol"])), "supervision": results,
        "seconds": time.monotonic() - started, "market_model_fits": 0, "assessment_labels_decoded": False,
        "independent_confirmation_data_opened": False, "evidence_status": "exposed_development_source_preparation"})
    if not complete:
        raise ValueError("Preserve every failed and completed preparation; no silent coverage repair")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is None:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        worker(args.protocol.resolve(), args.output.resolve(), args.worker)
