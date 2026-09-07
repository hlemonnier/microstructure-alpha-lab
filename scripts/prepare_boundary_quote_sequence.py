"""Materialize fixed raw-quote histories for exposed development observations."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_events import read_archive
from lob_forge.boundary_quote_sequence import (
    EVENT_STRIDES,
    QUOTE_SEQUENCE_SEMANTICS,
    SEQUENCE_CHANNELS,
    SEQUENCE_LENGTH,
    QuoteSequenceSource,
)
from lob_forge.features import BOOK_TICKER_COLUMNS
from run_boundary_confirmation import noon, write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered quote-sequence preparation input changed: {name}")
    source = json.loads((ROOT / protocol["source_manifest"]).read_text())
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
                "python": platform.python_version(), "dependencies": {k: version(k) for k in ("numpy", "pandas", "pyarrow")}}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing sequence artifacts after an identity change")
    write_json(frozen, identity)
    requested = [s for s in source["sessions"] if s["session_date"] in protocol["dates"] and s["symbol"] in protocol["symbols"]]
    if len(requested) != len(protocol["dates"]) * len(protocol["symbols"]):
        raise ValueError("Every registered exposed quote session is required")
    sessions = []
    for original in requested:
        begin = time.monotonic()
        symbol, day = original["symbol"], original["session_date"]
        metadata = output / symbol / f"{day}.json"
        if metadata.exists():
            cached = json.loads(metadata.read_text())
            if cached["identity"] != identity or any(sha256_file(ROOT / name) != checksum for name, checksum in cached["artifact_hashes"].items()):
                raise ValueError("Cached quote-sequence artifacts changed")
            sessions.append(cached)
            continue
        canonical_path = ROOT / original["features_path"]
        if sha256_file(canonical_path) != original["sha256"]:
            raise ValueError("Original observation session changed")
        raw = original["sources"]["bookTicker"]
        raw_path = Path(raw["local_path"])
        if raw_path.stat().st_size != raw["bytes"] or sha256_file(raw_path) != raw["sha256"] or raw["sha256"] != raw["provider_checksum_sha256"]:
            raise ValueError("Raw quotes require official byte-count and SHA256 parity")
        canonical = pd.read_parquet(canonical_path, columns=["decision_time", "bid", "ask", "bid_qty", "ask_qty"])
        clocks = canonical.decision_time.to_numpy()
        selected = ((clocks - utc_ms(day)) % 4000 == 0) | noon(clocks, day)
        canonical = canonical.loc[selected].reset_index(drop=True)
        clocks = canonical.decision_time.to_numpy()
        raw_frame = read_archive(raw_path, BOOK_TICKER_COLUMNS)
        history = QuoteSequenceSource.from_frame(raw_frame)
        last = np.searchsorted(history.times, clocks, side="left") - 1
        for values, column in ((history.bid, "bid"), (history.ask, "ask"), (history.bid_quantity, "bid_qty"), (history.ask_quantity, "ask_qty")):
            np.testing.assert_array_equal(values[last], canonical[column].to_numpy())
        del raw_frame, canonical
        gc.collect()
        directory = output / symbol
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{day}.npy"
        temporary = path.with_name(path.name + ".partial")
        shape = (len(clocks), len(EVENT_STRIDES), SEQUENCE_LENGTH, len(SEQUENCE_CHANNELS))
        saved = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, len(clocks), 4096):
            saved[start : start + 4096] = history.observe(clocks[start : start + 4096])
        saved.flush()
        del saved
        temporary.replace(path)
        saved = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(clocks), 4096):
            np.testing.assert_array_equal(saved[start : start + 4096], history.observe(clocks[start : start + 4096]))
        time_path = directory / f"{day}_times.npy"
        with time_path.open("wb") as handle:
            np.save(handle, clocks, allow_pickle=False)
        np.testing.assert_array_equal(np.load(time_path, allow_pickle=False), clocks)
        record = {"identity": identity, "symbol": symbol, "session_date": day, "rows": len(clocks),
            "array_path": str(path.relative_to(ROOT)), "times_path": str(time_path.relative_to(ROOT)),
            "shape": list(shape), "semantics": QUOTE_SEQUENCE_SEMANTICS, "channels": list(SEQUENCE_CHANNELS),
            "original_features_path": original["features_path"], "original_features_sha256": original["sha256"],
            "source": raw, "current_raw_quote_parity": True, "save_reload_parity": True,
            "only_observation_columns_read": True, "seconds": time.monotonic() - begin,
            "artifact_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in (path, time_path)}}
        write_json(metadata, record)
        sessions.append(record)
        write_json(output / "progress.json", {"completed_sessions": len(sessions), "planned_sessions": len(requested), "latest_session": [symbol, day]})
        print(f"quote_sequence_prepared={len(sessions)}/{len(requested)} {symbol}/{day} rows={len(clocks)} seconds={record['seconds']:.1f}", flush=True)
        del history, saved
        gc.collect()
    write_json(output / "sequence_manifest.json", {"identity": identity, "sessions": sessions,
        "evidence_status": "development_only_on_exposed_dates", "forecast_only": True, "semantics": QUOTE_SEQUENCE_SEMANTICS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_quote_sequence_data_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_quote_sequence_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare to materialize fixed, strictly observed quote histories.")
