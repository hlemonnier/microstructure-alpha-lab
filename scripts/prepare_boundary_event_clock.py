"""Append event-clock observations to immutable, exposed exact-label sessions."""

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
from lob_forge.boundary_event_clock import EVENT_CLOCK_FEATURES, EVENT_CLOCK_SEMANTICS, event_clock_frame
from lob_forge.boundary_events import read_archive
from lob_forge.features import AGG_TRADE_COLUMNS, BOOK_TICKER_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered preparation input changed: {name}")
    source = json.loads((ROOT / protocol["source_manifest"]).read_text())
    identity = {
        "protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow")},
    }
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing event-clock observations after an identity change")
    write_json(frozen, identity)
    sessions = []
    requested = [s for s in source["sessions"] if s["session_date"] in protocol["dates"] and s["symbol"] in protocol["symbols"]]
    if len(requested) != len(protocol["dates"]) * len(protocol["symbols"]):
        raise ValueError("Every registered exposed session is required")
    for original in requested:
        started = time.monotonic()
        symbol, day = original["symbol"], original["session_date"]
        metadata = output / symbol / f"{day}.json"
        if metadata.exists():
            cached = json.loads(metadata.read_text())
            if cached["event_clock_identity"] != identity or sha256_file(ROOT / cached["features_path"]) != cached["sha256"]:
                raise ValueError("Cached event-clock observations changed")
            sessions.append(cached)
            continue
        source_path = ROOT / original["features_path"]
        if sha256_file(source_path) != original["sha256"]:
            raise ValueError("Exact-label parity source changed")
        for raw in original["sources"].values():
            path = Path(raw["local_path"])
            if path.stat().st_size != raw["bytes"] or sha256_file(path) != raw["sha256"] or raw["sha256"] != raw["provider_checksum_sha256"]:
                raise ValueError("Raw source must match its official SHA256 and byte count")
        canonical = pd.read_parquet(source_path)
        quotes = read_archive(Path(original["sources"]["bookTicker"]["local_path"]), BOOK_TICKER_COLUMNS)
        trades = read_archive(Path(original["sources"]["aggTrades"]["local_path"]), AGG_TRADE_COLUMNS)
        additions = event_clock_frame(quotes, trades, canonical.decision_time.to_numpy())
        np.testing.assert_array_equal(additions.decision_time, canonical.decision_time)
        quote_rows = np.searchsorted(quotes.event_time.to_numpy(), canonical.decision_time.to_numpy(), side="left") - 1
        for raw_name, saved_name in (("best_bid_price", "bid"), ("best_ask_price", "ask"), ("best_bid_qty", "bid_qty"), ("best_ask_qty", "ask_qty")):
            np.testing.assert_array_equal(quotes[raw_name].to_numpy()[quote_rows], canonical[saved_name].to_numpy())
        augmented = pd.concat([canonical.reset_index(drop=True), additions[list(EVENT_CLOCK_FEATURES)]], axis=1)
        augmented["event_clock_semantics_version"] = EVENT_CLOCK_SEMANTICS
        pd.testing.assert_frame_equal(augmented[canonical.columns], canonical.reset_index(drop=True), check_exact=True)
        path = output / symbol / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        augmented.to_parquet(temporary, index=False)
        temporary.replace(path)
        restored = pd.read_parquet(path)
        pd.testing.assert_frame_equal(restored, augmented, check_exact=True)
        record = {
            **original, "event_clock_identity": identity, "features_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "original_features_path": original["features_path"], "original_features_sha256": original["sha256"],
            "event_clock_semantics_version": EVENT_CLOCK_SEMANTICS, "event_clock_features": list(EVENT_CLOCK_FEATURES),
            "original_columns_bitwise_unchanged": True, "current_raw_quote_parity": True, "save_reload_parity": True,
            "event_clock_seconds": time.monotonic() - started,
        }
        write_json(metadata, record)
        sessions.append(record)
        write_json(output / "progress.json", {"completed_sessions": len(sessions), "planned_sessions": len(requested), "latest_session": [symbol, day]})
        print(f"event_clock_prepared={len(sessions)}/{len(requested)} {symbol}/{day} rows={len(augmented)} seconds={record['event_clock_seconds']:.1f}", flush=True)
        del canonical, quotes, trades, additions, augmented, restored
        gc.collect()
    write_json(output / "dataset_manifest.json", {
        "identity": identity, "sessions": sessions, "evidence_status": "development_only_on_exposed_dates",
        "forecast_only": True, "event_clock_semantics_version": EVENT_CLOCK_SEMANTICS,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_event_clock_data_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_event_clock_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol, args.output)
    else:
        print("Pass --prepare to append registered event-clock observations to exposed exact-label sessions.")
