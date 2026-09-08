"""Resolve historical auxiliary targets separately from observation artifacts."""

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
from lob_forge.boundary_events import read_archive
from lob_forge.boundary_multihorizon_targets import HORIZONS_MS, MULTIHORIZON_SEMANTICS, resolve_multihorizon_targets
from prepare_boundary_event_clock import write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered multi-horizon target input changed: {path}")
    source = json.loads((ROOT / protocol["source_manifest"]).read_text())
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow")}}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing target preparation after changes")
    write_json(frozen, identity)
    requested = [s for s in source["sessions"] if s["session_date"] in protocol["dates"] and s["symbol"] in protocol["symbols"]]
    if len(requested) != len(protocol["dates"]) * len(protocol["symbols"]):
        raise ValueError("All historical target sessions are required")
    sessions = []
    for original in requested:
        started = time.monotonic()
        symbol, day = original["symbol"], original["session_date"]
        metadata = output / symbol / f"{day}.json"
        if metadata.exists():
            record = json.loads(metadata.read_text())
            if record["identity"] != identity or sha256_file(ROOT / record["targets_path"]) != record["sha256"]:
                raise ValueError("Cached future targets changed")
            sessions.append(record)
            continue
        source_path = ROOT / original["features_path"]
        if sha256_file(source_path) != original["sha256"]:
            raise ValueError("Immutable observation features changed")
        raw = original["sources"]["bookTicker"]
        raw_path = Path(raw["local_path"])
        if raw_path.stat().st_size != raw["bytes"] or sha256_file(raw_path) != raw["sha256"] or raw["sha256"] != raw["provider_checksum_sha256"]:
            raise ValueError("Raw quotes must match official checksum and bytes")
        frame = pd.read_parquet(source_path, columns=["decision_time", "label", "entry_event_time", "future_event_time", "entry_mid", "future_mid"])
        quotes = read_archive(raw_path, ["best_bid_price", "best_ask_price", "event_time"])
        values = resolve_multihorizon_targets(quotes.event_time.to_numpy(), quotes.best_bid_price.to_numpy(),
            quotes.best_ask_price.to_numpy(), frame.decision_time.to_numpy(), min_tick=protocol["min_tick"][symbol])
        resolved = values["available"][:, 0]
        np.testing.assert_array_equal(resolved, frame.label.notna())
        np.testing.assert_array_equal(values["labels"][resolved, 0], frame.loc[resolved, "label"])
        for name, expected in (("entry_event_time", values["entry_times"]), ("future_event_time", values["future_times"][:, 0]),
                               ("entry_mid", values["entry_mids"]), ("future_mid", values["future_mids"][:, 0])):
            np.testing.assert_array_equal(frame.loc[resolved, name].to_numpy(), expected[resolved])
        path = output / symbol / f"{day}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(path)
        with np.load(path, allow_pickle=False) as saved:
            for field in values:
                np.testing.assert_array_equal(saved[field], values[field])
        record = {"symbol": symbol, "session_date": day, "identity": identity,
            "source_features_path": original["features_path"], "source_features_sha256": original["sha256"],
            "raw_quotes": raw, "targets_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": len(frame),
            "available_by_horizon": {str(h): int(values["available"][:, i].sum()) for i, h in enumerate(HORIZONS_MS)},
            "primary_label_and_raw_resolution_exact": True, "save_reload_exact": True,
            "semantics": MULTIHORIZON_SEMANTICS, "seconds": time.monotonic() - started}
        write_json(metadata, record)
        sessions.append(record)
        print(f"multihorizon_targets_prepared={len(sessions)}/{len(requested)} {symbol}/{day} rows={len(frame)}", flush=True)
        del frame, quotes, values
        gc.collect()
    write_json(output / "target_manifest.json", {"identity": identity, "sessions": sessions, "evidence_status": protocol["evidence_status"],
        "semantics": MULTIHORIZON_SEMANTICS, "targets_only_never_forecast_inputs": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_multihorizon_preparation_20260908.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_multihorizon_targets_20260908")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare to resolve the separate historical training targets.")
