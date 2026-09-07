"""Build separate exact move-size targets without modifying forecast features."""

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
from lob_forge.boundary_distribution_targets import COARSE_CLASSES, DISTRIBUTION_SEMANTICS, exact_move_bins
from lob_forge.boundary_events import read_archive
from prepare_boundary_event_clock import write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered distribution-target input changed: {path}")
    source = json.loads((ROOT / protocol["source_manifest"]).read_text())
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
                "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow")}}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing target preparation after identity changes")
    write_json(frozen, identity)
    requested = [s for s in source["sessions"] if s["session_date"] in protocol["dates"] and s["symbol"] in protocol["symbols"]]
    if len(requested) != len(protocol["dates"]) * len(protocol["symbols"]):
        raise ValueError("All registered exposed sessions are required")
    sessions = []
    for original in requested:
        started = time.monotonic()
        symbol, day = original["symbol"], original["session_date"]
        metadata = output / symbol / f"{day}.json"
        if metadata.exists():
            record = json.loads(metadata.read_text())
            if record["identity"] != identity or sha256_file(ROOT / record["targets_path"]) != record["sha256"]:
                raise ValueError("Cached distribution targets changed")
            sessions.append(record)
            continue
        source_path = ROOT / original["features_path"]
        if sha256_file(source_path) != original["sha256"]:
            raise ValueError("The immutable forecast feature source changed")
        raw = original["sources"]["bookTicker"]
        raw_path = Path(raw["local_path"])
        if raw_path.stat().st_size != raw["bytes"] or sha256_file(raw_path) != raw["sha256"] or raw["sha256"] != raw["provider_checksum_sha256"]:
            raise ValueError("Raw quotes must match the official checksum and byte count")
        frame = pd.read_parquet(source_path, columns=["decision_time", "label", "entry_event_time", "future_event_time", "entry_mid", "future_mid"])
        quotes = read_archive(raw_path, ["best_bid_price", "best_ask_price", "event_time"])
        times = quotes.event_time.to_numpy(dtype=np.int64)
        if (np.diff(times) < 0).any():
            raise ValueError("Raw quotes must be chronological")
        entry = np.searchsorted(times, frame.decision_time.to_numpy() + 100)
        future = np.searchsorted(times, frame.decision_time.to_numpy() + 5100)
        resolved = future < len(times)
        np.testing.assert_array_equal(resolved, frame.label.notna())
        bid, ask = quotes.best_bid_price.to_numpy(), quotes.best_ask_price.to_numpy()
        eb, ea, fb, fa = bid[entry[resolved]], ask[entry[resolved]], bid[future[resolved]], ask[future[resolved]]
        for name, value in (("entry_event_time", times[entry[resolved]]), ("future_event_time", times[future[resolved]]),
                            ("entry_mid", (eb + ea) / 2), ("future_mid", (fb + fa) / 2)):
            np.testing.assert_array_equal(frame.loc[resolved, name].to_numpy(), value)
        bins = exact_move_bins(eb, ea, fb, fa, min_tick=protocol["min_tick"][symbol])
        np.testing.assert_array_equal(COARSE_CLASSES[bins], frame.loc[resolved, "label"].to_numpy())
        target = pd.DataFrame({"decision_time": frame.loc[resolved, "decision_time"].to_numpy(), "move_bin": bins,
                               "label": COARSE_CLASSES[bins], "future_event_time": times[future[resolved]]})
        path = output / symbol / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        target.to_parquet(temporary, index=False)
        temporary.replace(path)
        pd.testing.assert_frame_equal(target, pd.read_parquet(path), check_exact=True)
        record = {"symbol": symbol, "session_date": day, "identity": identity,
                  "source_features_path": original["features_path"], "source_features_sha256": original["sha256"],
                  "raw_quotes": raw, "targets_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                  "rows": len(target), "coarse_label_parity": True, "raw_resolution_parity": True,
                  "save_reload_parity": True, "semantics": DISTRIBUTION_SEMANTICS, "seconds": time.monotonic() - started}
        write_json(metadata, record)
        sessions.append(record)
        print(f"distribution_targets_prepared={len(sessions)}/{len(requested)} {symbol}/{day} rows={len(target)}", flush=True)
        del frame, quotes, target, times, entry, future, bid, ask, eb, ea, fb, fa, bins
        gc.collect()
    write_json(output / "target_manifest.json", {"identity": identity, "sessions": sessions, "evidence_status": "development_only_on_exposed_dates",
               "semantics": DISTRIBUTION_SEMANTICS, "targets_only_never_forecast_inputs": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_distribution_targets_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_distribution_targets_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare to create exact magnitude targets separately from forecast features.")
