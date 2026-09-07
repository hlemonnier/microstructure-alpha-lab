"""Prepare delayed spot features and an identically transformed perpetual control."""

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
from lob_forge.boundary_spot_data import read_spot_trades
from lob_forge.boundary_spot_features import SPOT_FEATURES, SPOT_SEMANTICS, spot_feature_frame
from lob_forge.features import AGG_TRADE_COLUMNS
from prepare_boundary_event_clock import write_json

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"perp100": ("perpetual", 100), "spot100": ("spot", 100), "spot500": ("spot", 500)}


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered spot feature input changed: {path}")
    original_manifest = json.loads((ROOT / protocol["original_manifest"]).read_text())
    spot_manifest = json.loads((ROOT / protocol["spot_manifest"]).read_text())
    originals = {(s["symbol"], s["session_date"]): s for s in original_manifest["sessions"]}
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
                "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow")}}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing spot features after an identity change")
    write_json(frozen, identity)
    requested = [s for s in spot_manifest["sessions"] if s["symbol"] in protocol["symbols"] and s["session_date"] in protocol["dates"]]
    if len(requested) != 28:
        raise ValueError("All 28 registered spot sessions are required")
    records = []
    for spot in requested:
        started = time.monotonic()
        symbol, day = spot["symbol"], spot["session_date"]
        original = originals[symbol, day]
        metadata = output / symbol / f"{day}.json"
        if metadata.exists():
            record = json.loads(metadata.read_text())
            if record["identity"] != identity or sha256_file(ROOT / record["features_path"]) != record["sha256"]:
                raise ValueError("Cached auxiliary-trade features changed")
            records.append(record)
            continue
        original_path, spot_path = ROOT / original["features_path"], ROOT / spot["local_path"]
        if sha256_file(original_path) != original["sha256"] or sha256_file(spot_path) != spot["sha256"] or spot["sha256"] != spot["provider_checksum_sha256"]:
            raise ValueError("Original observations and official spot archive must match their hashes")
        perpetual = original["sources"]["aggTrades"]
        if sha256_file(perpetual["local_path"]) != perpetual["sha256"] or perpetual["sha256"] != perpetual["provider_checksum_sha256"]:
            raise ValueError("The perpetual-trade control must match its official hash")
        known = pd.read_parquet(original_path, columns=["decision_time", "bid", "ask", "bid_qty", "ask_qty"])
        trades = {"spot": read_spot_trades(spot_path, day_start_ms=utc_ms(day)),
                  "perpetual": read_archive(Path(perpetual["local_path"]), AGG_TRADE_COLUMNS)}
        pieces = [known[["decision_time"]].reset_index(drop=True)]
        for name, (stream, delay) in VARIANTS.items():
            features = spot_feature_frame(known, trades[stream], information_delay_ms=delay)
            np.testing.assert_array_equal(features.decision_time, known.decision_time)
            pieces.append(features[list(SPOT_FEATURES)].add_prefix(name + "_"))
        result = pd.concat(pieces, axis=1)
        path = output / symbol / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        result.to_parquet(temporary, index=False)
        temporary.replace(path)
        pd.testing.assert_frame_equal(result, pd.read_parquet(path), check_exact=True)
        if sha256_file(original_path) != original["sha256"]:
            raise ValueError("Original observations must remain untouched")
        record = {"symbol": symbol, "session_date": day, "identity": identity, "features_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                  "original_features_path": original["features_path"], "original_features_sha256": original["sha256"],
                  "spot_archive": spot, "perpetual_archive": perpetual, "rows": len(result), "variants": VARIANTS,
                  "features_per_variant": len(SPOT_FEATURES), "semantics": SPOT_SEMANTICS, "original_file_unchanged": True,
                  "all_original_decision_times_preserved": True, "save_reload_parity": True, "seconds": time.monotonic() - started}
        write_json(metadata, record)
        records.append(record)
        print(f"spot_features_prepared={len(records)}/28 {symbol}/{day} rows={len(result)} seconds={record['seconds']:.1f}", flush=True)
        del known, trades, pieces, features, result
        gc.collect()
    write_json(output / "feature_manifest.json", {"identity": identity, "sessions": records, "semantics": SPOT_SEMANTICS,
               "evidence_status": "development_only_on_exposed_dates", "observations_only": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_spot_features_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_spot_features_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare for the fixed delayed auxiliary-trade feature family.")
