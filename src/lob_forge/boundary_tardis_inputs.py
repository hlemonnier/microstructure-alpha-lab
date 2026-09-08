"""Join captured target-venue depth onto the exact existing source union."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_bybit_inputs import BYBIT_VARIANTS, PEER_COMMON, PEER_DEPTH
from lob_forge.boundary_combined_inputs import load_combined_inputs, original_combined_columns
from lob_forge.boundary_tardis_features import tardis_depth_features

TARDIS_VARIANTS = dict(BYBIT_VARIANTS)
TARDIS_PREFIXES = ("target_book_", "peer_target_book_")


def original_tardis_columns(frame):
    return [c for c in frame.columns if not c.startswith(TARDIS_PREFIXES)]


def select_tardis_variant(frame, variant):
    base = original_tardis_columns(frame)
    if variant == "observations":
        return frame[original_combined_columns(frame[base])].copy()
    if variant == "combined":
        return frame[base].copy()
    if variant not in TARDIS_VARIANTS:
        raise ValueError("Use a registered target-depth representation")
    own, peer = f"target_book_{variant}__", f"peer_target_book_{variant}__"
    added = [c for c in frame.columns if c.startswith((own, peer))]
    if not added:
        raise ValueError("Requested native depth is absent")
    chosen = frame[base + added].copy()
    return chosen.rename(columns={c: ("native_depth_" + c[len(own):] if c.startswith(own)
                                      else "peer_native_depth_" + c[len(peer):]) for c in added})


def load_tardis_inputs(root, manifest_path, native_manifest_path, bybit_manifest_path, spot_manifest_path):
    x, y, times = load_combined_inputs(root, manifest_path, bybit_manifest_path, spot_manifest_path)
    original = json.loads(manifest_path.read_text())
    native = json.loads(native_manifest_path.read_text())
    base = {(r["symbol"], r["session_date"]): r for r in original["sessions"]}
    sources = {(r["symbol"], r["session_date"]): r for r in native["sessions"]}
    for day in sorted({d for s, d in base}):
        additions, source_clocks = {}, {}
        for symbol in ("BTCUSDT", "ETHUSDT"):
            record = sources[symbol, day]
            path = root / record["observation_path"]
            if sha256_file(path) != record["observation_sha256"] or record["original_features_sha256"] != base[symbol, day]["sha256"]:
                raise ValueError("Native depth and original source identities must match")
            raw = pd.read_parquet(root / base[symbol, day]["features_path"],
                columns=["decision_time", "bid", "ask", "bid_qty", "ask_qty"]).set_index("decision_time", drop=False)
            with np.load(path, allow_pickle=False) as archive:
                observed = {k: archive[k].copy() for k in archive.files}
            clock = observed["decision_times"]
            current = raw.loc[clock].reset_index(drop=True)
            source_clocks[symbol] = clock
            for variant, (levels, delay) in TARDIS_VARIANTS.items():
                values = tardis_depth_features(current, observed, delay_ms=delay, levels=levels)
                values.index = clock
                additions[symbol, variant] = values
        for symbol in ("BTCUSDT", "ETHUSDT"):
            key = symbol, day
            other = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            # Restrict only to the frozen source interval, never to availability.
            clock = source_clocks[symbol]
            keep = (times[key] >= clock[0]) & (times[key] <= clock[-1])
            retained = times[key][keep]
            original_frame = x[key].loc[keep].reset_index(drop=True)
            if any(c.startswith(TARDIS_PREFIXES) for c in original_frame.columns):
                raise ValueError("Native namespace collides with an original feature")
            pieces = [original_frame]
            for variant, (levels, delay) in TARDIS_VARIANTS.items():
                peer = ["venue_available", *PEER_COMMON, *(PEER_DEPTH if levels == 25 else ())]
                pieces.append(additions[symbol, variant].loc[retained].reset_index(drop=True).add_prefix(f"target_book_{variant}__"))
                pieces.append(additions[other, variant].loc[retained, peer].reset_index(drop=True).add_prefix(f"peer_target_book_{variant}__"))
            frame = pd.concat(pieces, axis=1)
            pd.testing.assert_frame_equal(frame[original_frame.columns], original_frame, check_exact=True)
            if frame.columns.duplicated().any() or not np.isfinite(frame.to_numpy()).all():
                raise ValueError("Finite observations and distinct feature namespaces required")
            x[key], y[key], times[key] = frame, y[key][keep], retained
    return x, y, times
