"""Join delayed foreign depth observations without changing the target universe."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_bybit_features import bybit_depth_features
from lob_forge.boundary_long_context_inputs import context_columns, load_context_inputs

BYBIT_VARIANTS = {"top1_100": (1, 100), "top25_100": (25, 100), "top1_500": (1, 500), "top25_500": (25, 500)}
PEER_COMMON = ("venue_mid_basis_bps", "venue_top_imbalance", "venue_return_bps_1s", "venue_return_bps_5s",
               "venue_log_spread_ratio", "venue_log_depth_ratio")
PEER_DEPTH = ("depth_5_imbalance", "depth_25_imbalance", "depth_kernel_5_imbalance", "depth_25_transition_pressure_1s")


def original_bybit_columns(frame):
    return [c for c in frame.columns if not c.startswith(("bybit_", "peer_bybit_"))]


def select_bybit_variant(frame, variant):
    if variant not in BYBIT_VARIANTS:
        raise ValueError("Use a registered Bybit representation and delay")
    own, peer = f"bybit_{variant}__", f"peer_bybit_{variant}__"
    columns = [c for c in frame.columns if c.startswith((own, peer))]
    if not columns:
        raise ValueError("Requested Bybit observations are absent")
    chosen = frame[original_bybit_columns(frame) + columns].copy()
    return chosen.rename(columns={c: ("foreign_" + c[len(own):] if c.startswith(own) else "peer_foreign_" + c[len(peer):]) for c in columns})


def load_bybit_inputs(root, manifest_path, depth_manifest_path):
    x, y, times = load_context_inputs(root, manifest_path)
    base_manifest = json.loads(manifest_path.read_text())
    source = json.loads(depth_manifest_path.read_text())
    depths = {(r["symbol"], r["session_date"]): r for r in source["sessions"]}
    base = {(r["symbol"], r["session_date"]): r for r in base_manifest["sessions"]}
    for day in sorted({d for s, d in base}):
        additions = {}
        for symbol in ("BTCUSDT", "ETHUSDT"):
            record = depths[symbol, day]
            path = root / record["observation_path"]
            if sha256_file(path) != record["observation_sha256"] or record["original_features_sha256"] != base[symbol, day]["sha256"]:
                raise ValueError("Depth observations and original Binance source must match their frozen identities")
            current = pd.read_parquet(root / base[symbol, day]["features_path"], columns=["decision_time", "bid", "ask", "bid_qty", "ask_qty"])
            with np.load(path, allow_pickle=False) as archive:
                observations = {k: archive[k].copy() for k in archive.files}
            for variant, (levels, delay) in BYBIT_VARIANTS.items():
                frame = bybit_depth_features(current, observations, delay_ms=delay, levels=levels)
                frame.index = current.decision_time.to_numpy()
                additions[symbol, variant] = frame
        for symbol in ("BTCUSDT", "ETHUSDT"):
            other = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            frame = x[symbol, day][context_columns(x[symbol, day], "observations")].reset_index(drop=True)
            original = frame.copy()
            pieces = [frame]
            for variant, (levels, delay) in BYBIT_VARIANTS.items():
                peer = [*PEER_COMMON, *(PEER_DEPTH if levels == 25 else ())]
                pieces.append(additions[symbol, variant].loc[times[symbol, day]].reset_index(drop=True).add_prefix(f"bybit_{variant}__"))
                pieces.append(additions[other, variant].loc[times[symbol, day], peer].reset_index(drop=True).add_prefix(f"peer_bybit_{variant}__"))
            frame = pd.concat(pieces, axis=1)
            pd.testing.assert_frame_equal(frame[original.columns], original, check_exact=True)
            if not np.isfinite(frame.to_numpy()).all() or frame.columns.duplicated().any():
                raise ValueError("Finite observations and an unambiguous feature schema required")
            x[symbol, day] = frame
    return x, y, times
