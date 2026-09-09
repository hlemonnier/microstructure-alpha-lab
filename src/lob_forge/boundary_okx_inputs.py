"""Join a matched counted-depth representation to unchanged original features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_combined_inputs import load_combined_inputs

OKX_VARIANTS = {"q25_100": (25, 100, False), "n25_100": (25, 100, True),
                "q100_100": (100, 100, False), "n100_100": (100, 100, True),
                "q100_500": (100, 500, False), "n100_500": (100, 500, True)}
PEER_QUANTITY = ("venue_available", "venue_log_age_ms", "venue_mid_basis_bps", "venue_top_quantity_imbalance",
    "venue_return_bps_1s", "venue_return_bps_5s", "venue_history_available_1s", "venue_history_available_5s",
    "depth_25_quantity_imbalance", "depth_25_sampled_quantity_pressure_1s", "depth_25_sampled_quantity_pressure_5s")
PEER_COUNTS = ("count_1_log_orders", "count_1_imbalance", "count_25_imbalance", "count_25_mean_order_imbalance",
               "count_25_sampled_count_pressure_1s", "count_25_sampled_count_pressure_5s")


def counted_columns(columns, include_counts):
    columns = list(columns)
    if (not columns or len(set(columns)) != len(columns)
        or any(not c.startswith(("venue_", "depth_", "count_")) for c in columns)):
        raise ValueError("Only unique explicitly named observation fields are allowed")
    return [c for c in columns if include_counts or not c.startswith("count_")]


def append_counted_features(original, decision_times, own, peer, *, include_counts):
    times = np.asarray(decision_times)
    if len(original) != len(times) or not np.issubdtype(times.dtype, np.integer) or (np.diff(times) <= 0).any():
        raise ValueError("Unchanged original row count and increasing integer clocks required")
    for frame in (own, peer):
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Exactly one sorted own/peer observation per clock required")
    own_fields = counted_columns(own.columns, include_counts)
    peer_fields = [*PEER_QUANTITY, *(PEER_COUNTS if include_counts else ())]
    counted_columns(peer.columns, include_counts)
    original = original.reset_index(drop=True)
    joined = pd.concat([original, own.loc[times, own_fields].reset_index(drop=True).add_prefix("okx_"),
        peer.loc[times, peer_fields].reset_index(drop=True).add_prefix("peer_okx_")], axis=1)
    if joined.columns.duplicated().any() or not np.isfinite(joined.to_numpy()).all():
        raise ValueError("Finite, unambiguous counted-depth joins required")
    pd.testing.assert_frame_equal(joined[original.columns], original, check_exact=True)
    return joined


def load_okx_inputs(root, manifest_path, depth_manifest, spot_manifest, feature_manifest, variant):
    if variant not in OKX_VARIANTS:
        raise ValueError("Use a registered matched quantity/count representation")
    levels, delay, counts = OKX_VARIANTS[variant]
    x, y, times = load_combined_inputs(root, manifest_path, depth_manifest, spot_manifest)
    base = {(r["symbol"], r["session_date"]): r for r in json.loads(manifest_path.read_text())["sessions"]}
    records = json.loads(feature_manifest.read_text())["sessions"]
    sources = {(r["symbol"], r["date"]): r for r in records}
    if len(sources) != len(records):
        raise ValueError("Exactly one counted-feature source per asset/date required")
    for day in sorted({d for s, d in base}):
        additions = {}
        for symbol in ("BTCUSDT", "ETHUSDT"):
            record = sources[symbol, day]
            if record["original_features_sha256"] != base[symbol, day]["sha256"]:
                raise ValueError("Counted features and original decision sources have different identities")
            group = record["groups"][f"top{levels}_{delay}"]
            path = root / group["features_path"]
            if sha256_file(path) != group["sha256"]:
                raise ValueError("A frozen counted-feature cache changed")
            columns = counted_columns(group["columns"], counts)
            frame = pd.read_parquet(path, columns=["decision_time", *columns]).set_index("decision_time")
            if len(frame) != record["decision_rows"]:
                raise ValueError("Cached counted-feature row inventory changed")
            additions[symbol] = frame
        for symbol in ("BTCUSDT", "ETHUSDT"):
            peer = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            x[symbol, day] = append_counted_features(x[symbol, day], times[symbol, day], additions[symbol], additions[peer], include_counts=counts)
    return x, y, times
