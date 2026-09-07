"""Join independent trade-stream sidecars onto the frozen long-context inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_long_context_inputs import context_columns, load_context_inputs
from lob_forge.boundary_spot_features import PEER_SPOT_FEATURES, SPOT_FEATURES

VARIANTS = ("perp100", "spot100", "spot500")


def load_spot_inputs(root, original_manifest_path, spot_manifest_path):
    x, y, times = load_context_inputs(root, original_manifest_path)
    original = json.loads(original_manifest_path.read_text())
    manifest = json.loads(spot_manifest_path.read_text())
    records = {(s["symbol"], s["session_date"]): s for s in manifest["sessions"]}
    raw = {}
    for session in original["sessions"]:
        key = session["symbol"], session["session_date"]
        record = records[key]
        if record["original_features_sha256"] != session["sha256"] or record["original_features_path"] != session["features_path"]:
            raise ValueError("Auxiliary trade features must match the frozen original observations")
        if sha256_file(root / record["features_path"]) != record["sha256"]:
            raise ValueError("Auxiliary trade feature hash changed")
        raw[key] = pd.read_parquet(root / record["features_path"]).set_index("decision_time")
    for key, frame in x.items():
        symbol, day = key
        peer = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
        base = frame[context_columns(frame, "observations")].copy()
        pieces = [base.reset_index(drop=True)]
        for variant in VARIANTS:
            own_names = [variant + "_" + name for name in SPOT_FEATURES]
            peer_names = [variant + "_" + name for name in PEER_SPOT_FEATURES]
            own = raw[key].loc[times[key], own_names].reset_index(drop=True)
            other = raw[peer, day].loc[times[key], peer_names].reset_index(drop=True)
            own.columns = [variant + "_aux_" + n.removeprefix("spot_") for n in SPOT_FEATURES]
            other.columns = [variant + "_peer_aux_" + n.removeprefix("spot_") for n in PEER_SPOT_FEATURES]
            pieces.extend([own, other])
        combined = pd.concat(pieces, axis=1)
        if not np.isfinite(combined.to_numpy()).all() or combined.columns.duplicated().any():
            raise ValueError("Combined auxiliary observations must be finite and unambiguous")
        np.testing.assert_array_equal(combined[base.columns].to_numpy(), base.to_numpy())
        x[key] = combined
    return x, y, times


def original_columns(frame):
    return [c for c in frame.columns if not c.startswith(tuple(v + "_" for v in VARIANTS))]


def select_spot_variant(frame, variant):
    if variant not in VARIANTS:
        raise ValueError("Use a registered auxiliary trade source/delay")
    prefix = variant + "_"
    columns = original_columns(frame) + [c for c in frame.columns if c.startswith(prefix)]
    return frame[columns].rename(columns=lambda c: c.removeprefix(prefix))
