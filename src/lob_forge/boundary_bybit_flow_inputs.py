"""Join causal native queue flow to the unchanged 399-field source union."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_bybit_flow import FLOW_FIELDS
from lob_forge.boundary_bybit_flow_features import bybit_flow_features
from lob_forge.boundary_combined_inputs import load_combined_inputs

FLOW_VARIANTS = {"bbo100": (100, False), "deep100": (100, True), "bbo500": (500, False), "deep500": (500, True)}
PEER_BBO = ("native_bbo_pressure_1s", "native_bbo_pressure_5s", "native_bbo_gross_1s",
            "native_message_rate_1s", "native_available_1s")
PEER_DEEP = tuple(f"native_deep_w{w}_pressure_{s}s" for w in (5, 20) for s in (1, 5)) + (
    "native_deep_unresolved_fraction_1s",)


def original_flow_columns(frame):
    return [c for c in frame.columns if not c.startswith(("flow_", "peer_flow_"))]


def select_flow_variant(frame, variant):
    if variant not in FLOW_VARIANTS:
        raise ValueError("Use a registered native-flow representation")
    own, peer = f"flow_{variant}__", f"peer_flow_{variant}__"
    columns = [c for c in frame.columns if c.startswith((own, peer))]
    if not columns:
        raise ValueError("Requested native-flow observations are absent")
    result = frame[original_flow_columns(frame) + columns].copy()
    return result.rename(columns={c: ("flow_" + c[len(own):] if c.startswith(own) else "peer_flow_" + c[len(peer):]) for c in columns})


def load_flow_inputs(root, manifest_path, depth_manifest_path, spot_manifest_path, flow_manifest_path):
    x, y, times = load_combined_inputs(root, manifest_path, depth_manifest_path, spot_manifest_path)
    base = {(r["symbol"], r["session_date"]): r for r in json.loads(manifest_path.read_text())["sessions"]}
    records = json.loads(flow_manifest_path.read_text())["sessions"]
    flows = {(r["symbol"], r["session_date"]): r for r in records}
    if len(flows) != len(records):
        raise ValueError("Native-flow sessions must be unique")
    for day in sorted({d for s, d in base}):
        additions = {}
        for symbol in ("BTCUSDT", "ETHUSDT"):
            record = flows[symbol, day]
            path = root / record["observation_path"]
            if (sha256_file(path) != record["observation_sha256"]
                or record["original_features_sha256"] != base[symbol, day]["sha256"]
                or record["fields"] != list(FLOW_FIELDS)):
                raise ValueError("Frozen native-flow identity, source and field order required")
            with np.load(path, allow_pickle=False) as archive:
                observations = {k: archive[k].copy() for k in archive.files}
            for variant, (delay, deep) in FLOW_VARIANTS.items():
                frame = bybit_flow_features(observations["decision_times"], observations, delay_ms=delay, include_deep=deep)
                additions[symbol, variant] = frame.set_axis(observations["decision_times"], axis=0)
        for symbol in ("BTCUSDT", "ETHUSDT"):
            other = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            original = x[symbol, day].reset_index(drop=True)
            pieces = [original]
            for variant, (delay, deep) in FLOW_VARIANTS.items():
                peer = [*PEER_BBO, *(PEER_DEEP if deep else ())]
                pieces.append(additions[symbol, variant].loc[times[symbol, day]].reset_index(drop=True).add_prefix(f"flow_{variant}__"))
                pieces.append(additions[other, variant].loc[times[symbol, day], peer].reset_index(drop=True).add_prefix(f"peer_flow_{variant}__"))
            combined = pd.concat(pieces, axis=1)
            pd.testing.assert_frame_equal(combined[original_flow_columns(combined)], original, check_exact=True)
            if not np.isfinite(combined.to_numpy()).all() or combined.columns.duplicated().any():
                raise ValueError("Finite unambiguous native-flow feature schemas required")
            x[symbol, day] = combined
    return x, y, times
