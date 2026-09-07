"""Attach long context to the existing event-clock observation universe."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.boundary_event_clock_inputs import load_clock_inputs
from lob_forge.boundary_long_context import (
    PEER_OBSERVED,
    PEER_RELEASED,
    observation_context,
    released_outcome_context,
)


def load_context_inputs(root, manifest_path):
    features, labels, times = load_clock_inputs(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sessions = {(s["symbol"], s["session_date"]): s for s in manifest["sessions"]}
    for day in sorted({d for s, d in sessions}):
        raw = {symbol: pd.read_parquet(root / sessions[symbol, day]["features_path"]).set_index("decision_time", drop=False)
               for symbol in ("BTCUSDT", "ETHUSDT")}
        common = np.intersect1d(raw["BTCUSDT"].index, raw["ETHUSDT"].index)
        breaks = np.r_[0, np.flatnonzero(np.diff(common) != 1000) + 1, len(common)]
        pieces = {symbol: [] for symbol in raw}
        for left, right in zip(breaks[:-1], breaks[1:]):
            if right - left <= 120:
                continue
            group = common[left:right]
            observations, released = {}, {}
            for symbol in raw:
                frame = raw[symbol].loc[group].reset_index(drop=True)
                observations[symbol] = observation_context(frame)
                valid = np.isfinite(frame.label.to_numpy()) & np.isfinite(frame.future_event_time.to_numpy())
                released[symbol] = released_outcome_context(
                    frame.label.to_numpy()[valid], group[valid], frame.future_event_time.to_numpy()[valid], group,
                )
            for symbol in raw:
                peer = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
                addition = pd.concat([
                    observations[symbol], observations[peer][list(PEER_OBSERVED)].add_prefix("peer_"),
                    released[symbol], released[peer][list(PEER_RELEASED)].add_prefix("peer_"),
                ], axis=1)
                addition.index = group
                pieces[symbol].append(addition)
        for symbol in raw:
            addition = pd.concat(pieces[symbol]).loc[times[symbol, day]].reset_index(drop=True)
            base = features[symbol, day]
            augmented = pd.concat([base.reset_index(drop=True), addition], axis=1)
            if not np.isfinite(augmented.to_numpy()).all() or augmented.columns.duplicated().any():
                raise ValueError("Context features must be finite with an unambiguous schema")
            np.testing.assert_array_equal(augmented[base.columns].to_numpy(), base.to_numpy())
            features[symbol, day] = augmented
    return features, labels, times


def context_columns(frame, representation):
    if representation == "observations":
        return [c for c in frame.columns if not c.startswith(("context_label_", "peer_context_label_"))]
    if representation == "released":
        return list(frame.columns)
    raise ValueError("Use the registered observations or released context representation")
