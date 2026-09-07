"""Append a fixed event-clock schema without changing the frozen row universe."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lob_forge.boundary_event_clock import EVENT_CLOCK_FEATURES, PEER_CLOCK_FEATURES
from lob_forge.boundary_event_inputs import load_event_inputs


def load_clock_inputs(root, manifest_path):
    # The frozen loader checks every augmented-file hash and applies exactly the
    # same simultaneous-session, warm-up and end-of-coverage rules as round one.
    features, labels, times = load_event_inputs(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    raw = {}
    for session in manifest["sessions"]:
        raw[session["symbol"], session["session_date"]] = pd.read_parquet(
            root / session["features_path"], columns=["decision_time", *EVENT_CLOCK_FEATURES],
        ).set_index("decision_time")
    for (symbol, day), frame in features.items():
        peer = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
        clock = times[symbol, day]
        own = raw[symbol, day].loc[clock].reset_index(drop=True)
        # Every retained second belongs to the same observed session for both
        # assets. This join cannot choose a future peer observation.
        other = raw[peer, day].loc[clock, list(PEER_CLOCK_FEATURES)].reset_index(drop=True).add_prefix("peer_")
        extended = pd.concat([frame.reset_index(drop=True), own, other], axis=1)
        if not np.isfinite(extended.to_numpy()).all() or extended.columns.duplicated().any():
            raise ValueError("The extended schema must be finite and unique")
        np.testing.assert_array_equal(extended[frame.columns].to_numpy(), frame.to_numpy())
        features[symbol, day] = extended
    return features, labels, times
