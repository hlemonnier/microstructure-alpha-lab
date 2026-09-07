"""Add basis innovations and simultaneous peer states to frozen observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_basis_state import basis_state_features
from lob_forge.boundary_combined_inputs import load_combined_inputs

PEER_BASIS_FIELDS = tuple(f"basis_state_{name}_innovation_z_{h}s" for name in ("bybit", "bybit_spot") for h in (60, 300)) + (
    "basis_state_bybit_available", "basis_state_bybit_spot_available",
)


def original_basis_columns(frame):
    return [c for c in frame.columns if not c.startswith(("basis_state_", "peer_basis_state_"))]


def load_basis_inputs(root, manifest, depth_manifest, spot_manifest):
    x, y, times = load_combined_inputs(root, manifest, depth_manifest, spot_manifest)
    additional = {key: basis_state_features(frame, times[key]).set_axis(times[key], axis=0) for key, frame in x.items()}
    for key, frame in x.items():
        symbol, day = key
        peer = ("ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT", day)
        own = additional[key].reset_index(drop=True)
        other = additional[peer].loc[times[key], list(PEER_BASIS_FIELDS)].reset_index(drop=True).add_prefix("peer_")
        combined = pd.concat([frame.reset_index(drop=True), own, other], axis=1)
        pd.testing.assert_frame_equal(combined[original_basis_columns(combined)], frame.reset_index(drop=True), check_exact=True)
        if not np.isfinite(combined.to_numpy()).all() or combined.columns.duplicated().any():
            raise ValueError("Finite, unambiguous causal basis observations required")
        x[key] = combined
    return x, y, times
