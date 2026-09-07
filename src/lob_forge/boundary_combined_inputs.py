"""Unite delayed spot trades and foreign depth on the exact original universe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_bybit_inputs import load_bybit_inputs, select_bybit_variant
from lob_forge.boundary_spot_inputs import load_spot_inputs, select_spot_variant

ADDITIONAL_PREFIXES = ("aux_", "peer_aux_", "foreign_", "peer_foreign_")


def original_combined_columns(frame):
    return [c for c in frame.columns if not c.startswith(ADDITIONAL_PREFIXES)]


def combine_observed_frames(depth, spot):
    """Require exact shared fields before appending the disjoint source fields."""
    original = original_combined_columns(depth)
    if original_combined_columns(spot) != original:
        raise ValueError("Both sources require the same original feature schema")
    pd.testing.assert_frame_equal(depth[original], spot[original], check_exact=True)
    extra = [c for c in spot.columns if c not in original]
    combined = pd.concat([depth, spot[extra]], axis=1)
    if combined.columns.duplicated().any() or not np.isfinite(combined.to_numpy()).all():
        raise ValueError("Source observations must be finite with disjoint schemas")
    return combined


def load_combined_inputs(root, manifest, depth_manifest, spot_manifest):
    dx, dy, dt = load_bybit_inputs(root, manifest, depth_manifest)
    sx, sy, st = load_spot_inputs(root, manifest, spot_manifest)
    if set(dx) != set(sx):
        raise ValueError("Both sources require the same sessions")
    for key in dx:
        np.testing.assert_array_equal(dy[key], sy[key])
        np.testing.assert_array_equal(dt[key], st[key])
        dx[key] = combine_observed_frames(select_bybit_variant(dx[key], "top25_100"),
                                          select_spot_variant(sx[key], "spot100"))
    return dx, dy, dt
