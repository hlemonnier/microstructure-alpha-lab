"""Matched target-venue BBO/depth features with recorded capture dependencies."""

from __future__ import annotations

import numpy as np

from lob_forge.boundary_bybit_features import bybit_depth_features

TARDIS_FEATURE_SEMANTICS = "captured_target_venue_top1_vs_top25_v1"


def tardis_depth_features(canonical, observations, *, delay_ms, levels):
    """Reuse the tested depth-curve algebra after checking both source clocks.

    The original Binance quote is the price/quantity reference. The added book
    is the same venue's delayed native source, with an explicit shared coverage
    mask in both representations. No label or future release field is read.
    """
    delays = np.asarray(observations["delays_ms"])
    if delays.ndim != 1 or not np.issubdtype(delays.dtype, np.integer):
        raise ValueError("Integer registered source delays required")
    loc = np.flatnonzero(delays == delay_ms)
    if len(loc) != 1:
        raise ValueError("Exactly one stored registered capture delay required")
    slot = int(loc[0])
    clock = canonical.decision_time.to_numpy()
    capture, publisher = (np.asarray(observations[k]) for k in ("capture_times_ns", "publisher_times_ms"))
    available = np.asarray(observations["available"])
    shape = (len(clock), len(delays))
    if available.shape != shape or available.dtype != bool or any(
        v.shape != shape or not np.issubdtype(v.dtype, np.integer) for v in (capture, publisher)
    ):
        raise ValueError("Aligned integer dependency clocks and Boolean coverage required")
    present = available[:, slot]
    cutoff = (clock - delay_ms) * 1_000_000
    if (((capture[:, slot] < 0) | (publisher[:, slot] < 0)) & present).any():
        raise ValueError("Available depth requires both source dependency clocks")
    if ((capture[:, slot] >= cutoff) & present).any() or ((publisher[:, slot] * 1_000_000 >= cutoff) & present).any():
        raise ValueError("Native depth crossed its strict capture or publisher boundary")
    # Unavailable source metadata can include a future publisher timestamp.
    # It is retained in the source artifact but contributes no model feature.
    mapped = {**observations, "publisher_times": np.where(available, publisher, -1)}
    return bybit_depth_features(canonical, mapped, delay_ms=delay_ms, levels=levels)
