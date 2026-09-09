"""Retain absolute published quantity scale in the matched information control."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantity_scale_features(observations, *, levels, delay_ms):
    """Log source quantities without treating them as cross-venue currency units.

    The count-specific mean uses the same quantity total as its control. Model
    preprocessing remains train-only and the asset indicator remains explicit.
    These scale fields complement the separate unit-invariant shape features.
    """
    if levels not in (25, 100):
        raise ValueError("Only the registered depth scales are supported")
    slots = np.flatnonzero(np.asarray(observations["delays_ms"]) == delay_ms)
    if len(slots) != 1:
        raise ValueError("Exactly one registered delayed source state required")
    slot = int(slots[0])
    clock = np.asarray(observations["decision_times"])
    cutoff = np.asarray(observations["query_cutoffs"][:, slot])
    publisher = np.asarray(observations["publisher_times"][:, slot])
    known = np.asarray(observations["known_depths"][:, slot])
    quantity = np.asarray(observations["depth"][:, slot, :levels])[:, :, (1, 3)]
    counts = np.asarray(observations["order_counts"][:, slot, :levels])
    n = len(clock)
    if (clock.ndim != 1 or not np.issubdtype(clock.dtype, np.integer) or (np.diff(clock) <= 0).any()
        or quantity.shape != (n, levels, 2) or counts.shape != quantity.shape or known.shape != (n, 2)
        or not np.issubdtype(counts.dtype, np.integer) or (counts >= 2**53).any()
        or not np.isfinite(quantity).all() or (quantity < 0).any() or (counts < 0).any()):
        raise ValueError("Aligned finite source quantities and bounded integer counts required")
    present = publisher >= 0
    available = present & (known.min(axis=1) >= levels)
    if (not np.array_equal(cutoff, ((clock - delay_ms) // 100) * 100)
        or not (publisher[present] < cutoff[present]).all()
        or ((cutoff[present] - publisher[present]) > 1000).any()
        or (quantity[available] <= 0).any() or (counts[available] <= 0).any()):
        raise ValueError("Available quantity scale violates depth or timing constraints")
    values = {}
    for k in (1, 5, 10, 25, 50, 100):
        if k > levels:
            continue
        total_q = quantity[available, :k].sum(axis=(1, 2))
        total_n = counts[available, :k].sum(axis=(1, 2), dtype=np.int64)
        if not np.isfinite(total_q).all() or (total_q <= 0).any() or (total_n <= 0).any():
            raise ValueError("Quantity and count totals must remain positive and finite")
        quantity_log, mean_log = np.zeros(n), np.zeros(n)
        quantity_log[available] = np.log(total_q)
        mean_log[available] = np.log(total_q) - np.log(total_n.astype(float))
        values[f"depth_{k}_log_source_quantity"] = quantity_log
        values[f"count_{k}_log_mean_source_quantity"] = mean_log
    return pd.DataFrame(values)
