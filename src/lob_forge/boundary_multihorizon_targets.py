"""Separate exact future targets for historical multi-horizon supervision."""

from __future__ import annotations

import numpy as np

from lob_forge.label_math import exact_label_array

HORIZONS_MS = (5000, 1000, 2000, 10000, 30000)
MULTIHORIZON_SEMANTICS = "exact_common_entry_100ms_five_future_horizons_v1"


def resolve_multihorizon_targets(quote_times, bid, ask, decision_times, *, min_tick):
    t, d = np.asarray(quote_times), np.asarray(decision_times)
    b, a = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    if t.ndim != 1 or d.ndim != 1 or not np.issubdtype(t.dtype, np.integer) or not np.issubdtype(d.dtype, np.integer):
        raise ValueError("Integer millisecond quote and decision vectors required")
    if b.shape != t.shape or a.shape != t.shape or not np.isfinite(b).all() or not np.isfinite(a).all() or (b <= 0).any() or (a < b).any():
        raise ValueError("Aligned finite positive uncrossed quotes required")
    if (t < 0).any() or (d < 0).any() or (t[1:] < t[:-1]).any() or (d[1:] <= d[:-1]).any():
        raise ValueError("Chronological quotes and strictly increasing nonnegative decisions required")
    if not np.isfinite(min_tick) or min_tick <= 0 or (t > np.iinfo(np.int64).max).any() or (d > np.iinfo(np.int64).max - 100 - max(HORIZONS_MS)).any():
        raise ValueError("Positive minimum tick and nonoverflowing decision timestamps required")
    t, d = t.astype(np.int64), d.astype(np.int64)
    entry = np.searchsorted(t, d + 100, side="left")
    entry_valid = entry < len(t)
    entry_time = np.zeros(len(d), dtype=np.int64)
    entry_mid = np.zeros(len(d), dtype=float)
    entry_time[entry_valid] = t[entry[entry_valid]]
    entry_mid[entry_valid] = (b[entry[entry_valid]] + a[entry[entry_valid]]) / 2
    labels = np.full((len(d), len(HORIZONS_MS)), -2, dtype=np.int8)
    available = np.zeros(labels.shape, dtype=bool)
    future_times = np.zeros(labels.shape, dtype=np.int64)
    future_mids = np.zeros(labels.shape, dtype=float)
    for head, horizon in enumerate(HORIZONS_MS):
        future = np.searchsorted(t, d + 100 + horizon, side="left")
        valid = entry_valid & (future < len(t))
        e, f = entry[valid], future[valid]
        labels[valid, head] = exact_label_array(b[e], a[e], b[f], a[f], min_tick=min_tick)
        available[:, head] = valid
        future_times[valid, head] = t[f]
        future_mids[valid, head] = (b[f] + a[f]) / 2
    return {"decision_times": d.copy(), "labels": labels, "available": available,
            "entry_times": entry_time, "entry_available": entry_valid, "entry_mids": entry_mid,
            "future_times": future_times, "future_mids": future_mids,
            "horizons_ms": np.asarray(HORIZONS_MS, dtype=np.int64)}
