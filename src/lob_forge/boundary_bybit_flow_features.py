"""Normalize integrated native queue revisions without losing gross activity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_bybit_flow import FLOW_FIELDS, FLOW_WIDTHS

FLOW_FEATURE_SEMANTICS = "causal_native_flow_top_depth_normalization_v1"
FLOW_WINDOWS_MS = (1000, 5000, 15000)


def _ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=b > 0)


def _signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def bybit_flow_features(decision_times, observations, *, delay_ms, include_deep):
    """Expose a matched BBO-flow control and additional known-price queue flow.

    All volume ratios use the last known displayed BBO depth at the delayed
    boundary. Gross revisions include additions that were subsequently removed;
    signed endpoint book differences cannot recover that information. Removed
    quantity is not identified as a cancellation or a trade.
    """
    clock = np.asarray(decision_times)
    delays, windows = np.asarray(observations["delays_ms"]), np.asarray(observations["windows_ms"])
    slots = np.flatnonzero(delays == delay_ms)
    if (clock.ndim != 1 or not len(clock) or not np.issubdtype(clock.dtype, np.integer)
        or (np.diff(clock) <= 0).any() or not np.array_equal(clock, observations["decision_times"])
        or delays.ndim != 1 or len(slots) != 1 or delay_ms < 0 or not np.array_equal(windows, FLOW_WINDOWS_MS)):
        raise ValueError("Exact increasing integer decisions and registered delayed windows required")
    slot = int(slots[0])
    totals = np.asarray(observations["flow_totals"], dtype=float)
    covered = np.asarray(observations["window_available"])
    publisher = np.asarray(observations["source_publisher_times"])
    depth = np.asarray(observations["top_depth"], dtype=float)
    shape = (len(clock), len(delays))
    if (totals.shape != (*shape, len(windows), len(FLOW_FIELDS)) or covered.shape != totals.shape[:-1]
        or covered.dtype != np.dtype(bool) or publisher.shape != shape or depth.shape != shape
        or not np.issubdtype(publisher.dtype, np.integer) or not np.isfinite(totals).all()
        or not np.isfinite(depth).all() or (depth < 0).any()):
        raise ValueError("Aligned finite native flow observations and explicit coverage required")
    present = publisher[:, slot] >= 0
    available = covered[:, slot]
    if ((publisher[:, slot][present] + delay_ms >= clock[present]).any()
        or (available.any(axis=1) & (~present | (depth[:, slot] <= 0))).any()
        or (available.any(axis=1) & (clock - delay_ms - publisher[:, slot] > 1000)).any()):
        raise ValueError("Native flow crossed its strict information or freshness boundary")
    nonnegative = [*range(12), 13, 15, 16, 17, 18]
    if (totals[..., nonnegative] < 0).any():
        raise ValueError("Gross native flow and update counts cannot be negative")
    values = {}
    for index, window in enumerate(windows):
        valid = available[:, index]
        row, scale = totals[:, slot, index], depth[:, slot]
        suffix = f"_{window // 1000}s"
        ofi, absolute_ofi, ret, absolute_ret, messages = row[:, 12:17].T
        common = {
            "available": valid.astype(float),
            "bbo_pressure": _signed_log(_ratio(ofi, scale)),
            "bbo_gross": np.log1p(_ratio(absolute_ofi, scale)),
            "bbo_efficiency": _ratio(ofi, absolute_ofi),
            "mid_return": _signed_log(ret),
            "mid_path_length": np.log1p(absolute_ret),
            "mid_efficiency": _ratio(ret, absolute_ret),
            "message_rate": np.log1p(messages * 1000 / window),
        }
        for name, value in common.items():
            values[f"native_{name}{suffix}"] = np.where(valid, value, 0)
        if include_deep:
            for i, width in enumerate(FLOW_WIDTHS):
                bi, bd, ai, ad = row[:, 4 * i:4 * i + 4].T
                gross, pressure = bi + bd + ai + ad, bi - bd - ai + ad
                # The fraction of gross revision mass cancelled by opposite
                # signed revisions on the same side, not a trade attribution.
                recycling = np.maximum(0, gross - np.abs(bi - bd) - np.abs(ai - ad))
                fields = {
                    "bid_increase": np.log1p(_ratio(bi, scale)),
                    "bid_decrease": np.log1p(_ratio(bd, scale)),
                    "ask_increase": np.log1p(_ratio(ai, scale)),
                    "ask_decrease": np.log1p(_ratio(ad, scale)),
                    "pressure": _signed_log(_ratio(pressure, scale)),
                    "imbalance": _ratio(pressure, gross),
                    "recycling": _ratio(recycling, gross),
                }
                for name, value in fields.items():
                    values[f"native_deep_w{width}_{name}{suffix}"] = np.where(valid, value, 0)
            known, unresolved = row[:, 17:19].T
            values[f"native_deep_known_update_rate{suffix}"] = np.where(valid, np.log1p(known * 1000 / window), 0)
            values[f"native_deep_unresolved_fraction{suffix}"] = np.where(valid, _ratio(unresolved, known + unresolved), 0)
    result = pd.DataFrame(values)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Normalized native flow features must remain finite")
    return result
