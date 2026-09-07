"""Integrate observed queue revisions between sampled foreign book states."""

from __future__ import annotations

import math

import numpy as np

from lob_forge.boundary_bybit_depth import BybitDepthState

FLOW_WIDTHS = (1, 5, 20)
FLOW_FIELDS = tuple(f"{side}_{kind}_width{width}" for width in FLOW_WIDTHS
                    for side in ("bid", "ask") for kind in ("increase", "decrease")) + (
    "bbo_ofi", "absolute_bbo_ofi", "mid_return_bps", "absolute_mid_return_bps",
    "delta_messages", "known_level_updates", "unresolved_level_updates",
)
FLOW_SEMANTICS = "native_bybit_known_queue_revisions_v1"


def native_bybit_flow_events(messages, symbol):
    """Validate atomic messages, then measure changes inside prior visible depth.

    Decreases cannot distinguish trades from cancellations. New prices beyond
    the previously visible deepest level have unknown prior quantity: they are
    excluded from volume flow and counted separately. Snapshots are not orders.
    """
    book = BybitDepthState(symbol)
    clocks, values, resets, depths, complete = [], [], [], [], []
    previous = None
    previous_clock = None
    last_reset = None
    snapshot_count = 0
    for payload in messages:
        old_bids, old_asks = book.bids, book.asks
        book.apply(payload)
        now = book.timestamp
        present = len(book.bids) >= 25 and len(book.asks) >= 25
        current = None
        if present:
            bid, ask = max(book.bids), min(book.asks)
            current = (bid, ask, book.bids[bid], book.asks[ask], min(book.bids), max(book.asks))
        snapshot = payload["type"] == "snapshot"
        gap = previous_clock is not None and now - previous_clock > 1000
        if snapshot or gap or previous is None or not present:
            last_reset = now
        row = np.zeros(len(FLOW_FIELDS), dtype=float)
        if not snapshot and not gap and previous is not None and current is not None:
            bid, ask, bq, aq, lowest_bid, highest_ask = previous
            mid, spread = (bid + ask) / 2, ask - bid
            for side, old, edge, sign, offset in (("b", old_bids, lowest_bid, -1, 0), ("a", old_asks, highest_ask, 1, 2)):
                for raw_price, raw_quantity in payload["data"][side]:
                    price, quantity = float(raw_price), float(raw_quantity)
                    known = price >= edge if side == "b" else price <= edge
                    if not known:
                        row[18] += 1
                        continue
                    row[17] += 1
                    change = quantity - old.get(price, 0.0)
                    direction = int(change < 0)
                    distance = max(0.0, sign * (price - mid) / spread)
                    for i, width in enumerate(FLOW_WIDTHS):
                        row[4 * i + offset + direction] += abs(change) * math.exp(-distance / width)
            nb, na, nbq, naq = current[:4]
            ofi = (nb >= bid) * nbq - (nb <= bid) * bq - (na <= ask) * naq + (na >= ask) * aq
            ret = 10000 * math.log((nb + na) / (bid + ask))
            row[12:17] = (ofi, abs(ofi), ret, abs(ret), 1)
        clocks.append(now)
        values.append(row)
        resets.append(last_reset)
        depths.append(0.0 if current is None else current[2] + current[3])
        complete.append(present)
        previous, previous_clock = current, now
        snapshot_count += int(snapshot)
    if not clocks:
        raise ValueError("A nonempty validated native depth stream is required")
    matrix = np.asarray(values)
    if not np.isfinite(matrix).all():
        raise ValueError("Finite native queue-flow measurements required")
    return {"publisher_times": np.asarray(clocks, dtype=np.int64), "flows": matrix,
            "last_reset_times": np.asarray(resets, dtype=np.int64), "top_depth": np.asarray(depths),
            "available": np.asarray(complete, dtype=bool)}, {
                "messages": len(clocks), "snapshots": snapshot_count, "semantics": FLOW_SEMANTICS,
                "known_level_updates": int(matrix[:, 17].sum()), "unresolved_level_updates": int(matrix[:, 18].sum()),
                "atomic_replay_verified": True, "snapshots_and_gaps_do_not_generate_flow": True,
                "outside_prior_visible_depth_excluded": True}


def sample_native_bybit_flow(events, decision_times, *, delays_ms=(100, 500), windows_ms=(1000, 5000, 15000)):
    """Integrate [decision-delay-window, decision-delay), with coverage masks."""
    clock, delays, windows = map(np.asarray, (decision_times, delays_ms, windows_ms))
    if (clock.ndim != 1 or not len(clock) or not np.issubdtype(clock.dtype, np.integer)
        or (np.diff(clock) <= 0).any() or any(a.ndim != 1 or not len(a) or not np.issubdtype(a.dtype, np.integer) for a in (delays, windows))
        or (delays < 0).any() or (windows <= 0).any() or len(np.unique(delays)) != len(delays) or len(np.unique(windows)) != len(windows)):
        raise ValueError("Increasing integer decisions and distinct valid delays/windows required")
    t, flow = events["publisher_times"], events["flows"]
    if (len(t) == 0 or flow.shape != (len(t), len(FLOW_FIELDS)) or (np.diff(t) < 0).any()
        or not np.isfinite(flow).all() or any(events[k].shape != t.shape for k in ("last_reset_times", "top_depth", "available"))):
        raise ValueError("Aligned, ordered and finite native flow events required")
    query = clock[:, None] - delays[None, :]
    right = np.searchsorted(t, query, side="left")
    last = np.maximum(right - 1, 0)
    available = (right > 0) & events["available"][last] & (query - t[last] <= 1000)
    cumulative = np.vstack([np.zeros((1, flow.shape[1])), np.cumsum(flow, axis=0)])
    lower = query[:, :, None] - windows[None, None, :]
    left = np.searchsorted(t, lower, side="left")
    covered = available[:, :, None] & (events["last_reset_times"][last, None] <= lower)
    totals = cumulative[right[:, :, None]] - cumulative[left]
    totals = np.where(covered[:, :, :, None], totals, 0)
    return {"flow_totals": totals, "window_available": covered,
            "source_publisher_times": np.where(right > 0, t[last], -1),
            "top_depth": np.where(available, events["top_depth"][last], 0),
            "decision_times": clock.copy(), "delays_ms": delays.copy(), "windows_ms": windows.copy()}
