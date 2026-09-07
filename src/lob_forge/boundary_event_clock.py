"""Forecast observations in aggregate-trade and quote-update event time.

Only raw observations before a decision enter its features. Aggregate trades get
a fixed additional 100 ms information delay; this is an explicit research
assumption, not a measured network-latency guarantee. BBO quantity decreases are
called decreases, never inferred cancellations or identified market orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_CLOCK_SEMANTICS = "event_clock_observations_v1_trade_delay_100ms"
TRADE_DELAY_MS = 100
TRADE_WINDOWS = (8, 32, 128)
QUOTE_WINDOWS = (16, 64, 256)
TRADE_FIELDS = (
    "side_mean", "volume_imbalance", "side_persistence", "volume_concentration", "largest_volume_share",
    "log_span_ms", "vwap_offset_spreads", "price_return_bps", "signed_volume_depth",
)
TRADE_CLOCK_FEATURES = (
    "clock_trade_available", "clock_log_trade_age_ms", "clock_log_buy_age_ms", "clock_log_sell_age_ms",
    "clock_last_trade_offset_spreads", "clock_signed_log_run_length",
    *(f"clock_trade_{window}_{name}" for window in TRADE_WINDOWS for name in TRADE_FIELDS),
)
QUEUE_CLOCK_FEATURES = (
    *(f"clock_{side}_{change}_{horizon}ms" for horizon in (200, 1000) for side in ("bid", "ask") for change in ("increase_depth", "decrease_depth", "log_increase_count", "log_decrease_count")),
    *(f"clock_quote_{window}_{name}" for window in QUOTE_WINDOWS for name in ("ofi_depth", "imbalance_mean", "imbalance_change", "price_change_fraction", "signed_price_change_fraction", "log_span_ms")),
)
EVENT_CLOCK_FEATURES = TRADE_CLOCK_FEATURES + QUEUE_CLOCK_FEATURES
PEER_CLOCK_FEATURES = (
    "clock_last_trade_offset_spreads", "clock_signed_log_run_length", "clock_trade_32_volume_imbalance",
    "clock_trade_32_vwap_offset_spreads", "clock_quote_64_ofi_depth", "clock_quote_64_signed_price_change_fraction",
)


def _sum(values, left, right):
    prefix = np.r_[0.0, np.cumsum(values, dtype=float)]
    return prefix[right] - prefix[left]


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(np.asarray(numerator, dtype=float)), where=denominator > 0)


def event_clock_frame(quotes, trades, decision_times):
    """Observe requested decisions without receiving labels or future prices."""
    raw_clock = np.asarray(decision_times)
    if raw_clock.ndim != 1 or not len(raw_clock) or not np.isfinite(raw_clock).all() or not np.equal(raw_clock, np.floor(raw_clock)).all():
        raise ValueError("A nonempty integer-millisecond decision vector is required")
    clock = raw_clock.astype(np.int64)
    qt = quotes.event_time.to_numpy(dtype=np.int64)
    ids = quotes.update_id.to_numpy(dtype=np.int64)
    if not len(qt) or (np.diff(qt) < 0).any() or (np.diff(ids) <= 0).any() or (np.diff(clock) <= 0).any() or clock[0] <= qt[0]:
        raise ValueError("Chronological quotes, increasing update IDs, and a quote strictly before each decision are required")
    bid, ask, bq, aq = (quotes[k].to_numpy(dtype=float) for k in ("best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"))
    if any(not np.isfinite(a).all() for a in (bid, ask, bq, aq)) or (bid <= 0).any() or (ask <= bid).any() or (bq < 0).any() or (aq < 0).any():
        raise ValueError("Finite positive uncrossed quotes and nonnegative quantities are required")
    right = np.searchsorted(qt, clock, side="left")
    last = right - 1
    mid = (bid + ask) / 2
    depth = bq + aq
    spread_now, mid_now, depth_now = ask[last] - bid[last], mid[last], depth[last]
    result = {}
    for side, price, quantity in (("bid", bid, bq), ("ask", ask, aq)):
        # A best-price transition replaces the observed queue, so its size change
        # is excluded from the same-price increase/decrease measurements.
        change = np.r_[0.0, np.where(price[1:] == price[:-1], np.diff(quantity), 0.0)]
        for horizon in (200, 1000):
            left = np.searchsorted(qt, clock - horizon, side="left")
            for name, values in (("increase", np.maximum(change, 0)), ("decrease", np.maximum(-change, 0))):
                result[f"clock_{side}_{name}_depth_{horizon}ms"] = _ratio(_sum(values, left, right), depth_now)
                result[f"clock_{side}_log_{name}_count_{horizon}ms"] = np.log1p(_sum(values > 0, left, right))
    ofi = np.zeros(len(qt))
    ofi[1:] = (
        (bid[1:] >= bid[:-1]) * bq[1:] - (bid[1:] <= bid[:-1]) * bq[:-1]
        - (ask[1:] <= ask[:-1]) * aq[1:] + (ask[1:] >= ask[:-1]) * aq[:-1]
    )
    imbalance = _ratio(bq - aq, depth)
    moves = np.r_[0.0, np.diff(mid)]
    for window in QUOTE_WINDOWS:
        left = np.maximum(0, right - window)
        size = right - left
        prefix = f"clock_quote_{window}_"
        result[prefix + "ofi_depth"] = _ratio(_sum(ofi, left, right), depth_now)
        result[prefix + "imbalance_mean"] = _sum(imbalance, left, right) / size
        result[prefix + "imbalance_change"] = imbalance[last] - imbalance[left]
        result[prefix + "price_change_fraction"] = _sum(moves != 0, left, right) / size
        result[prefix + "signed_price_change_fraction"] = _sum(np.sign(moves), left, right) / size
        result[prefix + "log_span_ms"] = np.log1p(clock - qt[left])

    tt = trades.transact_time.to_numpy(dtype=np.int64)
    trade_ids = trades.agg_trade_id.to_numpy(dtype=np.int64)
    quantity, price = (trades[k].to_numpy(dtype=float) for k in ("quantity", "price"))
    maker = trades.is_buyer_maker
    if (np.diff(tt) < 0).any() or (np.diff(trade_ids) <= 0).any() or not maker.isin([True, False, "true", "false"]).all():
        raise ValueError("Chronological aggregate trades, increasing aggregate IDs and boolean maker sides are required")
    if not np.isfinite(quantity).all() or not np.isfinite(price).all() or (quantity < 0).any() or (price <= 0).any():
        raise ValueError("Finite nonnegative aggregate quantities and positive prices are required")
    if not len(tt):
        for name in TRADE_CLOCK_FEATURES:
            result[name] = np.zeros(len(clock))
    else:
        sign = np.where(maker.astype(str).str.lower().eq("true").to_numpy(), -1.0, 1.0)
        available = tt + TRADE_DELAY_MS
        tright = np.searchsorted(available, clock, side="left")
        present = tright > 0
        tlast = np.maximum(0, tright - 1)
        result["clock_trade_available"] = present.astype(float)
        result["clock_log_trade_age_ms"] = np.where(present, np.log1p(np.maximum(0, clock - tt[tlast])), 0)
        for side, signed in (("buy", 1), ("sell", -1)):
            locations = np.maximum.accumulate(np.where(sign == signed, np.arange(len(sign)), -1))
            at = locations[tlast]
            exists = present & (at >= 0)
            result[f"clock_log_{side}_age_ms"] = np.where(exists, np.log1p(np.maximum(0, clock - tt[np.maximum(0, at)])), 0)
        run_start = np.maximum.accumulate(np.where(np.r_[True, sign[1:] != sign[:-1]], np.arange(len(sign)), 0))
        result["clock_signed_log_run_length"] = np.where(present, sign[tlast] * np.log1p(np.minimum(128, tlast - run_start[tlast] + 1)), 0)
        result["clock_last_trade_offset_spreads"] = np.where(present, (price[tlast] - mid_now) / spread_now, 0)
        persistence = np.r_[0.0, sign[1:] * sign[:-1]]
        for window in TRADE_WINDOWS:
            left = np.maximum(0, tright - window)
            size = tright - left
            safe_left = np.minimum(left, len(tt) - 1)
            volume = _sum(quantity, left, tright)
            signed_volume = _sum(quantity * sign, left, tright)
            prefix = f"clock_trade_{window}_"
            result[prefix + "side_mean"] = _ratio(_sum(sign, left, tright), size)
            result[prefix + "volume_imbalance"] = _ratio(signed_volume, volume)
            # The first event in the window has no in-window predecessor.
            result[prefix + "side_persistence"] = _ratio(_sum(persistence, np.minimum(left + 1, tright), tright), np.maximum(0, size - 1))
            result[prefix + "volume_concentration"] = np.clip(_ratio(_sum(quantity**2, left, tright), volume**2), 0, 1)
            largest = pd.Series(quantity).rolling(window, min_periods=1).max().to_numpy()[tlast]
            result[prefix + "largest_volume_share"] = np.where(present, np.clip(_ratio(largest, volume), 0, 1), 0)
            result[prefix + "log_span_ms"] = np.where(present, np.log1p(np.maximum(0, clock - tt[safe_left])), 0)
            # Centering the VWAP calculation reduces cumulative cancellation.
            vwap = price[0] + _ratio(_sum((price - price[0]) * quantity, left, tright), volume)
            result[prefix + "vwap_offset_spreads"] = np.where(volume > 0, (vwap - mid_now) / spread_now, 0)
            result[prefix + "price_return_bps"] = np.where(present, (price[tlast] - price[safe_left]) / mid_now * 10000, 0)
            result[prefix + "signed_volume_depth"] = _ratio(signed_volume, depth_now)
    frame = pd.DataFrame({name: result[name] for name in EVENT_CLOCK_FEATURES})
    frame.insert(0, "decision_time", clock)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Event-clock observations must remain finite")
    return frame
