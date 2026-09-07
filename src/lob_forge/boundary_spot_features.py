"""Strictly delayed spot trade observations relative to known perpetual quotes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_adaptive_priors import _clock
from lob_forge.boundary_event_clock import _ratio, _sum

TIME_WINDOWS_MS = (1000, 5000, 15000, 60000)
EVENT_WINDOWS = (8, 32, 128)
WINDOW_FIELDS = ("log_count", "log_quantity", "volume_imbalance", "signed_volume_depth", "vwap_basis_bps", "side_mean")
EVENT_FIELDS = WINDOW_FIELDS + ("side_persistence", "log_span_ms", "price_return_bps", "volume_concentration")
SPOT_FEATURES = (
    "spot_available", "spot_log_trade_age_ms", "spot_last_side", "spot_log_last_quantity", "spot_signed_log_run_length", "spot_last_basis_bps",
    *(f"spot_{name}_{h}ms" for h in TIME_WINDOWS_MS for name in ("return_bps", "return_available")),
    *(f"spot_basis_{name}_{h}ms" for h in (60000, 300000, 3600000) for name in ("mean", "deviation")),
    *(f"spot_time_{h}_{name}" for h in TIME_WINDOWS_MS for name in WINDOW_FIELDS),
    *(f"spot_events_{n}_{name}" for n in EVENT_WINDOWS for name in EVENT_FIELDS),
)
PEER_SPOT_FEATURES = ("spot_available", "spot_last_basis_bps", "spot_basis_deviation_60000ms", "spot_return_bps_5000ms",
                      "spot_time_5000_volume_imbalance", "spot_time_1000_log_count", "spot_events_32_volume_imbalance")
SPOT_SEMANTICS = "spot_trade_context_v1_strict_added_information_delay"


def spot_feature_frame(perpetual_observations, trades, *, information_delay_ms):
    if information_delay_ms not in (100, 500):
        raise ValueError("Use a registered 100ms or 500ms spot information delay")
    known = perpetual_observations.loc[:, ["decision_time", "bid", "ask", "bid_qty", "ask_qty"]].astype(float)
    clock = _clock(known.decision_time.to_numpy(), "Spot feature decision times")
    if not len(clock) or not np.isfinite(known.to_numpy()).all() or (known.bid <= 0).any() or (known.ask <= known.bid).any():
        raise ValueError("Finite known positive uncrossed perpetual quotes are required")
    if (known[["bid_qty", "ask_qty"]] < 0).any().any():
        raise ValueError("Known perpetual quantities must be nonnegative")
    if not len(trades):
        result = pd.DataFrame(np.zeros((len(clock), len(SPOT_FEATURES))), columns=SPOT_FEATURES)
        result.insert(0, "decision_time", clock)
        return result
    time_raw = trades.transact_time.to_numpy()
    ids = trades.agg_trade_id.to_numpy()
    if not np.isfinite(time_raw).all() or not np.equal(time_raw, np.floor(time_raw)).all() or (np.diff(time_raw) < 0).any() or (np.diff(ids) <= 0).any():
        raise ValueError("Chronological integer spot times and increasing aggregate IDs are required")
    time_ms = time_raw.astype(np.int64)
    quantity, price = trades.quantity.to_numpy(dtype=float), trades.price.to_numpy(dtype=float)
    if not np.isfinite(quantity).all() or not np.isfinite(price).all() or (quantity <= 0).any() or (price <= 0).any() or not trades.is_buyer_maker.isin([True, False]).all():
        raise ValueError("Spot prices/quantities must be positive finite and maker sides boolean")
    mid, depth = ((known.bid + known.ask) / 2).to_numpy(), (known.bid_qty + known.ask_qty).to_numpy()
    sign = np.where(trades.is_buyer_maker.to_numpy(dtype=bool), -1.0, 1.0)
    available = time_ms + information_delay_ms
    right = np.searchsorted(available, clock, side="left")
    present, last = right > 0, np.maximum(0, right - 1)
    run_start = np.maximum.accumulate(np.where(np.r_[True, sign[1:] != sign[:-1]], np.arange(len(sign)), 0))
    basis = np.where(present, 10000 * np.log(price[last] / mid), 0)
    result = {
        "spot_available": present.astype(float),
        "spot_log_trade_age_ms": np.where(present, np.log1p(np.maximum(0, clock - time_ms[last])), 0),
        "spot_last_side": np.where(present, sign[last], 0),
        "spot_log_last_quantity": np.where(present, np.log1p(quantity[last]), 0),
        "spot_signed_log_run_length": np.where(present, sign[last] * np.log1p(np.minimum(128, last - run_start[last] + 1)), 0),
        "spot_last_basis_bps": basis,
    }
    for horizon in TIME_WINDOWS_MS:
        previous = np.searchsorted(available, clock - horizon, side="left") - 1
        valid = present & (previous >= 0)
        result[f"spot_return_bps_{horizon}ms"] = np.where(valid, (price[last] / price[np.maximum(0, previous)] - 1) * 10000, 0)
        result[f"spot_return_available_{horizon}ms"] = valid.astype(float)
    observed_basis = pd.Series(np.where(present, basis, np.nan))
    for horizon in (60000, 300000, 3600000):
        slow = observed_basis.ewm(halflife=pd.Timedelta(milliseconds=horizon), times=pd.to_datetime(clock, unit="ms"), adjust=True).mean().fillna(0).to_numpy()
        result[f"spot_basis_mean_{horizon}ms"] = slow
        result[f"spot_basis_deviation_{horizon}ms"] = np.where(present, basis - slow, 0)
    persistence = np.r_[0.0, sign[1:] * sign[:-1]]
    windows = [(f"spot_time_{h}_", np.searchsorted(available, clock - h, side="left"), False) for h in TIME_WINDOWS_MS]
    windows += [(f"spot_events_{n}_", np.maximum(0, right - n), True) for n in EVENT_WINDOWS]
    for prefix, left, event_window in windows:
        count = right - left
        volume, signed_volume = _sum(quantity, left, right), _sum(quantity * sign, left, right)
        vwap = price[0] + _ratio(_sum((price - price[0]) * quantity, left, right), volume)
        result[prefix + "log_count"] = np.log1p(count)
        result[prefix + "log_quantity"] = np.log1p(volume)
        result[prefix + "volume_imbalance"] = _ratio(signed_volume, volume)
        result[prefix + "signed_volume_depth"] = _ratio(signed_volume, depth)
        result[prefix + "vwap_basis_bps"] = np.where(volume > 0, (vwap / mid - 1) * 10000, 0)
        result[prefix + "side_mean"] = _ratio(_sum(sign, left, right), count)
        if event_window:
            safe_left = np.minimum(left, len(time_ms) - 1)
            result[prefix + "side_persistence"] = _ratio(_sum(persistence, np.minimum(left + 1, right), right), np.maximum(0, count - 1))
            result[prefix + "log_span_ms"] = np.where(present, np.log1p(np.maximum(0, clock - time_ms[safe_left])), 0)
            result[prefix + "price_return_bps"] = np.where(present, (price[last] / price[safe_left] - 1) * 10000, 0)
            result[prefix + "volume_concentration"] = np.clip(_ratio(_sum(quantity ** 2, left, right), volume ** 2), 0, 1)
    frame = pd.DataFrame({name: result[name] for name in SPOT_FEATURES})
    frame.insert(0, "decision_time", clock)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Spot feature outputs must remain finite")
    return frame
