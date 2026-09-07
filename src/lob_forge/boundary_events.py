"""Vectorized forecast-only observations from immutable raw BBO and trade events.

This dataset deliberately omits execution-path and maker-fill fields. It does not
replace the canonical execution dataset or certify a trading strategy.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.features import AGG_TRADE_COLUMNS, BOOK_TICKER_COLUMNS
from lob_forge.label_math import exact_label_array

FORECAST_SEMANTICS_VERSION = "forecast_raw_events_v2_exact_labels"

EVENT_FEATURES = (
    "event_imbalance_mean",
    "event_imbalance_std",
    "event_price_change_count",
    "event_signed_price_changes",
    "event_return_variation_bps",
    "event_positive_ofi",
    "event_negative_ofi",
    "log_bid_price_age",
    "log_ask_price_age",
    "event_ofi_50ms",
    "event_count_50ms",
    "event_return_50ms",
    "event_trade_flow_50ms",
    "event_trade_volume_50ms",
    "event_ofi_200ms",
    "event_count_200ms",
    "event_return_200ms",
    "event_trade_flow_200ms",
    "event_trade_volume_200ms",
)


def read_archive(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if info.filename.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Exactly one CSV member is required")
        with archive.open(members[0]) as source:
            frame = pd.read_csv(source, usecols=columns, float_precision="round_trip")
    if list(frame.columns) != columns:
        raise ValueError("Unexpected archive column order")
    return frame


def event_frame(quotes: pd.DataFrame, trades: pd.DataFrame, *, min_tick: float) -> pd.DataFrame:
    if min_tick <= 0 or not np.isfinite(min_tick) or len(quotes) < 2:
        raise ValueError("Positive finite tick and at least two quote events required")
    t = quotes["event_time"].to_numpy(dtype=np.int64)
    update = quotes["update_id"].to_numpy(dtype=np.int64)
    bid, ask, bq, aq = (
        quotes[key].to_numpy(dtype=float)
        for key in ["best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"]
    )
    if (np.diff(t) < 0).any() or (np.diff(update) <= 0).any():
        raise ValueError("Quotes require increasing IDs and nondecreasing event timestamps")
    if (
        any(not np.isfinite(a).all() for a in [bid, ask, bq, aq])
        or (bid <= 0).any()
        or (ask <= bid).any()
        or (bq < 0).any()
        or (aq < 0).any()
    ):
        raise ValueError("Finite uncrossed positive quotes and nonnegative quantities required")
    bucket = t // 1000
    starts = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1]
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    counts = ends - starts + 1
    decisions = (bucket[ends] + 1) * 1000
    depth = bq + aq
    imbalance = np.divide(bq - aq, depth, out=np.zeros_like(depth), where=depth > 0)
    mid = (bid + ask) / 2
    ofi = np.zeros(len(t))
    ofi[1:] = (
        (bid[1:] >= bid[:-1]) * bq[1:]
        - (bid[1:] <= bid[:-1]) * bq[:-1]
        - (ask[1:] <= ask[:-1]) * aq[1:]
        + (ask[1:] >= ask[:-1]) * aq[:-1]
    )

    def total(values):
        return np.add.reduceat(values, starts)

    def normalized(values):
        return np.divide(values, depth[ends], out=np.zeros(len(ends)), where=depth[ends] > 0)

    frame = pd.DataFrame(
        {
            "decision_time": decisions,
            "bid": bid[ends],
            "ask": ask[ends],
            "bid_qty": bq[ends],
            "ask_qty": aq[ends],
            "mid": mid[ends],
            "quote_age_ms": decisions - t[ends],
            "quote_updates_in_bucket": counts,
            "top_imbalance": imbalance[ends],
            "quote_ofi": total(ofi),
        }
    )
    frame["quote_ofi_normalized"] = normalized(frame["quote_ofi"].to_numpy())
    frame["quote_ofi_5_normalized"] = normalized(frame["quote_ofi"].rolling(5, min_periods=1).sum().to_numpy())
    frame["top_imbalance_mean_5"] = frame["top_imbalance"].rolling(5, min_periods=1).mean()
    # Difference/division order matches the canonical scalar implementation.
    frame["mid_return_1"] = (frame["mid"].diff() / frame["mid"].shift()).fillna(0)
    frame["mid_return_5"] = ((frame["mid"] - frame["mid"].shift(5)) / frame["mid"].shift(5)).fillna(0)
    frame["realized_volatility_5"] = frame["mid_return_1"].pow(2).rolling(5, min_periods=1).sum().pow(0.5)
    for name in ["depth_imbalance_1pct", "notional_imbalance_1pct", "depth_imbalance_5pct", "notional_imbalance_5pct"]:
        frame[name] = 0.0
    frame["event_imbalance_mean"] = total(imbalance) / counts
    frame["event_imbalance_std"] = np.sqrt(
        np.maximum(0, total(imbalance**2) / counts - frame["event_imbalance_mean"] ** 2)
    )
    moves = np.r_[0, np.diff(mid)]
    frame["event_price_change_count"] = total((moves != 0).astype(float))
    frame["event_signed_price_changes"] = total(np.sign(moves))
    frame["event_return_variation_bps"] = total(np.r_[0, np.abs(moves[1:]) / mid[:-1] * 10000])
    frame["event_positive_ofi"] = normalized(total(np.maximum(ofi, 0)))
    frame["event_negative_ofi"] = normalized(total(np.maximum(-ofi, 0)))
    for name, price in [("bid", bid), ("ask", ask)]:
        last_change = np.maximum.accumulate(np.where(np.r_[True, np.diff(price) != 0], t, t[0]))
        frame[f"log_{name}_price_age"] = np.log1p(decisions - last_change[ends])
    for horizon in [50, 200]:
        recent = t % 1000 >= 1000 - horizon
        frame[f"event_ofi_{horizon}ms"] = normalized(total(ofi * recent))
        frame[f"event_count_{horizon}ms"] = total(recent.astype(float))
        previous_index = np.maximum(0, np.searchsorted(t, decisions - horizon, side="right") - 1)
        frame[f"event_return_{horizon}ms"] = (mid[ends] - mid[previous_index]) / mid[previous_index] * 10000

    tt = trades["transact_time"].to_numpy(dtype=np.int64)
    qty, price = trades["quantity"].to_numpy(dtype=float), trades["price"].to_numpy(dtype=float)
    maker = trades["is_buyer_maker"]
    if not maker.isin([True, False, "true", "false"]).all():
        raise ValueError("Trade aggressor side must be boolean")
    sell = maker.astype(str).str.lower().eq("true").to_numpy()
    if (
        (np.diff(tt) < 0).any()
        or not np.isfinite(qty).all()
        or not np.isfinite(price).all()
        or (qty < 0).any()
        or (price <= 0).any()
    ):
        raise ValueError("Chronological finite trades with positive prices and nonnegative quantities required")
    slots = np.searchsorted(bucket[ends], tt // 1000)
    valid = slots < len(ends)
    valid &= bucket[ends][np.minimum(slots, len(ends) - 1)] == tt // 1000

    def trade_sum(values):
        return np.bincount(slots[valid], weights=np.asarray(values)[valid], minlength=len(ends))

    frame["trade_count"] = trade_sum(np.ones(len(tt)))
    frame["buy_qty"] = trade_sum(qty * ~sell)
    frame["sell_qty"] = trade_sum(qty * sell)
    frame["trade_qty"] = frame["buy_qty"] + frame["sell_qty"]
    frame["trade_notional"] = trade_sum(qty * price)
    frame["trade_imbalance"] = (frame["buy_qty"] - frame["sell_qty"]) / frame["trade_qty"].where(
        frame["trade_qty"] != 0, 1
    )
    for horizon in [50, 200]:
        recent = tt % 1000 >= 1000 - horizon
        frame[f"event_trade_flow_{horizon}ms"] = normalized(trade_sum(qty * (1 - 2 * sell.astype(int)) * recent))
        frame[f"event_trade_volume_{horizon}ms"] = normalized(trade_sum(qty * recent))
    entry = np.searchsorted(t, decisions + 100)
    future = np.searchsorted(t, decisions + 5100)
    resolved = future < len(t)
    safe_entry, safe_future = np.minimum(entry, len(t) - 1), np.minimum(future, len(t) - 1)
    labels = exact_label_array(
        bid[safe_entry], ask[safe_entry], bid[safe_future], ask[safe_future], min_tick=min_tick
    )
    frame["label"] = np.where(resolved, labels, np.nan)
    frame["entry_event_time"] = np.where(resolved, t[safe_entry], np.nan)
    frame["future_event_time"] = np.where(resolved, t[safe_future], np.nan)
    frame["entry_mid"] = np.where(resolved, mid[safe_entry], np.nan)
    frame["future_mid"] = np.where(resolved, mid[safe_future], np.nan)
    frame["feature_semantics_version"] = FORECAST_SEMANTICS_VERSION
    # The last quote bucket is incomplete until a following bucket is observed.
    return frame.iloc[:-1].copy()


def build_event_dataset(book_ticker: Path, agg_trades: Path, *, min_tick: float) -> pd.DataFrame:
    return event_frame(
        read_archive(book_ticker, BOOK_TICKER_COLUMNS), read_archive(agg_trades, AGG_TRADE_COLUMNS), min_tick=min_tick
    )
