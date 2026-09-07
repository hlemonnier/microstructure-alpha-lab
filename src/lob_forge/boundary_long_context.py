"""Causal intraday activity and strictly released outcome context for forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_adaptive_priors import PRIOR_FLOOR, PRIOR_STRENGTH, _clock

HALF_LIVES = (300, 900, 3600)
OBSERVED_FIELDS = (
    "decision_time", "bid", "ask", "bid_qty", "ask_qty", "mid_return_1",
    "quote_updates_in_bucket", "trade_count", "trade_notional",
    "event_price_change_count", "quote_ofi_normalized", "top_imbalance",
)
PEER_OBSERVED = tuple(
    f"context_{name}_{half}" for half in HALF_LIVES
    for name in ("log_quote_count", "log_return_variance")
)
PEER_RELEASED = tuple(f"context_label_neutral_{half}" for half in HALF_LIVES)


def observation_context(frame):
    """Use observed closed one-second buckets only; reset at a coverage gap."""
    raw = frame.loc[:, OBSERVED_FIELDS].astype(float)
    clock = _clock(raw.decision_time.to_numpy(), "Observation times")
    if not len(clock) or not np.isfinite(raw.to_numpy()).all() or (np.diff(clock) != 1000).any():
        raise ValueError("Nonempty finite contiguous one-second observations are required")
    if (raw.bid <= 0).any() or (raw.ask <= raw.bid).any() or (raw[["bid_qty", "ask_qty"]] < 0).any().any():
        raise ValueError("Positive uncrossed quotes and nonnegative quantities are required")
    mid = (raw.bid + raw.ask) / 2
    positive = pd.DataFrame({
        "quote_count": raw.quote_updates_in_bucket,
        "trade_count": raw.trade_count,
        "trade_notional": raw.trade_notional,
        "price_changes": raw.event_price_change_count,
        "return_variance": (10000 * raw.mid_return_1) ** 2,
        "depth_notional": raw.bid * raw.bid_qty + raw.ask * raw.ask_qty,
        "spread_bps": (raw.ask - raw.bid) / mid * 10000,
        "absolute_flow": raw.quote_ofi_normalized.abs(),
    })
    if (positive < 0).any().any():
        raise ValueError("Counts, quantities and activity cannot be negative")
    signed = pd.DataFrame({"flow": raw.quote_ofi_normalized, "imbalance": raw.top_imbalance, "return_bps": 10000 * raw.mid_return_1})
    result = {"context_log_history_seconds": np.log1p((clock - clock[0]) / 1000)}
    for half in HALF_LIVES:
        # The frame is contiguous at one-second cadence, so a row half-life is
        # exactly a clock-time half-life. adjust=True normalizes available mass.
        means = positive.ewm(halflife=half, adjust=True).mean()
        for name in positive:
            result[f"context_log_{name}_{half}"] = np.log1p(means[name].to_numpy())
        for name in ("quote_count", "trade_count", "price_changes", "return_variance"):
            result[f"context_surprise_{name}_{half}"] = np.log1p(positive[name].to_numpy()) - np.log1p(means[name].to_numpy())
        for name, series in signed.ewm(halflife=half, adjust=True).mean().items():
            result[f"context_mean_{name}_{half}"] = series.to_numpy()
    return pd.DataFrame(result)


def released_outcome_context(labels, origin_times, available_times, query_times):
    """Expose smoothed label rates only after the actual future quote timestamp.

    Exponential age uses the original forecast time. This vectorized intraday
    expression cancels the common query-time decay factor when normalizing the
    three class counts. A one-day bound and >=300s half-lives avoid overflow.
    """
    y = np.asarray(labels)
    origins = _clock(origin_times, "Outcome origin times")
    queries = _clock(query_times, "Outcome query times")
    release = np.asarray(available_times)
    if (
        y.shape != origins.shape or not np.isin(y, [-1, 0, 1]).all()
        or release.shape != origins.shape or not np.isfinite(release).all()
        or not np.equal(release, np.floor(release)).all()
        or (release <= origins).any() or (np.diff(release) < 0).any()
    ):
        raise ValueError("Ordered, strictly delayed, finite three-class outcomes are required")
    if not len(queries):
        raise ValueError("At least one query is required")
    anchor = min(queries[0], origins[0]) if len(origins) else queries[0]
    endpoint = max(queries[-1], origins[-1]) if len(origins) else queries[-1]
    if endpoint - anchor > 86400000:
        raise ValueError("This intraday context must be reset at a daily/session boundary")
    # Strict inequality matches the known-quote feature contract: an event at
    # the decision timestamp is not assumed available to that same decision.
    counts_available = np.searchsorted(release, queries, side="left")
    result = {}
    for half in HALF_LIVES:
        weights = np.exp2((origins - anchor) / (1000 * half))
        weighted = np.zeros((len(y), 3))
        weighted[np.arange(len(y)), y.astype(int) + 1] = weights
        cumulative = np.vstack([np.zeros((1, 3)), np.cumsum(weighted, axis=0)])
        counts = PRIOR_STRENGTH / 3 + cumulative[counts_available]
        sums = counts.sum(axis=1)
        probabilities = PRIOR_FLOOR + (1 - 3 * PRIOR_FLOOR) * counts / sums[:, None]
        for column, name in enumerate(("down", "neutral", "up")):
            result[f"context_label_{name}_{half}"] = probabilities[:, column]
        result[f"context_label_log_mass_{half}"] = np.log1p(sums * np.exp2(-(queries - anchor) / (1000 * half)))
    return pd.DataFrame(result)
