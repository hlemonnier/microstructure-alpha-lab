"""Stationary, multiscale sequences of strictly observed BBO updates.

These are top-of-book observations, not reconstructed cancellations or L2.
Each coarse step summarizes eight consecutive events, rather than dropping the
seven intervening flows. All prices and sizes are relative to the known quote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUOTE_SEQUENCE_SEMANTICS = "strict_quote_event_blocks_v1_length64_strides1_8"
SEQUENCE_LENGTH = 64
EVENT_STRIDES = (1, 8)
SEQUENCE_CHANNELS = (
    "bid_offset_spreads", "ask_offset_spreads", "bid_quantity_depth", "ask_quantity_depth",
    "imbalance", "ofi_depth", "same_price_bid_change_depth", "same_price_ask_change_depth",
    "log_endpoint_age_ms", "log_block_duration_ms", "absolute_ofi_depth",
    "mid_change_spreads", "price_change_fraction", "available",
)


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass
class QuoteSequenceSource:
    times: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_quantity: np.ndarray
    ask_quantity: np.ndarray
    prefixes: np.ndarray

    @classmethod
    def from_frame(cls, quotes):
        raw_times = quotes.event_time.to_numpy()
        raw_ids = quotes.update_id.to_numpy()
        if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in (raw_times, raw_ids)):
            raise ValueError("Integer quote timestamps and update IDs required")
        times, ids = raw_times.astype(np.int64), raw_ids.astype(np.int64)
        if not len(times) or (np.diff(times) < 0).any() or (np.diff(ids) <= 0).any():
            raise ValueError("Chronological quotes with increasing update IDs required")
        bid, ask, bq, aq = (quotes[k].to_numpy(dtype=float) for k in (
            "best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"))
        if any(not np.isfinite(a).all() for a in (bid, ask, bq, aq)) or (bid <= 0).any() or (ask <= bid).any() or (bq < 0).any() or (aq < 0).any() or ((bq + aq) <= 0).any():
            raise ValueError("Finite positive uncrossed quotes and positive total displayed depth required")
        ofi = np.r_[0.0, (bid[1:] >= bid[:-1]) * bq[1:] - (bid[1:] <= bid[:-1]) * bq[:-1]
                    - (ask[1:] <= ask[:-1]) * aq[1:] + (ask[1:] >= ask[:-1]) * aq[:-1]]
        bid_change = np.r_[0.0, np.where(bid[1:] == bid[:-1], np.diff(bq), 0.0)]
        ask_change = np.r_[0.0, np.where(ask[1:] == ask[:-1], np.diff(aq), 0.0)]
        price_change = np.r_[0.0, (np.diff(bid) + np.diff(ask)) != 0]
        flow = np.column_stack([ofi, bid_change, ask_change, np.abs(ofi), price_change])
        prefixes = np.vstack([np.zeros((1, flow.shape[1])), np.cumsum(flow, axis=0, dtype=float)])
        return cls(times, bid, ask, bq, aq, prefixes)

    def observe(self, decision_times):
        clock = np.asarray(decision_times)
        if clock.ndim != 1 or not np.issubdtype(clock.dtype, np.integer) or not len(clock) or (np.diff(clock) <= 0).any():
            raise ValueError("Increasing nonempty integer-millisecond decisions required")
        right = np.searchsorted(self.times, clock, side="left")
        if (right == 0).any():
            raise ValueError("A quote strictly before every decision is required")
        current = right - 1
        spread = (self.ask[current] - self.bid[current])[:, None]
        depth = (self.bid_quantity[current] + self.ask_quantity[current])[:, None]
        # Compute differences against the bid to avoid adding large price levels.
        half_spread = spread / 2
        result = np.zeros((len(clock), len(EVENT_STRIDES), SEQUENCE_LENGTH, len(SEQUENCE_CHANNELS)), dtype=np.float32)
        offsets = np.arange(SEQUENCE_LENGTH - 1, -1, -1)
        for scale, stride in enumerate(EVENT_STRIDES):
            end = np.maximum(0, right[:, None] - stride * offsets)
            start = np.maximum(0, end - stride)
            available = end > 0
            last = np.maximum(0, end - 1)
            preceding = np.maximum(0, start - 1)
            size = np.maximum(1, end - start)
            event_depth = self.bid_quantity[last] + self.ask_quantity[last]
            flows = self.prefixes[end] - self.prefixes[start]
            values = (
                signed_log1p((self.bid[last] - self.bid[current, None] - half_spread) / spread),
                signed_log1p((self.ask[last] - self.bid[current, None] - half_spread) / spread),
                np.log1p(self.bid_quantity[last] / depth),
                np.log1p(self.ask_quantity[last] / depth),
                (self.bid_quantity[last] - self.ask_quantity[last]) / event_depth,
                signed_log1p(flows[:, :, 0] / depth),
                signed_log1p(flows[:, :, 1] / depth),
                signed_log1p(flows[:, :, 2] / depth),
                np.log1p(np.maximum(0, clock[:, None] - self.times[last])),
                np.log1p(self.times[last] - self.times[preceding]),
                np.log1p(np.maximum(0, flows[:, :, 3]) / depth),
                signed_log1p(((self.bid[last] - self.bid[preceding]) + (self.ask[last] - self.ask[preceding])) / (2 * spread)),
                np.clip(flows[:, :, 4] / size, 0, 1),
                available.astype(float),
            )
            for channel, value in enumerate(values):
                result[:, scale, :, channel] = np.where(available, value, 0)
        if not np.isfinite(result).all():
            raise ValueError("Quote sequences must remain finite")
        return result


def select_sequence_rows(saved_times, requested_times):
    saved, requested = np.asarray(saved_times), np.asarray(requested_times)
    if saved.ndim != 1 or requested.ndim != 1 or not len(saved) or not len(requested) or (np.diff(saved) <= 0).any() or (np.diff(requested) <= 0).any():
        raise ValueError("Nonempty increasing saved and requested clocks required")
    index = np.searchsorted(saved, requested)
    if (index >= len(saved)).any() or not np.array_equal(saved[np.minimum(index, len(saved) - 1)], requested):
        raise ValueError("Every requested observation must have an exact saved sequence")
    return index
