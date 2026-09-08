"""Equivalent native replay with cached clocks and atomic in-place book updates."""

from __future__ import annotations

import calendar
import gzip
import heapq
import json
from datetime import datetime
from functools import lru_cache

from lob_forge.boundary_tardis_depth import CAPTURE_PATTERN, BinanceFuturesDepthState


@lru_cache(maxsize=4096)
def _unix_second(text):
    return calendar.timegm(datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").timetuple())


def fast_capture_nanoseconds(text):
    match = CAPTURE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("Explicit UTC ISO capture timestamp with nanosecond precision required")
    return _unix_second(match[1]) * 1_000_000_000 + int(match[2].ljust(9, "0"))


def iter_tardis_messages_fast(path):
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    previous = None
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                yield None, None
                continue
            timestamp, message = line.rstrip("\n").split(" ", 1)
            now = fast_capture_nanoseconds(timestamp)
            if previous is not None and now < previous:
                raise ValueError("Raw provider lines must remain in capture order")
            item = json.loads(message)
            if not isinstance(item, dict) or not isinstance(item.get("stream"), str) or not isinstance(item.get("data"), dict):
                raise ValueError("Exchange-native stream and data objects required")
            previous = now
            yield now, item


class FastBinanceFuturesDepthState(BinanceFuturesDepthState):
    """Validate prospective best quotes before mutating either live side.

    Unlike the reference class, successful deltas mutate the side dictionaries
    in place. Consumers must not retain them as historical book snapshots.
    """

    def __init__(self, symbol):
        super().__init__(symbol)
        self._best_bid = self._best_ask = None

    def disconnect(self):
        super().disconnect()
        self._best_bid = self._best_ask = None

    def apply(self, capture_ns, message):
        if message["stream"].endswith("@depthSnapshot"):
            self._best_bid = self._best_ask = None
        return super().apply(capture_ns, message)

    def _delta(self, update):
        first, final, previous, event_ms, transaction_ms, bids, asks = update
        if not self.ready:
            if final < self.snapshot_id:
                return
            if not first <= self.snapshot_id <= final:
                raise ValueError("Initial futures delta does not overlap snapshot ID")
        elif previous != self.last_u or final <= self.last_u:
            self.ready = False
            raise ValueError("Binance Futures pu/u continuity failed; a new snapshot is required")
        best_bid = self._best_bid if self._best_bid is not None else max(self.bids)
        best_ask = self._best_ask if self._best_ask is not None else min(self.asks)
        if bids.get(best_bid) == 0:
            best_bid = max((p for p in self.bids if bids.get(p) != 0), default=None)
        if asks.get(best_ask) == 0:
            best_ask = min((p for p in self.asks if asks.get(p) != 0), default=None)
        new_bid = max((p for p, q in bids.items() if q > 0), default=None)
        new_ask = min((p for p, q in asks.items() if q > 0), default=None)
        if new_bid is not None:
            best_bid = new_bid if best_bid is None else max(best_bid, new_bid)
        if new_ask is not None:
            best_ask = new_ask if best_ask is None else min(best_ask, new_ask)
        if best_bid is None or best_ask is None or best_bid >= best_ask:
            self.ready = False
            raise ValueError("Atomic depth update produced an empty or crossed book")
        # All field, continuity and whole-message book checks precede mutation.
        for target, changes in ((self.bids, bids), (self.asks, asks)):
            for price, quantity in changes.items():
                if quantity == 0:
                    target.pop(price, None)
                else:
                    target[price] = quantity
        self._best_bid, self._best_ask = best_bid, best_ask
        self.last_u, self.publisher_time_ms, self.ready = final, event_ms, True
        self.applied_deltas += 1

    def snapshot(self, levels=25):
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError("Positive integer depth required")
        if not self.ready or len(self.bids) < levels or len(self.asks) < levels:
            return None
        bids, asks = heapq.nlargest(levels, self.bids), heapq.nsmallest(levels, self.asks)
        if bids[-1] < self.bid_frontier or asks[-1] > self.ask_frontier:
            return None
        return [(a, self.asks[a], b, self.bids[b]) for a, b in zip(asks, bids)]
