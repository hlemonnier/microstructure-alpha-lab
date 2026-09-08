"""Extend finite native-book knowledge using observed tick and best-quote facts."""

from __future__ import annotations

import heapq
from decimal import Decimal
from functools import lru_cache

from lob_forge.boundary_tardis_depth import _integer, _levels
from lob_forge.boundary_tardis_fast import FastBinanceFuturesDepthState

CERTIFIED_SEMANTICS = "captured_quote_and_explicit_tick_frontiers_v1"
HISTORICAL_TICKS = {"BTCUSDT": "0.1", "ETHUSDT": "0.01"}


class CertifiedBinanceFuturesDepthState(FastBinanceFuturesDepthState):
    """Maintain a proof of known quantities, including explicitly observed zeros.

    An absolute update reveals one tick. A best bid at ID k proves all higher
    bids zero at k; a best ask proves all lower asks zero. Those facts remain
    known after replaying every subsequent update. A quote cannot overwrite
    a quantity from a later batch endpoint, even when k lies inside that batch.

    Quote certificates only extend knowledge after both their capture and the
    necessary depth have arrived. They never refresh depth-source freshness.
    """

    def __init__(self, symbol, *, min_tick=None):
        super().__init__(symbol)
        self.tick = Decimal(str(HISTORICAL_TICKS[symbol] if min_tick is None else min_tick))
        if not self.tick.is_finite() or self.tick <= 0:
            raise ValueError("A finite positive historical price tick is required")
        # Per-instance caches avoid retaining completed books through bound keys.
        self._tick_of = lru_cache(maxsize=131072)(self._price_tick)
        self._price_of = lru_cache(maxsize=131072)(lambda n: float(self.tick * n))
        self._level_u = ({}, {})
        self._frontier_ticks = [None, None]
        self._pending = []
        self._last_capture_ns = None
        self._certificate_u = -1
        self.knowledge_capture_ns = self.knowledge_publisher_ms = None
        self.certification_counts = {"used_quotes": 0, "ignored_older_quotes": 0,
                                     "queued_quotes": 0, "explicit_frontier_ticks": 0}

    def _price_tick(self, price):
        value = Decimal(str(price)) / self.tick
        if not value.is_finite() or value <= 0 or value != value.to_integral_value():
            raise ValueError("Native price is outside the declared historical tick grid")
        return int(value)

    def _observe_clock(self, capture_ns):
        capture_ns = _integer(capture_ns)
        if self._last_capture_ns is not None and capture_ns < self._last_capture_ns:
            raise ValueError("Certificates and depth must remain in capture order")
        self._last_capture_ns = capture_ns

    def disconnect(self):
        super().disconnect()
        self._level_u = ({}, {})
        self._frontier_ticks = [None, None]
        self._pending.clear()
        self._certificate_u = -1
        self.knowledge_capture_ns = self.knowledge_publisher_ms = None

    def _advance_frontiers(self):
        for side, direction in ((0, -1), (1, 1)):
            frontier = self._frontier_ticks[side]
            while frontier + direction in self._level_u[side]:
                frontier += direction
                self.certification_counts["explicit_frontier_ticks"] += 1
            self._frontier_ticks[side] = frontier
        self.bid_frontier, self.ask_frontier = map(self._price_of, self._frontier_ticks)

    def _delta(self, update):
        ticks = [[self._tick_of(p) for p in changes] for changes in update[-2:]]
        before = self.applied_deltas
        super()._delta(update)
        if self.applied_deltas == before:
            return
        for known, prices in zip(self._level_u, ticks):
            known.update(dict.fromkeys(prices, update[1]))
        self._advance_frontiers()

    def apply(self, capture_ns, message):
        self._observe_clock(capture_ns)
        if message["stream"].endswith("@depthSnapshot"):
            data = message["data"]
            bids, asks = _levels(data["bids"], snapshot=True), _levels(data["asks"], snapshot=True)
            if not bids or not asks:
                raise ValueError("Nonempty snapshot sides required")
            self._frontier_ticks = [min(map(self._tick_of, bids)), max(map(self._tick_of, asks))]
            self._level_u = ({}, {})
            self._certificate_u = -1
            self.knowledge_capture_ns = self.knowledge_publisher_ms = None
        result = super().apply(capture_ns, message)
        self._drain_certificates()
        return result

    def observe_quote(self, capture_ns, data):
        self._observe_clock(capture_ns)
        if data.get("s") != self.symbol:
            raise ValueError("Native quote symbol does not match its book")
        uid, event, _transaction = (_integer(data[k]) for k in ("u", "E", "T"))
        bids = _levels([[data["b"], data["B"]]], snapshot=True)
        asks = _levels([[data["a"], data["A"]]], snapshot=True)
        bid, ask = next(iter(bids)), next(iter(asks))
        if bid >= ask:
            raise ValueError("Uncrossed positive native quote required")
        if self.snapshot_id is not None and uid < self.snapshot_id:
            return
        # Inside the already known ranges this quote supplies no new knowledge.
        if self.snapshot_id is not None and bid >= self.bid_frontier and ask <= self.ask_frontier:
            return
        if len(self._pending) >= 10000:
            raise ValueError("Unresolved quote-certificate buffer exceeded its fixed bound")
        heapq.heappush(self._pending, (uid, capture_ns, event, self._tick_of(bid), bids[bid], self._tick_of(ask), asks[ask]))
        self.certification_counts["queued_quotes"] += 1
        self._drain_certificates()

    def _drain_certificates(self):
        while self.ready and self._pending and self._pending[0][0] <= self.last_u:
            quote = heapq.heappop(self._pending)
            uid, capture_ns, event, bid, bid_q, ask, ask_q = quote
            if uid < self.snapshot_id or uid <= self._certificate_u:
                self.certification_counts["ignored_older_quotes"] += 1
                continue
            extends = bid < self._frontier_ticks[0] or ask > self._frontier_ticks[1]
            if not extends:
                continue
            additions = []
            for side, tick, quantity, live in ((0, bid, bid_q, self.bids), (1, ask, ask_q, self.asks)):
                known, frontier = self._level_u[side], self._frontier_ticks[side]
                price = self._price_of(tick)
                in_range = tick >= frontier if side == 0 else tick <= frontier
                last = known.get(tick, self.snapshot_id)
                # A batch ending after this quote owns the current quantity.
                if last <= uid:
                    if in_range or tick in known:
                        if live.get(price, 0.0) != quantity:
                            self.ready = False
                            raise ValueError("Quote certificate contradicts a known price quantity")
                    else:
                        additions.append((side, tick, price, quantity))
                # Existing positives at better prices must postdate the quote.
                for p in live:
                    better = p > price if side == 0 else p < price
                    if better and known.get(self._tick_of(p), self.snapshot_id) <= uid:
                        self.ready = False
                        raise ValueError("Quote certificate contradicts a known better price")
            new_bid, new_ask = self._best_bid, self._best_ask
            for side, tick, price, quantity in additions:
                if side == 0:
                    new_bid = price if new_bid is None else max(new_bid, price)
                else:
                    new_ask = price if new_ask is None else min(new_ask, price)
            if new_bid >= new_ask:
                self.ready = False
                raise ValueError("Quote certificate produced a crossed current book")
            # Both sides and the prospective current BBO passed before mutation.
            for side, tick, price, quantity in additions:
                (self.bids if side == 0 else self.asks)[price] = quantity
                self._level_u[side][tick] = uid
            self._best_bid, self._best_ask = new_bid, new_ask
            self._frontier_ticks = [min(self._frontier_ticks[0], bid), max(self._frontier_ticks[1], ask)]
            self._advance_frontiers()
            self._certificate_u = uid
            self.knowledge_capture_ns = max(self.knowledge_capture_ns or 0, capture_ns)
            self.knowledge_publisher_ms = max(self.knowledge_publisher_ms or 0, event)
            self.certification_counts["used_quotes"] += 1
