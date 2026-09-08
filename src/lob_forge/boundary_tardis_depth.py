"""Capture-ordered Binance Futures snapshots and native depth reconstruction."""

from __future__ import annotations

import calendar
import gzip
import json
import math
import re
from collections import deque
from datetime import datetime

CAPTURE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{1,9})Z")


def capture_nanoseconds(text):
    match = CAPTURE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("Explicit UTC ISO capture timestamp with nanosecond precision required")
    second = calendar.timegm(datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").timetuple())
    return second * 1_000_000_000 + int(match[2].ljust(9, "0"))


def iter_tardis_messages(path):
    """Preserve capture order and disconnects; inspect bytes, not file suffixes."""
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
            now = capture_nanoseconds(timestamp)
            if previous is not None and now < previous:
                raise ValueError("Raw provider lines must remain in capture order")
            item = json.loads(message)
            if not isinstance(item, dict) or not isinstance(item.get("stream"), str) or not isinstance(item.get("data"), dict):
                raise ValueError("Exchange-native stream and data objects required")
            previous = now
            yield now, item


def _integer(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Nonnegative native integer IDs and exchange clocks required")
    return value


def _levels(rows, *, snapshot):
    result = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError("A native level must contain price and absolute quantity")
        price, quantity = map(float, row)
        if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity < 0 or (snapshot and quantity == 0):
            raise ValueError("Finite positive prices and valid absolute quantities required")
        if price in result:
            raise ValueError("Duplicate price in an atomic native update")
        result[price] = quantity
    return result


class BinanceFuturesDepthState:
    """Apply the futures U<=snapshot_id<=u bridge, then pu==previous_u.

    Buffered updates are applied at snapshot capture time, never backdated.
    Finite initial snapshot frontiers constrain which depth levels are known.
    A disconnect invalidates the book until another snapshot bridges correctly.
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.buffer = deque(maxlen=10000)
        self.bids, self.asks = {}, {}
        self.ready = False
        self.snapshot_id = self.last_u = None
        self.available_time_ns = self.publisher_time_ms = None
        self.bid_frontier = self.ask_frontier = None
        self.snapshots = self.applied_deltas = self.buffer_evictions = 0

    def disconnect(self):
        self.ready = False
        self.snapshot_id = self.last_u = None
        self.buffer.clear()
        self.bids, self.asks = {}, {}

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
        new_bids, new_asks = self.bids.copy(), self.asks.copy()
        for target, changes in ((new_bids, bids), (new_asks, asks)):
            for price, quantity in changes.items():
                if quantity == 0:
                    target.pop(price, None)
                else:
                    target[price] = quantity
        if not new_bids or not new_asks or max(new_bids) >= min(new_asks):
            self.ready = False
            raise ValueError("Atomic depth update produced an empty or crossed book")
        self.bids, self.asks = new_bids, new_asks
        self.last_u, self.publisher_time_ms, self.ready = final, event_ms, True
        self.applied_deltas += 1

    def apply(self, capture_ns, message):
        capture_ns = _integer(capture_ns)
        stream, data = message["stream"], message["data"]
        if not stream.startswith(self.symbol.lower() + "@"):
            raise ValueError("Depth stream symbol does not match its book")
        if self.available_time_ns is not None and capture_ns < self.available_time_ns:
            raise ValueError("Book messages must remain in capture order")
        if stream.endswith("@depthSnapshot"):
            snapshot_id = _integer(data["lastUpdateId"])
            event_ms = _integer(data["E"])
            bids, asks = _levels(data["bids"], snapshot=True), _levels(data["asks"], snapshot=True)
            if not message.get("generated") or not bids or not asks or max(bids) >= min(asks):
                raise ValueError("Valid generated uncrossed REST snapshot required")
            self.bids, self.asks = bids, asks
            self.bid_frontier, self.ask_frontier = min(bids), max(asks)
            self.snapshot_id = self.last_u = snapshot_id
            self.ready, self.publisher_time_ms = False, event_ms
            self.snapshots += 1
            for update in self.buffer:
                if update[1] >= snapshot_id:
                    self._delta(update)
        elif "@depth@" in stream or stream.endswith("@depth"):
            if data.get("e") != "depthUpdate" or data.get("s") != self.symbol:
                raise ValueError("Native futures depth event and symbol required")
            first, final, previous, event_ms, transaction_ms = (_integer(data[k]) for k in ("U", "u", "pu", "E", "T"))
            if first > final:
                raise ValueError("Native update ID interval is reversed")
            update = (first, final, previous, event_ms, transaction_ms, _levels(data["b"], snapshot=False), _levels(data["a"], snapshot=False))
            self.buffer_evictions += int(len(self.buffer) == self.buffer.maxlen)
            self.buffer.append(update)
            if self.snapshot_id is not None:
                self._delta(update)
        else:
            raise ValueError("Only native depth and generated snapshot messages are accepted")
        self.available_time_ns = capture_ns
        return self.ready

    def snapshot(self, levels=25):
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError("Positive integer depth required")
        if not self.ready or len(self.bids) < levels or len(self.asks) < levels:
            return None
        bids, asks = sorted(self.bids, reverse=True)[:levels], sorted(self.asks)[:levels]
        if bids[-1] < self.bid_frontier or asks[-1] > self.ask_frontier:
            return None
        return [(a, self.asks[a], b, self.bids[b]) for a, b in zip(asks, bids)]
