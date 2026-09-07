"""Atomic Bybit depth observations with explicit publisher-time delay assumptions.

The 2023 public archives expose publisher timestamps, not local receive clocks.
No matching-engine or receipt timestamp is inferred. This reader preserves the
native message boundary and rejects sequence gaps before producing later states.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field

import numpy as np

BYBIT_DEPTH_SEMANTICS = "bybit_ob500_atomic_publisher_delay_v1"


@dataclass
class BybitDepthState:
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    timestamp: int | None = None
    update_id: int | None = None
    sequence: int | None = None
    initialized: bool = False

    def apply(self, payload):
        try:
            data = payload["data"]
            kind, timestamp = payload["type"], payload["ts"]
            update, sequence = data["u"], data["seq"]
            if payload["topic"] != f"orderbook.500.{self.symbol}" or data["s"] != self.symbol or kind not in ("snapshot", "delta"):
                raise ValueError("A single registered Bybit 500-level stream is required")
            if any(type(value) is not int or value < 0 for value in (timestamp, update, sequence)):
                raise ValueError("Nonnegative integer publisher time and sequence identifiers required")
            if self.timestamp is not None and timestamp < self.timestamp:
                raise ValueError("Publisher timestamps cannot move backward")
            if kind == "delta" and (not self.initialized or update != self.update_id + 1 or sequence <= self.sequence):
                raise ValueError("A delta requires a snapshot, consecutive update ID and increasing cross sequence")
            changes = {}
            for side in ("b", "a"):
                if not isinstance(data[side], list):
                    raise ValueError("Explicit bid and ask update arrays required")
                changes[side] = []
                seen = set()
                for level in data[side]:
                    if not isinstance(level, (list, tuple)) or len(level) != 2:
                        raise ValueError("Every level requires exactly price and absolute quantity")
                    price, quantity = map(float, level)
                    if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity < 0 or price in seen:
                        raise ValueError("Finite positive unique prices and nonnegative quantities required")
                    seen.add(price)
                    changes[side].append((price, quantity))
            bids, asks = ({}, {}) if kind == "snapshot" else (self.bids.copy(), self.asks.copy())
            for side, book in (("b", bids), ("a", asks)):
                for price, quantity in changes[side]:
                    if quantity == 0:
                        book.pop(price, None)
                    else:
                        book[price] = quantity
            if bids and asks and max(bids) >= min(asks):
                raise ValueError("Crossed order book after a complete message")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.initialized = False
            raise ValueError(f"Invalid atomic Bybit message: {error}") from error
        self.bids, self.asks = bids, asks
        self.timestamp, self.update_id, self.sequence = timestamp, update, sequence
        self.initialized = True

    def snapshot(self, depth):
        if type(depth) is not int or depth < 1:
            raise ValueError("Positive integer snapshot depth required")
        values = np.zeros((depth, 4), dtype=float)
        if not self.initialized:
            return values
        bids = sorted(self.bids.items(), reverse=True)[:depth]
        asks = sorted(self.asks.items())[:depth]
        for i, (price, quantity) in enumerate(asks):
            values[i, :2] = price, quantity
        for i, (price, quantity) in enumerate(bids):
            values[i, 2:] = price, quantity
        return values


def sample_bybit_depth(messages, decision_times, symbol, *, depth=25, delays_ms=(100, 500)):
    """Sample complete states with publisher_time + delay < decision_time.

    Query times are merged without reordering exchange messages. All equal-time
    messages are excluded at a strict query boundary. Unavailable sides remain
    explicitly unavailable; callers cannot treat zero padding as real liquidity.
    """
    clock, delays = np.asarray(decision_times), np.asarray(delays_ms)
    if clock.ndim != 1 or not len(clock) or not np.issubdtype(clock.dtype, np.integer) or (np.diff(clock) <= 0).any():
        raise ValueError("Increasing nonempty integer-millisecond decisions required")
    if delays.ndim != 1 or not len(delays) or not np.issubdtype(delays.dtype, np.integer) or (delays < 0).any() or len(np.unique(delays)) != len(delays) or type(depth) is not int or depth < 1:
        raise ValueError("Distinct nonnegative integer delays and positive depth required")
    query = (clock[:, None] - delays[None, :]).reshape(-1)
    order = np.argsort(query, kind="stable")
    book = BybitDepthState(symbol)
    values = np.zeros((len(query), depth, 4), dtype=float)
    timestamps = np.full(len(query), -1, dtype=np.int64)
    updates, sequences = timestamps.copy(), timestamps.copy()
    available = np.zeros(len(query), dtype=bool)
    cursor, count, snapshots = 0, 0, 0
    first, last = None, None

    def emit(index):
        if book.initialized:
            values[index] = book.snapshot(depth)
            timestamps[index], updates[index], sequences[index] = book.timestamp, book.update_id, book.sequence
            available[index] = len(book.bids) >= depth and len(book.asks) >= depth

    for payload in messages:
        now = payload.get("ts")
        if type(now) is not int:
            raise ValueError("An integer publisher timestamp is required before sampling")
        if last is not None and now < last:
            raise ValueError("Messages cannot be reordered by timestamp")
        while cursor < len(order) and query[order[cursor]] <= now:
            emit(order[cursor])
            cursor += 1
        book.apply(payload)
        count += 1
        snapshots += int(payload["type"] == "snapshot")
        first = now if first is None else first
        last = now
    if not count:
        raise ValueError("A nonempty observed Bybit archive is required")
    while cursor < len(order):
        emit(order[cursor])
        cursor += 1
    shape = (len(clock), len(delays))
    timestamps = timestamps.reshape(shape)
    present = timestamps >= 0
    if not np.all((timestamps + delays[None, :] < clock[:, None])[present]):
        raise ValueError("A source update crossed the strict information boundary")
    return {"depth": values.reshape(*shape, depth, 4), "publisher_times": timestamps,
        "update_ids": updates.reshape(shape), "sequences": sequences.reshape(shape), "available": available.reshape(shape),
        "decision_times": clock.copy(), "delays_ms": delays.copy()}, {
            "messages": count, "snapshots": snapshots, "first_publisher_time": first, "last_publisher_time": last,
            "strict_publisher_delay_checks": True, "sequence_continuity_checks": True,
            "atomic_uncrossed_book_checks": True, "semantics": BYBIT_DEPTH_SEMANTICS}


def bybit_archive_messages(path):
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename.endswith(".data") and not m.is_dir()]
        if len(members) != 1:
            raise ValueError("Exactly one native Bybit data member required")
        with archive.open(members[0]) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
