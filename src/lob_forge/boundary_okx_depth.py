"""Causal counted OKX books with a shrinking, explicitly known price range."""

from __future__ import annotations

import bisect
import gzip
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from lob_forge.boundary_okx_source import CountedDepthMessage, CountedLevel, parse_counted_depth

OKX_DEPTH_SEMANTICS = "okx_counted_absolute_depth_known_range_strict_cutoff_v1"


@dataclass
class _Side:
    bid: bool
    capacity: int
    prices: list[Decimal] = field(default_factory=list)
    levels: dict[Decimal, CountedLevel] = field(default_factory=dict)
    frontier: Decimal | None = None

    def reset(self, levels):
        self.levels = {level.price: level for level in levels}
        self.prices = sorted(self.levels)
        self.frontier = self.prices[0 if self.bid else -1]

    def apply(self, changes):
        outside = trimmed = 0
        for level in changes:
            price = level.price
            if (self.bid and price < self.frontier) or (not self.bid and price > self.frontier):
                outside += 1
                continue
            if not level.quantity:
                if price in self.levels:
                    del self.levels[price]
                    self.prices.pop(bisect.bisect_left(self.prices, price))
            else:
                if price not in self.levels:
                    bisect.insort(self.prices, price)
                self.levels[price] = level
        # Apply an entire side atomically before trimming; update ordering must
        # not discard levels which remain in the final visible window.
        while len(self.prices) > self.capacity:
            removed = self.prices.pop(0 if self.bid else -1)
            del self.levels[removed]
            trimmed += 1
        if trimmed:
            self.frontier = self.prices[0 if self.bid else -1]
        return outside, trimmed

    def top(self, depth):
        prices = self.prices[-depth:][::-1] if self.bid else self.prices[:depth]
        return [self.levels[p] for p in prices]


class CountedDepthState:
    """Maintain only prices whose complete range is still known.

    The completeness assertion assumes the provider publishes all absolute
    changes within its advertised visible window. There are no sequence IDs in
    these historical records, so this is not a packet-loss certificate.
    """

    def __init__(self, instrument, *, capacity=400, maximum_gap_ms=1000):
        if type(capacity) is not int or not 1 <= capacity <= 32767 or type(maximum_gap_ms) is not int or maximum_gap_ms < 1:
            raise ValueError("Positive integer capacity and publisher gap limit required")
        self.instrument = instrument
        self.capacity, self.maximum_gap_ms = capacity, maximum_gap_ms
        self.asks, self.bids = _Side(False, capacity), _Side(True, capacity)
        self.timestamp = None
        self.initialized = False
        self.segment = 0
        self.stats = Counter()

    def apply(self, message: CountedDepthMessage):
        if message.instrument != self.instrument:
            raise ValueError("A counted book cannot mix instruments")
        now = message.timestamp_ms
        if self.timestamp is not None:
            gap = now - self.timestamp
            if gap < 0:
                raise ValueError("Publisher timestamps cannot be reordered")
            self.stats["maximum_gap_ms"] = max(self.stats["maximum_gap_ms"], gap)
            self.stats["equal_timestamp_messages"] += int(gap == 0)
            if gap > self.maximum_gap_ms:
                self.initialized = False
                self.stats["publisher_gaps"] += 1
        self.timestamp = now
        self.stats["messages"] += 1
        if message.action == "snapshot":
            if len(message.asks) != self.capacity or len(message.bids) != self.capacity:
                raise ValueError("The registered source requires full visible-window snapshots")
            if not self.initialized:
                self.segment += 1
            self.asks.reset(message.asks)
            self.bids.reset(message.bids)
            self.initialized = True
            if not self.stats["snapshots"]:
                self.stats["first_snapshot_time"] = now
            self.stats["snapshots"] += 1
        elif message.action == "update":
            if not self.initialized:
                self.stats["updates_ignored_until_snapshot"] += 1
                if not self.stats["snapshots"]:
                    self.stats["initial_updates_ignored"] += 1
                return
            for side, changes in ((self.asks, message.asks), (self.bids, message.bids)):
                outside, trimmed = side.apply(changes)
                self.stats["outside_known_range_updates"] += outside
                self.stats["levels_removed_at_visible_frontier"] += trimmed
        else:
            raise ValueError("Only complete snapshots or absolute updates are supported")
        if not self.asks.prices or not self.bids.prices:
            self.initialized = False
            self.stats["empty_known_side_resets"] += 1
        elif self.bids.prices[-1] >= self.asks.prices[0]:
            self.initialized = False
            self.stats["crossed_book_resets"] += 1

    def available(self, cutoff, depth):
        return (self.initialized and self.timestamp < cutoff
                and cutoff - self.timestamp <= self.maximum_gap_ms
                and min(len(self.asks.prices), len(self.bids.prices)) >= depth)


def sample_counted_depth(messages, decision_times, instrument, *, depth=100,
                         delays_ms=(100, 500), capacity=400, maximum_gap_ms=1000, cadence_ms=100):
    """Sample publisher times strictly below each coarsened delayed cutoff.

    The 100ms query grid is an information restriction, not a reconstruction of
    the free live feed or an observation of network arrival time. No intermediate
    10ms event-count features are produced. Missing values have explicit masks.
    """
    clock, delays = np.asarray(decision_times), np.asarray(delays_ms)
    if (clock.ndim != 1 or not len(clock) or not np.issubdtype(clock.dtype, np.integer)
        or (clock < 0).any() or (clock >= 2**53).any() or (np.diff(clock) <= 0).any()):
        raise ValueError("Increasing exact nonnegative integer decision clocks required")
    if (delays.ndim != 1 or not len(delays) or not np.issubdtype(delays.dtype, np.integer)
        or (delays < 0).any() or (delays >= 2**53).any() or len(np.unique(delays)) != len(delays)
        or type(depth) is not int or not 1 <= depth <= capacity
        or type(cadence_ms) is not int or cadence_ms < 1):
        raise ValueError("Valid distinct delays, depth and integer query cadence required")
    clock, delays = clock.astype(np.int64), delays.astype(np.int64)
    cutoffs = ((clock[:, None] - delays[None, :]) // cadence_ms) * cadence_ms
    query = cutoffs.ravel()
    order = np.argsort(query, kind="stable")
    shape = (len(query), depth)
    values = np.zeros((*shape, 4), dtype=np.float64)
    counts = np.zeros((*shape, 2), dtype=np.int64)
    timestamps = np.full(len(query), -1, dtype=np.int64)
    segments = np.full(len(query), -1, dtype=np.int64)
    known_depths = np.zeros((len(query), 2), dtype=np.int16)
    book = CountedDepthState(instrument, capacity=capacity, maximum_gap_ms=maximum_gap_ms)
    cursor, first = 0, None

    def emit(index):
        cutoff = query[index]
        if not book.available(cutoff, 1):
            return
        timestamps[index], segments[index] = book.timestamp, book.segment
        known_depths[index] = len(book.asks.prices), len(book.bids.prices)
        asks, bids = book.asks.top(depth), book.bids.top(depth)
        for side, levels in enumerate((asks, bids)):
            n = len(levels)
            values[index, :n, side * 2] = [float(level.price) for level in levels]
            values[index, :n, side * 2 + 1] = [float(level.quantity) for level in levels]
            counts[index, :n, side] = [level.order_count for level in levels]

    for message in messages:
        now = message.timestamp_ms
        if book.timestamp is not None and now < book.timestamp:
            raise ValueError("Publisher clocks cannot regress, including after the last query")
        if first is None:
            first = now
        while cursor < len(order) and query[order[cursor]] <= now:
            emit(order[cursor])
            cursor += 1
        book.apply(message)
    if first is None:
        raise ValueError("A nonempty counted-depth source is required")
    while cursor < len(order):
        emit(order[cursor])
        cursor += 1
    shape = (len(clock), len(delays))
    present = timestamps >= 0
    if (not np.all(timestamps[present] < query[present]) or not np.isfinite(values).all()
        or (values[:, :, (1, 3)] < 0).any()):
        raise ValueError("Source features crossed the strict boundary or became nonfinite")
    for side in (0, 1):
        occupied = np.arange(depth)[None, :] < known_depths[:, side, None]
        px, qty = values[:, :, side * 2], values[:, :, side * 2 + 1]
        if ((px[occupied] <= 0).any() or (qty[occupied] <= 0).any() or (counts[:, :, side][occupied] <= 0).any()
            or (((np.diff(px, axis=1) <= 0) if side == 0 else (np.diff(px, axis=1) >= 0)) & occupied[:, 1:]).any()):
            raise ValueError("Float64 projection destroyed positive ordered counted depth")
    if (values[present, 0, 2] >= values[present, 0, 0]).any():
        raise ValueError("Float64 projection locked or crossed an available book")
    return {"depth": values.reshape(*shape, depth, 4), "order_counts": counts.reshape(*shape, depth, 2),
        "publisher_times": timestamps.reshape(shape), "segment_ids": segments.reshape(shape),
        "known_depths": known_depths.reshape(*shape, 2), "query_cutoffs": cutoffs,
        "decision_times": clock, "delays_ms": delays}, {
            **dict(book.stats), "first_publisher_time": first, "last_publisher_time": book.timestamp,
            "strict_publisher_cutoff_checks": True, "decimal_price_ordering": True,
            "source_sequence_ids_available": False, "source_arrival_times_available": False,
            "semantics": OKX_DEPTH_SEMANTICS}


def counted_archive_messages(path, *, instrument, start_ms, end_ms, audit,
                             maximum_member_bytes=16 * 1024**3, maximum_line_bytes=1024**2):
    """Stream one data member and consume the full gzip trailer without extraction."""
    expected_member = path.name.removesuffix(".tar.gz") + ".data"
    count = source_bytes = maximum_line = 0
    with gzip.open(path, "rb") as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as package:
            for member in package:
                if (count or not member.isfile() or member.name != expected_member
                    or not 0 < member.size <= maximum_member_bytes):
                    raise ValueError("One bounded, correctly named historical data member required")
                count += 1
                with package.extractfile(member) as source:
                    while line := source.readline(maximum_line_bytes + 1):
                        source_bytes += len(line)
                        maximum_line = max(maximum_line, len(line))
                        if len(line) > maximum_line_bytes or source_bytes > member.size or not line.strip():
                            raise ValueError("Malformed or oversized source record")
                        message = parse_counted_depth(line, instrument=instrument)
                        if not start_ms <= message.timestamp_ms < end_ms:
                            raise ValueError("Source timestamp lies outside its registered UTC date")
                        yield message
                if source_bytes != member.size:
                    raise ValueError("Incomplete decompressed source member")
        tail = 0
        while block := decoded.read(1024 * 1024):
            tail += len(block)
            if tail > 1024**2 or any(block):
                raise ValueError("Unexpected nonzero or oversized tar tail")
    if count != 1:
        raise ValueError("Exactly one complete source member required")
    audit.update(member=expected_member, member_bytes=source_bytes, maximum_line_bytes=maximum_line,
                 complete_gzip_crc_and_size_trailer_verified=True, trailing_tar_bytes=tail)
