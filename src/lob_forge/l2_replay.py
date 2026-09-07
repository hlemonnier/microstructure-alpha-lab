from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from lob_forge.data_sources import NormalizedL2Row


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class L2BookSnapshot:
    bids: list[BookLevel]
    asks: list[BookLevel]
    crossed: bool

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None


@dataclass(frozen=True)
class ReplayUpdate:
    row_index: int
    event_type: str
    side: str
    price: float
    size: float
    sequence_gap: bool
    reset: bool
    crossed: bool
    best_bid: float | None
    best_ask: float | None
    duplicate_update: bool = False
    needs_resnapshot: bool = False


@dataclass(frozen=True)
class ReplayValidation:
    rows_checked: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


class OrderBookReplayer:
    """Row-level replay diagnostic. Use AtomicOrderBookReplayer for observations.

    Intermediate per-row states may be crossed while an exchange message is being
    expanded. They are deliberately not the source for training or validation.
    """

    def __init__(self, *, max_sequence_step: int = 1) -> None:
        if max_sequence_step <= 0:
            raise ValueError("max_sequence_step must be positive")
        self.max_sequence_step = max_sequence_step
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_sequence: int | None = None
        self.last_update_id: int | None = None
        self.needs_resnapshot = False
        self._active_snapshot_key: tuple[int, int | None, int | None] | None = None
        self._seen_delta_keys: set[tuple[int, int | None, int | None, str, float]] = set()

    def apply(self, row: NormalizedL2Row, *, row_index: int = 0) -> ReplayUpdate:
        _validate_row(row)
        reset = False
        duplicate_update = self._detect_duplicate_update(row)
        sequence_gap = self._detect_sequence_gap(row)

        if row.event_type == "snapshot":
            sequence_gap = False
            snapshot_key = (row.exchange_timestamp, row.sequence, row.update_id)
            if snapshot_key != self._active_snapshot_key:
                self.bids.clear()
                self.asks.clear()
                self._seen_delta_keys.clear()
                self._active_snapshot_key = snapshot_key
                reset = True
                self.last_sequence = row.sequence
                self.last_update_id = row.update_id
            self.needs_resnapshot = False
        else:
            self._active_snapshot_key = None
            if sequence_gap or duplicate_update:
                self.bids.clear()
                self.asks.clear()
                self.needs_resnapshot = True
                reset = True

        levels = self.bids if row.side == "bid" else self.asks
        if row.size <= 0.0:
            levels.pop(row.price, None)
        else:
            levels[row.price] = row.size

        if row.sequence is not None:
            self.last_sequence = row.sequence if self.last_sequence is None else max(self.last_sequence, row.sequence)
        if row.update_id is not None:
            self.last_update_id = (
                row.update_id if self.last_update_id is None else max(self.last_update_id, row.update_id)
            )

        snapshot = self.snapshot(depth=1)
        return ReplayUpdate(
            row_index=row_index,
            event_type=row.event_type,
            side=row.side,
            price=row.price,
            size=row.size,
            sequence_gap=sequence_gap,
            reset=reset,
            needs_resnapshot=self.needs_resnapshot,
            crossed=snapshot.crossed,
            best_bid=snapshot.best_bid.price if snapshot.best_bid else None,
            best_ask=snapshot.best_ask.price if snapshot.best_ask else None,
            duplicate_update=duplicate_update,
        )

    def snapshot(self, *, depth: int) -> L2BookSnapshot:
        if depth <= 0:
            raise ValueError("depth must be positive")
        bids = [
            BookLevel(price, size)
            for price, size in sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[:depth]
        ]
        asks = [BookLevel(price, size) for price, size in sorted(self.asks.items(), key=lambda item: item[0])[:depth]]
        crossed = bool(bids and asks and bids[0].price >= asks[0].price)
        return L2BookSnapshot(bids=bids, asks=asks, crossed=crossed)

    def top_n_tensor(self, *, depth: int, pad: bool = True) -> list[list[float]]:
        snapshot = self.snapshot(depth=depth)
        rows: list[list[float]] = []
        for index in range(depth):
            bid = snapshot.bids[index] if index < len(snapshot.bids) else None
            ask = snapshot.asks[index] if index < len(snapshot.asks) else None
            if not pad and bid is None and ask is None:
                continue
            rows.append(
                [
                    ask.price if ask else 0.0,
                    ask.size if ask else 0.0,
                    bid.price if bid else 0.0,
                    bid.size if bid else 0.0,
                ]
            )
        return rows

    def _detect_duplicate_update(self, row: NormalizedL2Row) -> bool:
        if row.event_type != "delta":
            return False
        key = (row.exchange_timestamp, row.sequence, row.update_id, row.side, row.price)
        if key in self._seen_delta_keys:
            return True
        self._seen_delta_keys.add(key)
        return False

    def _detect_sequence_gap(self, row: NormalizedL2Row) -> bool:
        gap = False
        venue = (row.venue or "").lower()
        update_id_is_primary = venue == "bybit" and row.update_id is not None
        if row.sequence is not None and self.last_sequence is not None and not update_id_is_primary:
            if row.sequence < self.last_sequence:
                gap = True
            elif row.sequence > self.last_sequence + self.max_sequence_step:
                gap = True
        if row.update_id is not None and self.last_update_id is not None:
            if row.update_id < self.last_update_id:
                gap = True
            elif row.update_id > self.last_update_id + self.max_sequence_step:
                gap = True
        return gap


def replay_l2_rows(
    rows: Iterable[NormalizedL2Row],
    *,
    depth: int = 10,
    max_sequence_step: int = 1,
) -> tuple[OrderBookReplayer, list[ReplayUpdate], list[L2BookSnapshot]]:
    replayer = OrderBookReplayer(max_sequence_step=max_sequence_step)
    updates: list[ReplayUpdate] = []
    snapshots: list[L2BookSnapshot] = []
    for index, row in enumerate(rows, start=1):
        updates.append(replayer.apply(row, row_index=index))
        snapshots.append(replayer.snapshot(depth=depth))
    return replayer, updates, snapshots


def validate_monotonic_snapshot(snapshot: L2BookSnapshot) -> list[str]:
    errors: list[str] = []
    bid_prices = [level.price for level in snapshot.bids]
    ask_prices = [level.price for level in snapshot.asks]
    if bid_prices != sorted(bid_prices, reverse=True):
        errors.append("bid prices are not descending")
    if ask_prices != sorted(ask_prices):
        errors.append("ask prices are not ascending")
    if snapshot.crossed:
        errors.append("book is crossed")
    return errors


def validate_l2_replay_contract(
    rows: Iterable[NormalizedL2Row],
    *,
    max_sequence_step: int = 1,
) -> ReplayValidation:
    replayer = AtomicOrderBookReplayer(max_sequence_step=max_sequence_step)
    errors: list[str] = []
    count = 0
    previous_row = None
    for event in iter_l2_events(rows):
        count += len(event)
        row = event[0]
        if previous_row is not None:
            if row.exchange_timestamp < previous_row.exchange_timestamp:
                errors.append(f"through row {count}: exchange_timestamp moved backward")
            prior = previous_row.update_id if row.update_id is not None else previous_row.sequence
            current = row.update_id if row.update_id is not None else row.sequence
            if row.event_type == "delta" and prior is not None and current is not None and current > prior + max_sequence_step:
                errors.append(f"through row {count}: sequence gap requires resnapshot")
        previous_row = row
        try:
            replayer.apply_event(event)
        except ValueError as exc:
            errors.append(f"through row {count}: {exc}")
    return ReplayValidation(rows_checked=count, errors=tuple(errors))


def _validate_row(row: NormalizedL2Row) -> None:
    if row.event_type not in {"snapshot", "delta"}:
        raise ValueError("event_type must be snapshot or delta")
    if row.side not in {"bid", "ask"}:
        raise ValueError("side must be bid or ask")
    if not math.isfinite(row.price) or row.price <= 0.0:
        raise ValueError("price must be finite and positive")
    if not math.isfinite(row.size) or row.size < 0.0:
        raise ValueError("size must be finite and non-negative")
    for name in ("exchange_timestamp", "local_timestamp", "publisher_timestamp", "sequence", "update_id"):
        value = getattr(row, name)
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer")


def l2_event_key(row: NormalizedL2Row) -> tuple:
    """Identity of one exchange message, shared by ingestion and every tensor reader."""
    return (row.venue, row.symbol, row.event_type, row.exchange_timestamp,
            row.local_timestamp, row.sequence, row.update_id)


def iter_l2_events(rows: Iterable[NormalizedL2Row], *, max_rows: int | None = None):
    """Yield whole messages only; a row budget never fabricates a partial book update."""
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    pending: list[NormalizedL2Row] = []
    key = None
    emitted = 0
    for row in rows:
        next_key = l2_event_key(row)
        if pending and next_key != key:
            if max_rows is not None and emitted + len(pending) > max_rows:
                return
            yield pending
            emitted += len(pending)
            if max_rows is not None and emitted >= max_rows:
                return
            pending = []
        key = next_key
        pending.append(row)
    if pending and (max_rows is None or emitted + len(pending) <= max_rows):
        yield pending


class AtomicOrderBookReplayer:
    """Strict single-instrument replay. Validate a complete message before committing it.

    Explicit snapshots establish a new sequence epoch. Deltas require continuity;
    missing/resorted/duplicate messages fail closed. A book may temporarily lose one
    side, but consumers cannot turn that state into a two-sided observation.
    """

    def __init__(self, *, max_sequence_step: int = 1) -> None:
        if max_sequence_step <= 0:
            raise ValueError("max_sequence_step must be positive")
        self.max_sequence_step = max_sequence_step
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.identity: tuple[str | None, str | None] | None = None
        self.last_timestamp: int | None = None
        self.last_local_timestamp: int | None = None
        self.last_sequence: int | None = None
        self.last_update_id: int | None = None
        self.initialized = False
        self.segment = -1

    def apply_event(self, event: list[NormalizedL2Row]) -> L2BookSnapshot:
        if not event:
            raise ValueError("empty L2 event")
        row = event[0]
        errors: list[str] = []
        identity = (row.venue, row.symbol)
        if not row.venue or not row.symbol:
            errors.append("L2 event requires venue and symbol")
        if self.identity is not None and identity != self.identity:
            errors.append("mixed venue/symbol in one replay stream")
        if self.last_timestamp is not None and row.exchange_timestamp < self.last_timestamp:
            errors.append("exchange_timestamp moved backward")
        if (self.last_local_timestamp is not None and row.local_timestamp is not None
                and row.local_timestamp < self.last_local_timestamp):
            errors.append("local_timestamp moved backward")
        if row.sequence is None and row.update_id is None:
            errors.append("sequence or update_id is required")
        seen = set()
        for item in event:
            try:
                _validate_row(item)
            except ValueError as exc:
                errors.append(str(exc))
            if l2_event_key(item) != l2_event_key(row):
                errors.append("inconsistent message identity")
            level_key = (item.side, item.price)
            if level_key in seen:
                errors.append("duplicate update of side/price in one message")
            seen.add(level_key)
        if row.event_type == "delta":
            if not self.initialized:
                errors.append("delta requires an initial valid snapshot or resnapshot")
            primary = row.update_id if row.update_id is not None else row.sequence
            previous = self.last_update_id if row.update_id is not None else self.last_sequence
            if self.last_sequence is not None and row.sequence is not None and row.sequence < self.last_sequence:
                errors.append("secondary sequence moved backward")
            if previous is not None and primary is not None:
                if primary <= previous:
                    errors.append("duplicate update or backward sequence")
                elif primary > previous + self.max_sequence_step:
                    errors.append("sequence gap requires resnapshot")
        bids = {} if row.event_type == "snapshot" else self.bids.copy()
        asks = {} if row.event_type == "snapshot" else self.asks.copy()
        if not errors:
            for item in event:
                levels = bids if item.side == "bid" else asks
                if item.size == 0:
                    levels.pop(item.price, None)
                else:
                    levels[item.price] = item.size
            if bids and asks and max(bids) >= min(asks):
                errors.append("crossed book after atomic message")
        if errors:
            self.initialized = False
            raise ValueError("; ".join(errors))
        primary = row.update_id if row.update_id is not None else row.sequence
        previous = self.last_update_id if row.update_id is not None else self.last_sequence
        new_epoch = (not self.initialized or previous is None or primary is None
                     or primary != previous + 1)
        self.bids, self.asks = bids, asks
        self.identity = identity
        self.last_timestamp = row.exchange_timestamp
        if row.local_timestamp is not None:
            self.last_local_timestamp = row.local_timestamp
        self.last_sequence, self.last_update_id = row.sequence, row.update_id
        self.initialized = True
        if row.event_type == "snapshot" and new_epoch:
            self.segment += 1
        return L2BookSnapshot(
            bids=[BookLevel(p, s) for p, s in sorted(bids.items(), reverse=True)],
            asks=[BookLevel(p, s) for p, s in sorted(asks.items())], crossed=False,
        )
