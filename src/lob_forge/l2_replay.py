from __future__ import annotations

from dataclasses import dataclass
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
    needs_resnapshot: bool = False


class OrderBookReplayer:
    """Deterministic market-by-price book builder for normalized L2 rows."""

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

    def apply(self, row: NormalizedL2Row, *, row_index: int = 0) -> ReplayUpdate:
        _validate_row(row)
        reset = False
        sequence_gap = self._detect_sequence_gap(row)

        if row.event_type == "snapshot":
            snapshot_key = (row.exchange_timestamp, row.sequence, row.update_id)
            if snapshot_key != self._active_snapshot_key:
                self.bids.clear()
                self.asks.clear()
                self._active_snapshot_key = snapshot_key
                reset = True
            self.needs_resnapshot = False
        else:
            self._active_snapshot_key = None
            if sequence_gap:
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
            self.last_update_id = row.update_id if self.last_update_id is None else max(self.last_update_id, row.update_id)

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
        )

    def snapshot(self, *, depth: int) -> L2BookSnapshot:
        if depth <= 0:
            raise ValueError("depth must be positive")
        bids = [
            BookLevel(price, size)
            for price, size in sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[:depth]
        ]
        asks = [
            BookLevel(price, size)
            for price, size in sorted(self.asks.items(), key=lambda item: item[0])[:depth]
        ]
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

    def _detect_sequence_gap(self, row: NormalizedL2Row) -> bool:
        gap = False
        if row.sequence is not None and self.last_sequence is not None:
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


def _validate_row(row: NormalizedL2Row) -> None:
    if row.event_type not in {"snapshot", "delta"}:
        raise ValueError("event_type must be snapshot or delta")
    if row.side not in {"bid", "ask"}:
        raise ValueError("side must be bid or ask")
    if row.price <= 0.0:
        raise ValueError("price must be positive")
    if row.size < 0.0:
        raise ValueError("size must be non-negative")
