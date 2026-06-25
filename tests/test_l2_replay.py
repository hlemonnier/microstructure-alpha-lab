from __future__ import annotations

from lob_forge.data_sources import NormalizedL2Row
from lob_forge.l2_replay import (
    OrderBookReplayer,
    replay_l2_rows,
    validate_l2_replay_contract,
    validate_monotonic_snapshot,
)


def test_replayer_applies_snapshot_group_and_deltas() -> None:
    rows = [
        _row("snapshot", "bid", 99.0, 2.0, sequence=1),
        _row("snapshot", "ask", 100.0, 1.0, sequence=1),
        _row("delta", "bid", 99.5, 3.0, sequence=2),
        _row("delta", "ask", 100.0, 0.0, sequence=3),
        _row("delta", "ask", 100.5, 1.5, sequence=4),
    ]

    replayer, updates, snapshots = replay_l2_rows(rows, depth=2)

    assert updates[0].reset
    assert not updates[1].reset
    assert replayer.snapshot(depth=1).best_bid.price == 99.5
    assert replayer.snapshot(depth=1).best_ask.price == 100.5
    assert snapshots[-1].asks[0].size == 1.5


def test_replayer_flags_sequence_gaps_and_crossed_books() -> None:
    replayer = OrderBookReplayer()

    first = replayer.apply(_row("snapshot", "bid", 101.0, 1.0, sequence=10), row_index=1)
    second = replayer.apply(_row("snapshot", "ask", 100.0, 1.0, sequence=10), row_index=2)
    third = replayer.apply(_row("delta", "ask", 102.0, 1.0, sequence=15), row_index=3)

    assert first.reset
    assert second.crossed
    assert third.sequence_gap
    assert third.reset
    assert third.needs_resnapshot
    assert validate_monotonic_snapshot(replayer.snapshot(depth=2)) == []


def test_top_n_tensor_orders_asks_then_bids_and_pads() -> None:
    replayer = OrderBookReplayer()
    replayer.apply(_row("snapshot", "bid", 99.0, 2.0, sequence=1))
    replayer.apply(_row("snapshot", "bid", 98.0, 3.0, sequence=1))
    replayer.apply(_row("snapshot", "ask", 100.0, 1.0, sequence=1))

    tensor = replayer.top_n_tensor(depth=2)

    assert tensor == [
        [100.0, 1.0, 99.0, 2.0],
        [0.0, 0.0, 98.0, 3.0],
    ]


def test_invalid_rows_are_rejected() -> None:
    replayer = OrderBookReplayer()

    try:
        replayer.apply(_row("delta", "bid", 0.0, 1.0))
    except ValueError as exc:
        assert "price" in str(exc)
    else:
        raise AssertionError("invalid price should be rejected")


def test_l2_replay_contract_validator_accepts_clean_rows() -> None:
    rows = [
        _row("snapshot", "bid", 99.0, 2.0, sequence=1),
        _row("snapshot", "ask", 100.0, 1.0, sequence=1),
        _row("delta", "bid", 99.5, 3.0, sequence=2),
        _row("delta", "ask", 100.5, 1.5, sequence=3),
    ]

    report = validate_l2_replay_contract(rows)

    assert report.rows_checked == 4
    assert report.passed


def test_bybit_replay_uses_update_id_not_seq_jump_for_continuity() -> None:
    rows = [
        _row("snapshot", "bid", 99.0, 2.0, sequence=100, update_id=1, venue="bybit"),
        _row("snapshot", "ask", 100.0, 1.0, sequence=100, update_id=1, venue="bybit"),
        _row("delta", "bid", 99.5, 3.0, sequence=350, update_id=2, venue="bybit"),
    ]

    _, updates, _ = replay_l2_rows(rows)
    report = validate_l2_replay_contract(rows)

    assert [update.sequence_gap for update in updates] == [False, False, False]
    assert report.passed


def test_l2_replay_contract_validator_reports_bad_rows() -> None:
    rows = [
        _row("snapshot", "bid", 101.0, 1.0, sequence=1, timestamp=1000),
        _row("snapshot", "ask", 100.0, 1.0, sequence=1, timestamp=1000),
        _row("delta", "ask", 102.0, 1.0, sequence=5, timestamp=990),
        _row("delta", "ask", 102.0, 1.0, sequence=5, timestamp=990),
    ]

    report = validate_l2_replay_contract(rows)

    assert not report.passed
    assert any("crossed book" in error for error in report.errors)
    assert any("moved backward" in error for error in report.errors)
    assert any("sequence gap" in error for error in report.errors)
    assert any("duplicate update" in error for error in report.errors)


def _row(
    event_type: str,
    side: str,
    price: float,
    size: float,
    *,
    sequence: int | None = None,
    update_id: int | None = None,
    timestamp: int = 1,
    venue: str = "test",
) -> NormalizedL2Row:
    return NormalizedL2Row(
        event_type=event_type,
        exchange_timestamp=timestamp,
        side=side,
        price=price,
        size=size,
        sequence=sequence,
        update_id=update_id,
        venue=venue,
        symbol="BTC-USDT",
    )
