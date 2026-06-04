import asyncio
import csv
import json
from pathlib import Path

from lob_forge.live_collectors import (
    LiveL2SequenceTracker,
    capture_live_l2,
    build_snapshot_url,
    build_subscribe_message,
    build_websocket_url,
    collector_specs_markdown,
    get_live_collector_spec,
    normalize_live_l2_message,
)


def test_binance_live_collector_builds_snapshot_and_stream_url() -> None:
    stream_url = build_websocket_url("binance", symbol="BTCUSDT")
    snapshot_url = build_snapshot_url("binance", symbol="BTCUSDT", depth=5000)

    assert stream_url.endswith("/btcusdt@depth@100ms")
    assert "symbol=BTCUSDT" in snapshot_url
    assert "limit=5000" in snapshot_url
    assert build_subscribe_message("binance", symbol="BTCUSDT") is None


def test_okx_bybit_coinbase_subscription_messages() -> None:
    okx = build_subscribe_message("okx", symbol="BTC-USDT-SWAP")
    bybit = build_subscribe_message("bybit", symbol="BTCUSDT", depth=200)
    coinbase = build_subscribe_message("coinbase", symbol="BTC-USD")

    assert '"channel": "books"' in okx
    assert "orderbook.200.BTCUSDT" in bybit
    assert '"channel": "level2"' in coinbase


def test_live_collector_specs_expose_sequence_fields() -> None:
    bybit = get_live_collector_spec("bybit")
    markdown = collector_specs_markdown()

    assert "u" in bybit.sequence_fields
    assert "seq" in bybit.sequence_fields
    assert "coinbase" in markdown


def test_binance_live_snapshot_and_delta_normalize_to_l2_rows() -> None:
    snapshot = normalize_live_l2_message(
        "binance",
        {"lastUpdateId": 100, "bids": [["100.0", "1.5"]], "asks": [["100.1", "2.0"]]},
        symbol="BTCUSDT",
        local_timestamp_ms=1_700_000_000_000,
    )
    delta = normalize_live_l2_message(
        "binance",
        {"e": "depthUpdate", "E": 1_700_000_000_010, "s": "BTCUSDT", "U": 101, "u": 102, "b": [["99.9", "0"]], "a": [["100.2", "0.4"]]},
        symbol="BTCUSDT",
        local_timestamp_ms=1_700_000_000_011,
    )
    tracker = LiveL2SequenceTracker("binance")

    assert snapshot.event_type == "snapshot"
    assert snapshot.update_id == 100
    assert snapshot.rows[0].side == "bid"
    assert snapshot.rows[1].side == "ask"
    assert delta.event_type == "delta"
    assert delta.first_update_id == 101
    assert delta.update_id == 102
    assert tracker.apply(snapshot) is False
    assert tracker.apply(delta) is False


def test_okx_bybit_coinbase_live_messages_normalize_sequence_fields() -> None:
    okx = normalize_live_l2_message(
        "okx",
        {
            "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
            "action": "update",
            "data": [{"ts": "1700000000000", "seqId": 12, "prevSeqId": 11, "bids": [["100", "1"]], "asks": [["101", "2"]]}],
        },
        symbol="BTC-USDT-SWAP",
    )
    bybit = normalize_live_l2_message(
        "bybit",
        {"type": "snapshot", "ts": 1_700_000_000_000, "data": {"s": "BTCUSDT", "u": 3, "seq": 7, "b": [["100", "1"]], "a": [["101", "2"]]}},
        symbol="BTCUSDT",
    )
    coinbase = normalize_live_l2_message(
        "coinbase",
        {
            "channel": "l2_data",
            "sequence_num": 9,
            "events": [
                {
                    "type": "update",
                    "product_id": "BTC-USD",
                    "updates": [{"side": "bid", "event_time": "2023-11-14T22:13:20Z", "price_level": "100", "new_quantity": "1.25"}],
                }
            ],
        },
        symbol="BTC-USD",
    )

    assert okx.sequence == 12
    assert okx.prev_sequence == 11
    assert okx.rows[0].venue == "okx"
    assert bybit.sequence == 7
    assert bybit.update_id == 3
    assert coinbase.sequence == 9
    assert coinbase.rows[0].exchange_timestamp == 1_700_000_000_000
    assert coinbase.rows[0].size == 1.25


def test_live_capture_writes_normalized_rows_with_fake_websocket(tmp_path: Path) -> None:
    output = tmp_path / "live_l2.csv"
    fake_ws = _FakeWebSocket(
        [
            json.dumps(
                {
                    "e": "depthUpdate",
                    "E": 1_700_000_000_010,
                    "s": "BTCUSDT",
                    "U": 101,
                    "u": 102,
                    "b": [["99.9", "0"]],
                    "a": [["100.2", "0.4"]],
                }
            )
        ]
    )

    summary = asyncio.run(
        capture_live_l2(
            venue="binance",
            symbol="BTCUSDT",
            output_path=output,
            seconds=1,
            max_messages=1,
            depth=100,
            snapshot_fetcher=lambda url: {
                "lastUpdateId": 100,
                "bids": [["100.0", "1.5"]],
                "asks": [["100.1", "2.0"]],
            },
            websocket_factory=lambda url: _FakeWebSocketContext(fake_ws),
            clock_ms=lambda: 1_700_000_000_000,
        )
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary.messages_seen == 1
    assert summary.rows_written == 4
    assert summary.sequence_gaps == 0
    assert fake_ws.sent == []
    assert rows[0]["event_type"] == "snapshot"
    assert rows[0]["update_id"] == "100"
    assert rows[-1]["event_type"] == "delta"
    assert rows[-1]["update_id"] == "102"


def test_live_capture_resets_on_sequence_gap_without_writing_gap_delta(tmp_path: Path) -> None:
    output = tmp_path / "gap_reset.csv"
    fetches: list[str] = []
    fake_ws = _FakeWebSocket(
        [
            json.dumps(
                {
                    "e": "depthUpdate",
                    "E": 1_700_000_000_010,
                    "s": "BTCUSDT",
                    "U": 105,
                    "u": 106,
                    "b": [["99.9", "0"]],
                    "a": [["100.2", "0.4"]],
                }
            )
        ]
    )

    def snapshot_fetcher(url: str):
        fetches.append(url)
        return {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.5"]],
            "asks": [["100.1", "2.0"]],
        }

    summary = asyncio.run(
        capture_live_l2(
            venue="binance",
            symbol="BTCUSDT",
            output_path=output,
            seconds=1,
            max_messages=1,
            snapshot_fetcher=snapshot_fetcher,
            websocket_factory=lambda url: _FakeWebSocketContext(fake_ws),
            clock_ms=lambda: 1_700_000_000_000,
            fail_on_gap=False,
            reset_on_gap=True,
        )
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary.messages_seen == 1
    assert summary.sequence_gaps == 1
    assert len(fetches) == 2
    assert summary.rows_written == 4
    assert all(row["event_type"] == "snapshot" for row in rows)
    assert {row["update_id"] for row in rows} == {"100"}


class _FakeWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if not self.messages:
            await asyncio.sleep(10)
        return self.messages.pop(0)


class _FakeWebSocketContext:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None
