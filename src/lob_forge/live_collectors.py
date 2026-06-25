from __future__ import annotations

import asyncio
import csv
import inspect
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncContextManager, Awaitable, Callable, Mapping
from urllib.parse import urlencode

from lob_forge.data_sources import NormalizedL2Row
from lob_forge.l2_storage import NORMALIZED_L2_COLUMNS


@dataclass(frozen=True)
class LiveCollectorSpec:
    venue: str
    websocket_url: str
    channel: str
    snapshot_url_template: str | None
    sequence_fields: tuple[str, ...]
    reset_policy: str
    docs_url: str


@dataclass(frozen=True)
class LiveL2Batch:
    venue: str
    symbol: str
    event_type: str
    rows: tuple[NormalizedL2Row, ...]
    sequence: int | None = None
    prev_sequence: int | None = None
    update_id: int | None = None
    first_update_id: int | None = None


@dataclass(frozen=True)
class LiveL2CaptureSummary:
    venue: str
    symbol: str
    output_path: Path
    messages_seen: int
    rows_written: int
    sequence_gaps: int


class LiveL2SequenceTracker:
    """Track exchange sequence continuity for normalized live L2 batches."""

    def __init__(self, venue: str) -> None:
        self.venue = venue
        self.last_sequence: int | None = None
        self.last_update_id: int | None = None

    def reset(self) -> None:
        self.last_sequence = None
        self.last_update_id = None

    def apply(self, batch: LiveL2Batch) -> bool:
        gap = self._detect_gap(batch)
        if batch.sequence is not None:
            self.last_sequence = batch.sequence
        if batch.update_id is not None:
            self.last_update_id = batch.update_id
        return gap

    def _detect_gap(self, batch: LiveL2Batch) -> bool:
        if batch.event_type == "snapshot":
            return False
        if self.venue == "binance":
            if self.last_update_id is None or batch.update_id is None:
                return False
            if batch.first_update_id is not None:
                return batch.first_update_id > self.last_update_id + 1
            return batch.update_id > self.last_update_id + 1
        if self.venue == "okx":
            return (
                self.last_sequence is not None
                and batch.prev_sequence is not None
                and batch.prev_sequence != self.last_sequence
            )
        if self.venue == "coinbase":
            return (
                self.last_sequence is not None
                and batch.sequence is not None
                and batch.sequence != self.last_sequence + 1
            )
        return self.last_sequence is not None and batch.sequence is not None and batch.sequence <= self.last_sequence


LIVE_COLLECTOR_SPECS: tuple[LiveCollectorSpec, ...] = (
    LiveCollectorSpec(
        venue="binance",
        websocket_url="wss://stream.binance.com:9443/ws/{symbol_lower}@depth@100ms",
        channel="diff_depth",
        snapshot_url_template="https://api.binance.com/api/v3/depth?{query}",
        sequence_fields=("U", "u", "lastUpdateId"),
        reset_policy="buffer stream, fetch REST snapshot, require first event to bridge lastUpdateId, reset on U > lastUpdateId + 1",
        docs_url="https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md",
    ),
    LiveCollectorSpec(
        venue="okx",
        websocket_url="wss://ws.okx.com:8443/ws/v5/public",
        channel="books",
        snapshot_url_template="https://www.okx.com/api/v5/market/books?{query}",
        sequence_fields=("seqId", "prevSeqId"),
        reset_policy="subscribe to books/books-l2-tbt, apply snapshot then updates, reset on missing prevSeqId bridge",
        docs_url="https://www.okx.com/docs-v5/en/#order-book-trading-market-data-ws-order-book-channel",
    ),
    LiveCollectorSpec(
        venue="bybit",
        websocket_url="wss://stream.bybit.com/v5/public/{category}",
        channel="orderbook.{depth}.{symbol}",
        snapshot_url_template="https://api.bybit.com/v5/market/orderbook?{query}",
        sequence_fields=("u", "seq", "cts"),
        reset_policy="apply snapshot messages as reset, apply delta messages in u/seq order, reset on gap",
        docs_url="https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook",
    ),
    LiveCollectorSpec(
        venue="coinbase",
        websocket_url="wss://advanced-trade-ws.coinbase.com",
        channel="level2",
        snapshot_url_template=None,
        sequence_fields=("sequence_num", "sequence"),
        reset_policy="subscribe to level2, track increasing sequence per product, resubscribe or reinitialize on gap",
        docs_url="https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-overview",
    ),
)

LIVE_COLLECTORS_BY_VENUE = {spec.venue: spec for spec in LIVE_COLLECTOR_SPECS}


def get_live_collector_spec(venue: str) -> LiveCollectorSpec:
    try:
        return LIVE_COLLECTORS_BY_VENUE[venue]
    except KeyError as exc:
        valid = ", ".join(sorted(LIVE_COLLECTORS_BY_VENUE))
        raise ValueError(f"unknown live collector venue {venue!r}; expected one of: {valid}") from exc


def build_websocket_url(
    venue: str,
    *,
    symbol: str,
    category: str = "spot",
) -> str:
    spec = get_live_collector_spec(venue)
    return spec.websocket_url.format(symbol=symbol, symbol_lower=symbol.lower(), category=category)


def build_snapshot_url(
    venue: str,
    *,
    symbol: str,
    depth: int = 1000,
    category: str = "spot",
) -> str:
    spec = get_live_collector_spec(venue)
    if spec.snapshot_url_template is None:
        raise ValueError(f"{venue} has no public REST snapshot URL configured")
    if venue == "binance":
        query = urlencode({"symbol": symbol, "limit": depth})
    elif venue == "okx":
        query = urlencode({"instId": symbol, "sz": depth})
    elif venue == "bybit":
        query = urlencode({"category": category, "symbol": symbol, "limit": depth})
    else:
        query = urlencode({"symbol": symbol, "limit": depth})
    return spec.snapshot_url_template.format(query=query)


def build_subscribe_message(
    venue: str,
    *,
    symbol: str,
    depth: int = 50,
    category: str = "spot",
) -> str | None:
    if venue == "binance":
        return None
    if venue == "okx":
        return json.dumps(
            {
                "op": "subscribe",
                "args": [{"channel": "books", "instId": symbol}],
            },
            sort_keys=True,
        )
    if venue == "bybit":
        return json.dumps(
            {
                "op": "subscribe",
                "args": [f"orderbook.{depth}.{symbol}"],
            },
            sort_keys=True,
        )
    if venue == "coinbase":
        return json.dumps(
            {
                "type": "subscribe",
                "product_ids": [symbol],
                "channel": "level2",
            },
            sort_keys=True,
        )
    raise ValueError(f"unknown live collector venue {venue!r}")


def collector_specs_markdown() -> str:
    lines = [
        "| Venue | Channel | Sequence Fields | Reset Policy |",
        "| --- | --- | --- | --- |",
    ]
    for spec in LIVE_COLLECTOR_SPECS:
        lines.append(
            "| "
            + " | ".join(
                [
                    spec.venue,
                    spec.channel,
                    ", ".join(spec.sequence_fields),
                    spec.reset_policy,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def normalize_live_l2_message(
    venue: str,
    message: str | bytes | Mapping[str, Any],
    *,
    symbol: str,
    local_timestamp_ms: int | None = None,
) -> LiveL2Batch:
    payload = _decode_message(message)
    if venue == "binance":
        return _normalize_binance_message(payload, symbol=symbol, local_timestamp_ms=local_timestamp_ms)
    if venue == "okx":
        return _normalize_okx_message(payload, symbol=symbol, local_timestamp_ms=local_timestamp_ms)
    if venue == "bybit":
        return _normalize_bybit_message(payload, symbol=symbol, local_timestamp_ms=local_timestamp_ms)
    if venue == "coinbase":
        return _normalize_coinbase_message(payload, symbol=symbol, local_timestamp_ms=local_timestamp_ms)
    raise ValueError(f"unknown live collector venue {venue!r}")


async def capture_live_l2(
    *,
    venue: str,
    symbol: str,
    output_path: Path | str,
    seconds: float = 60.0,
    max_messages: int | None = None,
    depth: int = 50,
    category: str = "spot",
    fail_on_gap: bool = True,
    reset_on_gap: bool = False,
    snapshot_fetcher: Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]] | None = None,
    websocket_factory: Callable[[str], AsyncContextManager[Any]] | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> LiveL2CaptureSummary:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if max_messages is not None and max_messages <= 0:
        raise ValueError("max_messages must be positive when provided")
    if fail_on_gap and reset_on_gap:
        raise ValueError("fail_on_gap and reset_on_gap cannot both be true")

    get_live_collector_spec(venue)
    clock_ms = clock_ms or _now_ms
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracker = LiveL2SequenceTracker(venue)
    messages_seen = 0
    rows_written = 0
    sequence_gaps = 0

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_L2_COLUMNS)
        writer.writeheader()

        snapshot_url = None
        try:
            snapshot_url = build_snapshot_url(venue, symbol=symbol, depth=depth, category=category)
        except ValueError:
            snapshot_url = None
        if snapshot_url is not None:
            snapshot_payload = await _maybe_await((snapshot_fetcher or _fetch_json)(snapshot_url))
            snapshot_batch = normalize_live_l2_message(
                venue,
                snapshot_payload,
                symbol=symbol,
                local_timestamp_ms=clock_ms(),
            )
            tracker.apply(snapshot_batch)
            rows_written += _write_rows(writer, snapshot_batch.rows)

        websocket_url = build_websocket_url(venue, symbol=symbol, category=category)
        subscribe_message = build_subscribe_message(venue, symbol=symbol, depth=depth, category=category)
        websocket_factory = websocket_factory or _default_websocket_factory
        started = time.monotonic()
        async with websocket_factory(websocket_url) as websocket:
            if subscribe_message is not None:
                await websocket.send(subscribe_message)
            while True:
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    break
                if max_messages is not None and messages_seen >= max_messages:
                    break
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch = normalize_live_l2_message(
                    venue,
                    raw,
                    symbol=symbol,
                    local_timestamp_ms=clock_ms(),
                )
                gap = tracker.apply(batch)
                if gap:
                    sequence_gaps += 1
                    if fail_on_gap:
                        raise RuntimeError(f"{venue} live L2 sequence gap detected for {symbol}")
                    if reset_on_gap:
                        tracker.reset()
                        if snapshot_url is not None:
                            snapshot_payload = await _maybe_await((snapshot_fetcher or _fetch_json)(snapshot_url))
                            snapshot_batch = normalize_live_l2_message(
                                venue,
                                snapshot_payload,
                                symbol=symbol,
                                local_timestamp_ms=clock_ms(),
                            )
                            tracker.apply(snapshot_batch)
                            rows_written += _write_rows(writer, snapshot_batch.rows)
                        elif subscribe_message is not None:
                            await websocket.send(subscribe_message)
                        messages_seen += 1
                        continue
                messages_seen += 1
                rows_written += _write_rows(writer, batch.rows)

    return LiveL2CaptureSummary(
        venue=venue,
        symbol=symbol,
        output_path=output,
        messages_seen=messages_seen,
        rows_written=rows_written,
        sequence_gaps=sequence_gaps,
    )


def _normalize_binance_message(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    local_timestamp_ms: int | None,
) -> LiveL2Batch:
    if "lastUpdateId" in payload:
        update_id = _optional_int(payload.get("lastUpdateId"))
        exchange_timestamp = _optional_timestamp_ms(payload.get("E")) or local_timestamp_ms or 0
        rows = _rows_from_book_levels(
            venue="binance",
            symbol=symbol,
            event_type="snapshot",
            exchange_timestamp=exchange_timestamp,
            local_timestamp=local_timestamp_ms,
            bids=payload.get("bids", ()),
            asks=payload.get("asks", ()),
            update_id=update_id,
        )
        return LiveL2Batch("binance", symbol, "snapshot", tuple(rows), update_id=update_id)

    update_id = _optional_int(payload.get("u"))
    first_update_id = _optional_int(payload.get("U"))
    exchange_timestamp = _optional_timestamp_ms(payload.get("E")) or local_timestamp_ms or 0
    rows = _rows_from_book_levels(
        venue="binance",
        symbol=str(payload.get("s") or symbol),
        event_type="delta",
        exchange_timestamp=exchange_timestamp,
        local_timestamp=local_timestamp_ms,
        bids=payload.get("b", ()),
        asks=payload.get("a", ()),
        update_id=update_id,
    )
    return LiveL2Batch("binance", symbol, "delta", tuple(rows), update_id=update_id, first_update_id=first_update_id)


def _normalize_okx_message(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    local_timestamp_ms: int | None,
) -> LiveL2Batch:
    action = str(payload.get("action") or "snapshot").lower()
    event_type = "delta" if action in {"update", "delta"} else "snapshot"
    raw_arg = payload.get("arg")
    arg: Mapping[str, Any] = raw_arg if isinstance(raw_arg, Mapping) else {}
    resolved_symbol = str(arg.get("instId") or symbol)
    data = payload.get("data") or []
    if isinstance(data, Mapping):
        data = [data]
    rows: list[NormalizedL2Row] = []
    sequence = None
    prev_sequence = None
    for book in data:
        if not isinstance(book, Mapping):
            continue
        sequence = _optional_int(book.get("seqId"))
        prev_sequence = _optional_int(book.get("prevSeqId"))
        exchange_timestamp = _optional_timestamp_ms(book.get("ts")) or local_timestamp_ms or 0
        rows.extend(
            _rows_from_book_levels(
                venue="okx",
                symbol=resolved_symbol,
                event_type=event_type,
                exchange_timestamp=exchange_timestamp,
                local_timestamp=local_timestamp_ms,
                bids=book.get("bids", ()),
                asks=book.get("asks", ()),
                sequence=sequence,
            )
        )
    return LiveL2Batch("okx", resolved_symbol, event_type, tuple(rows), sequence=sequence, prev_sequence=prev_sequence)


def _normalize_bybit_message(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    local_timestamp_ms: int | None,
) -> LiveL2Batch:
    raw_data = payload.get("data")
    data: Mapping[str, Any] = raw_data if isinstance(raw_data, Mapping) else payload
    event_type = str(payload.get("type") or data.get("type") or "delta").lower()
    if event_type not in {"snapshot", "delta"}:
        event_type = "delta"
    resolved_symbol = str(data.get("s") or data.get("symbol") or symbol)
    sequence = _optional_int(data.get("seq") or payload.get("seq"))
    update_id = _optional_int(data.get("u") or payload.get("u"))
    exchange_timestamp = (
        _optional_timestamp_ms(data.get("cts")) or _optional_timestamp_ms(payload.get("ts")) or local_timestamp_ms or 0
    )
    rows = _rows_from_book_levels(
        venue="bybit",
        symbol=resolved_symbol,
        event_type=event_type,
        exchange_timestamp=exchange_timestamp,
        local_timestamp=local_timestamp_ms,
        bids=data.get("b", data.get("bids", ())),
        asks=data.get("a", data.get("asks", ())),
        sequence=sequence,
        update_id=update_id,
    )
    return LiveL2Batch("bybit", resolved_symbol, event_type, tuple(rows), sequence=sequence, update_id=update_id)


def _normalize_coinbase_message(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    local_timestamp_ms: int | None,
) -> LiveL2Batch:
    sequence = _optional_int(payload.get("sequence_num") or payload.get("sequence"))
    rows: list[NormalizedL2Row] = []
    event_type = "delta"
    if payload.get("type") == "snapshot":
        event_type = "snapshot"
        rows.extend(
            _rows_from_book_levels(
                venue="coinbase",
                symbol=str(payload.get("product_id") or symbol),
                event_type=event_type,
                exchange_timestamp=local_timestamp_ms or 0,
                local_timestamp=local_timestamp_ms,
                bids=payload.get("bids", ()),
                asks=payload.get("asks", ()),
                sequence=sequence,
            )
        )
    elif payload.get("type") == "l2update":
        event_type = "delta"
        exchange_timestamp = _optional_timestamp_ms(payload.get("time")) or local_timestamp_ms or 0
        for change in payload.get("changes", ()):
            side, price, size = _coinbase_change_parts(change)
            if side is None:
                continue
            rows.append(
                NormalizedL2Row(
                    event_type=event_type,
                    exchange_timestamp=exchange_timestamp,
                    local_timestamp=local_timestamp_ms,
                    side=side,
                    price=price,
                    size=size,
                    sequence=sequence,
                    venue="coinbase",
                    symbol=str(payload.get("product_id") or symbol),
                )
            )
    else:
        for event in payload.get("events", ()):
            if not isinstance(event, Mapping):
                continue
            raw_type = str(event.get("type") or "update").lower()
            event_type = "snapshot" if raw_type == "snapshot" else "delta"
            resolved_symbol = str(event.get("product_id") or symbol)
            for update in event.get("updates", ()):
                if not isinstance(update, Mapping):
                    continue
                side = _normalize_side(update.get("side"))
                if side is None:
                    continue
                price_level = update.get("price_level")
                new_quantity = update.get("new_quantity")
                if price_level is None or new_quantity is None:
                    continue
                rows.append(
                    NormalizedL2Row(
                        event_type=event_type,
                        exchange_timestamp=_optional_timestamp_ms(update.get("event_time")) or local_timestamp_ms or 0,
                        local_timestamp=local_timestamp_ms,
                        side=side,
                        price=float(price_level),
                        size=float(new_quantity),
                        sequence=sequence,
                        venue="coinbase",
                        symbol=resolved_symbol,
                    )
                )
    return LiveL2Batch("coinbase", symbol, event_type, tuple(rows), sequence=sequence)


def _rows_from_book_levels(
    *,
    venue: str,
    symbol: str,
    event_type: str,
    exchange_timestamp: int,
    local_timestamp: int | None,
    bids: Any,
    asks: Any,
    sequence: int | None = None,
    update_id: int | None = None,
) -> list[NormalizedL2Row]:
    rows: list[NormalizedL2Row] = []
    for side, levels in (("bid", bids), ("ask", asks)):
        for level in levels or ():
            price, size = _level_price_size(level)
            rows.append(
                NormalizedL2Row(
                    event_type=event_type,
                    exchange_timestamp=exchange_timestamp,
                    local_timestamp=local_timestamp,
                    side=side,
                    price=price,
                    size=size,
                    sequence=sequence,
                    update_id=update_id,
                    venue=venue,
                    symbol=symbol,
                )
            )
    return rows


def _write_rows(writer: csv.DictWriter, rows: tuple[NormalizedL2Row, ...]) -> int:
    for row in rows:
        writer.writerow({column: getattr(row, column) for column in NORMALIZED_L2_COLUMNS})
    return len(rows)


def _decode_message(message: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        return message
    if isinstance(message, bytes):
        message = message.decode()
    payload = json.loads(message)
    if not isinstance(payload, Mapping):
        raise ValueError("live L2 message must decode to a JSON object")
    return payload


def _level_price_size(level: Any) -> tuple[float, float]:
    if isinstance(level, Mapping):
        price = level.get("price") or level.get("price_level") or level.get("px")
        size = level.get("size") or level.get("qty") or level.get("new_quantity") or level.get("sz")
    else:
        price, size = level[0], level[1]
    if price is None or size is None:
        raise ValueError("book level missing price or size")
    return float(price), float(size)


def _coinbase_change_parts(change: Any) -> tuple[str | None, float, float]:
    side_value, price, size = change[0], change[1], change[2]
    side = _normalize_side(side_value)
    return side, float(price), float(size)


def _normalize_side(value: Any) -> str | None:
    text = str(value).lower()
    if text in {"bid", "bids", "buy"}:
        return "bid"
    if text in {"ask", "asks", "sell"}:
        return "ask"
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _optional_timestamp_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return int(numeric * 1000) if numeric < 10_000_000_000 else int(numeric)
    text = str(value).strip()
    parsed_int = _optional_int(text)
    if parsed_int is not None:
        return parsed_int * 1000 if parsed_int < 10_000_000_000 else parsed_int
    try:
        normalized = text.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except ValueError:
        return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fetch_json(url: str) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, Mapping):
        raise ValueError(f"snapshot URL did not return a JSON object: {url}")
    return payload


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _default_websocket_factory(url: str):
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("live L2 capture requires websockets; install .[research]") from exc
    return websockets.connect(url)
