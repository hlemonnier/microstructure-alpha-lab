from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


CANONICAL_L2_FIELDS = [
    "event_type",
    "exchange_timestamp",
    "local_timestamp",
    "side",
    "price",
    "size",
    "sequence",
    "update_id",
]

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "event_type": (
        "event_type",
        "type",
        "message_type",
        "action",
        "snapshot_delta",
        "is_snapshot",
    ),
    "exchange_timestamp": (
        "exchange_timestamp",
        "exchange_timestamp_ms",
        "exchange_time",
        "exchange_time_ms",
        "timestamp",
        "ts",
        "cts",
        "origin_time",
        "time",
    ),
    "local_timestamp": (
        "local_timestamp",
        "local_timestamp_ms",
        "receive_timestamp",
        "received_timestamp",
        "received_time",
        "local_time",
    ),
    "side": ("side", "book_side"),
    "price": ("price", "px", "level_price"),
    "size": ("size", "sz", "qty", "quantity", "amount", "volume", "level_size"),
    "sequence": ("sequence", "seqid", "seqId", "seq", "cross_sequence"),
    "update_id": ("update_id", "u", "last_update_id", "lastUpdateId", "first_update_id", "U"),
    "symbol": ("symbol", "instrument", "instrument_id", "inst_id", "instId", "instid", "s"),
    "venue": ("venue", "exchange"),
}

REQUIRED_REPLAY_FIELDS = [
    "event_type",
    "exchange_timestamp",
    "side",
    "price",
    "size",
]


@dataclass(frozen=True)
class SourceDescriptor:
    source_id: str
    name: str
    priority: int
    role: str
    free_historical_l2: bool
    live_l2: bool
    replay_grade: str
    canonical_urls: tuple[str, ...]
    data_types: tuple[str, ...]
    required_fields: tuple[str, ...] = tuple(REQUIRED_REPLAY_FIELDS)
    optional_fields: tuple[str, ...] = ("local_timestamp", "sequence", "update_id")
    limitations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_sequence_validation(self) -> bool:
        return self.replay_grade in {"deterministic_if_sequence_valid", "live_replay_if_stitched"}


SOURCE_DESCRIPTORS: tuple[SourceDescriptor, ...] = (
    SourceDescriptor(
        source_id="okx",
        name="OKX Historical Market Data",
        priority=1,
        role="primary historical crypto L2 source for the 60-90 day study",
        free_historical_l2=True,
        live_l2=True,
        replay_grade="schema_must_be_verified",
        canonical_urls=(
            "https://www.okx.com/en-gb/historical-data",
            "https://www.okx.com/docs-v5/en/#order-book-trading-market-data-ws-order-book-channel",
        ),
        data_types=("high-resolution L2 order book", "tick trades", "funding rates"),
        limitations=(
            "Do not claim deterministic replay until downloaded rows expose snapshot/delta semantics and sequence/update fields.",
        ),
    ),
    SourceDescriptor(
        source_id="bybit",
        name="Bybit Historical Data And V5 Orderbook",
        priority=2,
        role="second historical venue and live replay robustness check",
        free_historical_l2=True,
        live_l2=True,
        replay_grade="deterministic_if_sequence_valid",
        canonical_urls=(
            "https://www.bybit.com/derivatives/en/history-data",
            "https://bybit-exchange.github.io/docs/v5/market/orderbook",
            "https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook",
        ),
        data_types=("orderBook files", "REST snapshots", "WebSocket snapshot/delta updates"),
        optional_fields=("local_timestamp", "sequence", "update_id", "cts"),
        limitations=("Historical file schema must be checked separately from the documented REST/WebSocket shape.",),
    ),
    SourceDescriptor(
        source_id="binance",
        name="Binance Data Vision And Live Diff Depth",
        priority=3,
        role="supplemental historical top-depth data and live replay collection",
        free_historical_l2=True,
        live_l2=True,
        replay_grade="live_replay_if_stitched",
        canonical_urls=(
            "https://github.com/binance/binance-public-data",
            "https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md",
        ),
        data_types=("bookTicker", "bookDepth snapshots", "aggTrades", "trades", "diff-depth WebSocket"),
        limitations=(
            "Futures bookDepth files are not assumed to be exchange-native deterministic L2 replay streams.",
            "Replay-grade Binance data requires REST snapshot plus diff-depth WebSocket collection with update-id gap checks.",
        ),
    ),
    SourceDescriptor(
        source_id="coinbase",
        name="Coinbase Advanced Trade / Exchange WebSocket",
        priority=4,
        role="live collection only",
        free_historical_l2=False,
        live_l2=True,
        replay_grade="live_replay_if_stitched",
        canonical_urls=(
            "https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-overview",
            "https://docs.cdp.coinbase.com/exchange/websocket-feed/channels",
        ),
        data_types=("level2 WebSocket", "heartbeat sequence checks"),
        limitations=("Not treated as a free 90-day historical L2 source.",),
    ),
    SourceDescriptor(
        source_id="tardis",
        name="Tardis.dev CSV Samples",
        priority=5,
        role="sample and validation corpus",
        free_historical_l2=True,
        live_l2=False,
        replay_grade="deterministic_if_sequence_valid",
        canonical_urls=(
            "https://docs.tardis.dev/downloadable-csv-files/overview",
            "https://docs.tardis.dev/historical-data-details/bybit-spot",
            "https://docs.tardis.dev/historical-data-details/okex",
        ),
        data_types=("incremental_book_L2", "book_snapshot_25", "book_snapshot_5"),
        limitations=("First-day-of-month samples are free; arbitrary historical coverage generally needs API access.",),
    ),
    SourceDescriptor(
        source_id="crypto_lake",
        name="Crypto Lake Free Data",
        priority=6,
        role="sample and validation corpus",
        free_historical_l2=True,
        live_l2=False,
        replay_grade="schema_must_be_verified",
        canonical_urls=("https://crypto-lake.com/free-data/",),
        data_types=("L2 order book", "trades", "candles"),
        limitations=(
            "Free coverage is useful for validation but source schema must be inspected before replay claims.",
        ),
    ),
    SourceDescriptor(
        source_id="fi2010",
        name="FI-2010 Limit Order Book Benchmark",
        priority=7,
        role="equity LOB tensor sanity benchmark only",
        free_historical_l2=True,
        live_l2=False,
        replay_grade="fixed_tensor_benchmark_not_crypto_replay",
        canonical_urls=("https://arxiv.org/abs/1705.03233",),
        data_types=("10-level equity LOB tensors", "mid-price direction labels"),
        required_fields=(),
        limitations=(
            "Nasdaq Nordic equities, not crypto.",
            "Use for model sanity checks, not for crypto tradability conclusions.",
        ),
    ),
)

SOURCE_BY_ID = {descriptor.source_id: descriptor for descriptor in SOURCE_DESCRIPTORS}


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class L2RowValidation:
    source_id: str | None
    row_index: int | None
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class L2FileValidation:
    source_id: str | None
    path: Path
    rows_checked: int
    missing_columns: tuple[str, ...]
    row_errors: tuple[L2RowValidation, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_columns and not self.row_errors

    @property
    def passed(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class NormalizedL2Row:
    event_type: str
    exchange_timestamp: int
    side: str
    price: float
    size: float
    local_timestamp: int | None = None
    sequence: int | None = None
    update_id: int | None = None
    venue: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class FI2010Snapshot:
    row_index: int
    asks: list[tuple[float, float]]
    bids: list[tuple[float, float]]
    labels: tuple[int, ...] = ()


def list_sources() -> list[SourceDescriptor]:
    return sorted(SOURCE_DESCRIPTORS, key=lambda source: source.priority)


def get_historical_l2_sources() -> list[SourceDescriptor]:
    return list_sources()


def get_source(source_id: str) -> SourceDescriptor:
    try:
        return SOURCE_BY_ID[source_id]
    except KeyError as exc:
        valid = ", ".join(sorted(SOURCE_BY_ID))
        raise ValueError(f"unknown source_id {source_id!r}; expected one of: {valid}") from exc


def format_historical_l2_sources_csv(sources: Iterable[SourceDescriptor]) -> str:
    fields = [
        "priority",
        "source_id",
        "name",
        "role",
        "free_historical_l2",
        "live_l2",
        "replay_grade",
        "canonical_urls",
        "limitations",
    ]
    lines = [",".join(fields)]
    for source in sources:
        lines.append(
            _csv_line(
                [
                    str(source.priority),
                    source.source_id,
                    source.name,
                    source.role,
                    str(int(source.free_historical_l2)),
                    str(int(source.live_l2)),
                    source.replay_grade,
                    " ".join(source.canonical_urls),
                    " ".join(source.limitations),
                ]
            )
        )
    return "\n".join(lines)


def validate_l2_columns(columns: Iterable[str], *, source_id: str | None = None) -> tuple[str, ...]:
    descriptor = get_source(source_id) if source_id else None
    required_fields = descriptor.required_fields if descriptor else tuple(REQUIRED_REPLAY_FIELDS)
    normalized_columns = {column.strip() for column in columns}
    missing = [
        field
        for field in required_fields
        if _find_alias(normalized_columns, FIELD_ALIASES.get(field, (field,))) is None
    ]
    return tuple(missing)


def validate_l2_row(
    row: Mapping[str, object],
    *,
    source_id: str | None = None,
    row_index: int | None = None,
    require_sequence: bool | None = None,
) -> L2RowValidation:
    descriptor = get_source(source_id) if source_id else None
    should_require_sequence = (
        descriptor.needs_sequence_validation if require_sequence is None and descriptor else bool(require_sequence)
    )
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    event_type = _event_type(row)
    if event_type not in {"snapshot", "delta"}:
        errors.append(ValidationIssue("event_type", "row must identify snapshot or delta"))

    exchange_timestamp = _optional_int_field(row, "exchange_timestamp")
    local_timestamp = _optional_int_field(row, "local_timestamp")
    if exchange_timestamp is None and local_timestamp is None:
        errors.append(ValidationIssue("exchange_timestamp", "row needs an exchange or local timestamp"))

    side = _side(row)
    if side not in {"bid", "ask"}:
        errors.append(ValidationIssue("side", "side must be bid/ask or buy/sell"))

    price = _optional_float_field(row, "price")
    if price is None or price <= 0.0:
        errors.append(ValidationIssue("price", "price must be a positive number"))

    size = _optional_float_field(row, "size")
    if size is None or size < 0.0:
        errors.append(ValidationIssue("size", "size must be a non-negative number"))

    sequence = _optional_int_field(row, "sequence")
    update_id = _optional_int_field(row, "update_id")
    if should_require_sequence and sequence is None and update_id is None:
        errors.append(ValidationIssue("sequence", "replay-grade source requires sequence or update_id"))
    elif sequence is None and update_id is None:
        warnings.append(
            ValidationIssue(
                "sequence", "no sequence/update_id available; deterministic replay claim is gated", "warning"
            )
        )

    return L2RowValidation(
        source_id=source_id,
        row_index=row_index,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def normalize_l2_row(
    row: Mapping[str, object],
    *,
    source_id: str | None = None,
    require_sequence: bool | None = None,
) -> NormalizedL2Row:
    validation = validate_l2_row(row, source_id=source_id, require_sequence=require_sequence)
    if validation.errors:
        messages = "; ".join(f"{issue.field}: {issue.message}" for issue in validation.errors)
        raise ValueError(f"cannot normalize invalid L2 row: {messages}")
    exchange_timestamp = _optional_int_field(row, "exchange_timestamp")
    local_timestamp = _optional_int_field(row, "local_timestamp")
    return NormalizedL2Row(
        event_type=_event_type(row),
        exchange_timestamp=exchange_timestamp if exchange_timestamp is not None else local_timestamp or 0,
        local_timestamp=local_timestamp,
        side=_side(row),
        price=_optional_float_field(row, "price") or 0.0,
        size=_optional_float_field(row, "size") or 0.0,
        sequence=_optional_int_field(row, "sequence"),
        update_id=_optional_int_field(row, "update_id"),
        venue=_optional_text_field(row, "venue") or source_id,
        symbol=_optional_text_field(row, "symbol"),
    )


def validate_l2_csv(
    path: Path | str,
    *,
    source_id: str | None = None,
    max_rows: int = 1000,
    require_sequence: bool | None = None,
) -> L2FileValidation:
    path = Path(path)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_columns = validate_l2_columns(columns, source_id=source_id)
        row_errors: list[L2RowValidation] = []
        rows_checked = 0
        for index, row in enumerate(reader, start=1):
            if rows_checked >= max_rows:
                break
            rows_checked += 1
            validation = validate_l2_row(
                row,
                source_id=source_id,
                row_index=index,
                require_sequence=require_sequence,
            )
            if not validation.ok:
                row_errors.append(validation)

    return L2FileValidation(
        source_id=source_id,
        path=path,
        rows_checked=rows_checked,
        missing_columns=missing_columns,
        row_errors=tuple(row_errors),
    )


def validate_l2_csv_schema(
    path: Path | str,
    *,
    source_format: str,
    max_rows: int = 1000,
) -> L2FileValidation:
    source_id = _source_format_to_id(source_format)
    return validate_l2_csv(
        path,
        source_id=source_id,
        max_rows=max_rows,
        require_sequence=False,
    )


def iter_okx_l2_csv(path: Path | str, *, default_symbol: str) -> Iterable[NormalizedL2Row]:
    return _iter_normalized_l2_csv(path, source_id="okx", default_symbol=default_symbol)


def iter_bybit_l2_csv(path: Path | str, *, default_symbol: str) -> Iterable[NormalizedL2Row]:
    return _iter_normalized_l2_csv(path, source_id="bybit", default_symbol=default_symbol)


def iter_bybit_orderbook_data_zip(path: Path | str, *, default_symbol: str | None = None) -> Iterable[NormalizedL2Row]:
    """Read Bybit historical `*_ob*.data.zip` order-book archives.

    Bybit's history-data page serves ZIP files whose inner `.data` file contains
    newline-delimited JSON messages shaped like the V5 orderbook stream:
    `type`, `ts`, `data.s`, `data.seq`, `data.u`, `data.b`, and `data.a`.
    """
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as archive:
        data_names = [name for name in archive.namelist() if name.endswith(".data") or name.endswith(".json")]
        if not data_names:
            raise ValueError(f"No Bybit .data file found in {archive_path}")
        with archive.open(data_names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            for line_index, line in enumerate(text, start=1):
                payload_text = line.strip()
                if not payload_text:
                    continue
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid Bybit orderbook JSON at line {line_index}: {exc}") from exc
                yield from _bybit_orderbook_payload_to_rows(payload, default_symbol=default_symbol)


def validate_bybit_orderbook_data_zip(
    path: Path | str,
    *,
    max_rows: int = 1000,
    require_sequence: bool | None = True,
) -> L2FileValidation:
    path = Path(path)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    row_errors: list[L2RowValidation] = []
    rows_checked = 0
    for row in iter_bybit_orderbook_data_zip(path):
        if rows_checked >= max_rows:
            break
        rows_checked += 1
        validation = validate_l2_row(
            asdict(row),
            source_id="bybit",
            row_index=rows_checked,
            require_sequence=require_sequence,
        )
        if not validation.ok:
            row_errors.append(validation)
    return L2FileValidation(
        source_id="bybit",
        path=path,
        rows_checked=rows_checked,
        missing_columns=() if rows_checked else tuple(REQUIRED_REPLAY_FIELDS),
        row_errors=tuple(row_errors),
    )


def iter_tardis_incremental_book_l2_csv(path: Path | str) -> Iterable[NormalizedL2Row]:
    return _iter_normalized_l2_csv(path, source_id="tardis", default_symbol=None)


def iter_crypto_lake_book_delta_v2_csv(
    path: Path | str,
    *,
    default_symbol: str,
) -> Iterable[NormalizedL2Row]:
    return _iter_normalized_l2_csv(path, source_id="crypto_lake", default_symbol=default_symbol)


def write_normalized_l2_csv(events: Iterable[NormalizedL2Row], path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(NormalizedL2Row.__dataclass_fields__.keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))
    return output_path


def write_normalized_l2_parquet(events: Iterable[NormalizedL2Row], path: Path | str) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("write_normalized_l2_parquet requires pyarrow; install .[research]") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(event) for event in events])
    pq.write_table(table, output_path)
    return output_path


def load_fi2010_snapshots(
    path: Path | str,
    *,
    levels: int = 10,
    max_rows: int | None = None,
) -> list[FI2010Snapshot]:
    if levels <= 0:
        raise ValueError("levels must be positive")
    feature_count = levels * 4
    snapshots: list[FI2010Snapshot] = []
    with Path(path).open() as handle:
        for row_index, line in enumerate(handle):
            if max_rows is not None and len(snapshots) >= max_rows:
                break
            raw = line.strip()
            if not raw:
                continue
            values = _parse_numeric_line(raw)
            if len(values) < feature_count:
                raise ValueError(f"row {row_index + 1} has {len(values)} values; expected at least {feature_count}")
            asks: list[tuple[float, float]] = []
            bids: list[tuple[float, float]] = []
            for level in range(levels):
                base = level * 4
                asks.append((values[base], values[base + 1]))
                bids.append((values[base + 2], values[base + 3]))
            snapshots.append(
                FI2010Snapshot(
                    row_index=row_index,
                    asks=asks,
                    bids=bids,
                    labels=tuple(int(value) for value in values[feature_count:]),
                )
            )
    return snapshots


def fi2010_snapshots_to_events(
    snapshots: Iterable[FI2010Snapshot],
    *,
    symbol: str = "FI2010",
) -> Iterable[NormalizedL2Row]:
    for snapshot in snapshots:
        for level, (price, size) in enumerate(snapshot.asks, start=1):
            yield NormalizedL2Row(
                event_type="snapshot",
                exchange_timestamp=snapshot.row_index,
                side="ask",
                price=price,
                size=size,
                sequence=snapshot.row_index,
                venue="fi2010",
                symbol=symbol,
            )
        for level, (price, size) in enumerate(snapshot.bids, start=1):
            yield NormalizedL2Row(
                event_type="snapshot",
                exchange_timestamp=snapshot.row_index,
                side="bid",
                price=price,
                size=size,
                sequence=snapshot.row_index,
                venue="fi2010",
                symbol=symbol,
            )


def format_schema_validation(result: L2FileValidation) -> str:
    return json.dumps(
        {
            "source_id": result.source_id,
            "path": str(result.path),
            "rows_checked": result.rows_checked,
            "missing_columns": list(result.missing_columns),
            "row_errors": [
                {
                    "row_index": row.row_index,
                    "errors": [asdict(issue) for issue in row.errors],
                    "warnings": [asdict(issue) for issue in row.warnings],
                }
                for row in result.row_errors
            ],
            "passed": result.ok,
        },
        sort_keys=True,
    )


def source_priority_markdown() -> str:
    lines = [
        "| Priority | Source | Role | Replay Grade | Free Historical L2 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in list_sources():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(source.priority),
                    source.name,
                    source.role,
                    source.replay_grade,
                    "yes" if source.free_historical_l2 else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _iter_normalized_l2_csv(
    path: Path | str,
    *,
    source_id: str,
    default_symbol: str | None,
) -> Iterable[NormalizedL2Row]:
    with _open_text(Path(path)) as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            row_with_defaults = dict(row)
            if default_symbol and _value(row_with_defaults, "symbol") is None:
                row_with_defaults["symbol"] = default_symbol
            try:
                yield normalize_l2_row(
                    row_with_defaults,
                    source_id=source_id,
                    require_sequence=False,
                )
            except ValueError as exc:
                raise ValueError(f"row {row_index}: {exc}") from exc


def _source_format_to_id(source_format: str) -> str:
    return {
        "okx": "okx",
        "bybit": "bybit",
        "tardis": "tardis",
        "tardis_incremental_book_l2": "tardis",
        "crypto_lake": "crypto_lake",
        "crypto_lake_book_delta_v2": "crypto_lake",
    }.get(source_format, source_format)


@contextmanager
def _open_text(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            yield handle
        return
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"no CSV member found in ZIP archive: {path}")
            with archive.open(names[0]) as raw_handle:
                with io.TextIOWrapper(raw_handle, newline="") as text_handle:
                    yield text_handle
        return
    with path.open(newline="") as handle:
        yield handle


def _find_alias(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    column_set = set(columns)
    for alias in aliases:
        if alias in column_set:
            return alias
    lowered = {column.lower(): column for column in column_set}
    for alias in aliases:
        match = lowered.get(alias.lower())
        if match is not None:
            return match
    return None


def _value(row: Mapping[str, object], canonical_field: str) -> object | None:
    aliases = FIELD_ALIASES.get(canonical_field, (canonical_field,))
    for alias in aliases:
        if alias in row and row[alias] not in {"", None}:
            return row[alias]
    lowered = {str(key).lower(): key for key in row}
    for alias in aliases:
        key = lowered.get(alias.lower())
        if key is not None and row[key] not in {"", None}:
            return row[key]
    return None


def _optional_text_field(row: Mapping[str, object], canonical_field: str) -> str | None:
    value = _value(row, canonical_field)
    return str(value).strip() if value is not None else None


def _optional_float_field(row: Mapping[str, object], canonical_field: str) -> float | None:
    value = _value(row, canonical_field)
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _optional_int_field(row: Mapping[str, object], canonical_field: str) -> int | None:
    value = _value(row, canonical_field)
    if value is None:
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            if "timestamp" in canonical_field or canonical_field.endswith("_time"):
                return _parse_timestamp_ms(text)
            return None


def _event_type(row: Mapping[str, object]) -> str:
    value = _value(row, "event_type")
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"snapshot", "snap", "partial"}:
        return "snapshot"
    if text in {"delta", "update", "change", "incremental"}:
        return "delta"
    if text in {"true", "1", "yes"}:
        return "snapshot"
    if text in {"false", "0", "no"}:
        return "delta"
    return text


def _side(row: Mapping[str, object]) -> str:
    value = _value(row, "side")
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"bid", "bids", "buy", "b"}:
        return "bid"
    if text in {"ask", "asks", "sell", "a"}:
        return "ask"
    return text


def _bybit_orderbook_payload_to_rows(
    payload: Mapping[str, object],
    *,
    default_symbol: str | None,
) -> Iterable[NormalizedL2Row]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Bybit orderbook payload missing data object")
    event_type = _event_type(payload)
    if event_type not in {"snapshot", "delta"}:
        raise ValueError("Bybit orderbook payload must identify snapshot or delta")
    exchange_timestamp = _coalesce_int(data.get("cts"), payload.get("ts"))
    local_timestamp = _coalesce_int(payload.get("ts"))
    sequence = _coalesce_int(data.get("seq"), payload.get("seq"))
    update_id = _coalesce_int(data.get("u"), payload.get("u"))
    symbol = str(data.get("s") or payload.get("symbol") or default_symbol or "")

    for side, levels in (("bid", data.get("b")), ("ask", data.get("a"))):
        if levels is None:
            continue
        if not isinstance(levels, list):
            raise ValueError(f"Bybit orderbook side {side} must be a list")
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                raise ValueError(f"Bybit orderbook level must contain price and size: {level!r}")
            yield NormalizedL2Row(
                event_type=event_type,
                exchange_timestamp=exchange_timestamp or local_timestamp or 0,
                local_timestamp=local_timestamp,
                side=side,
                price=float(level[0]),
                size=float(level[1]),
                sequence=sequence,
                update_id=update_id,
                venue="bybit",
                symbol=symbol or None,
            )


def _coalesce_int(*values: object) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(str(value))
        except ValueError:
            try:
                return int(float(str(value)))
            except ValueError:
                continue
    return None


def _parse_timestamp_ms(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return None


def _parse_numeric_line(raw: str) -> list[float]:
    delimiter = "," if "," in raw else None
    parts = raw.split(delimiter) if delimiter else raw.split()
    return [float(part) for part in parts if part]


def _csv_line(values: list[str]) -> str:
    import io

    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(values)
    return handle.getvalue().strip("\r\n")
