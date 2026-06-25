from __future__ import annotations

import csv
import io
import zipfile
from bisect import bisect_left
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from lob_forge.memory_guard import assert_feature_build_budget


BOOK_TICKER_COLUMNS = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]

AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


BOOK_DEPTH_COLUMNS = [
    "timestamp",
    "percentage",
    "depth",
    "notional",
]


DEPTH_BANDS = [1, 2, 5]

DEPTH_FEATURE_COLUMNS = [
    "depth_snapshot_time",
    "depth_snapshot_age_ms",
]
for _band in DEPTH_BANDS:
    DEPTH_FEATURE_COLUMNS.extend(
        [
            f"bid_depth_{_band}pct",
            f"ask_depth_{_band}pct",
            f"bid_notional_{_band}pct",
            f"ask_notional_{_band}pct",
            f"depth_imbalance_{_band}pct",
            f"notional_imbalance_{_band}pct",
        ]
    )


QUOTE_CONTEXT_COLUMNS = [
    "quote_ofi",
    "quote_ofi_normalized",
    "quote_ofi_5",
    "quote_ofi_5_normalized",
    "mid_return_1",
    "mid_return_5",
    "realized_volatility_5",
    "spread_mean_5",
    "top_imbalance_mean_5",
]


EXECUTION_PATH_COLUMNS = [
    "horizon_min_ask",
    "horizon_max_bid",
    "maker_long_fillable",
    "maker_short_fillable",
    "maker_long_fill_event_time",
    "maker_short_fill_event_time",
]


FEATURE_COLUMNS = [
    "bucket_start_ms",
    "decision_time",
    "feature_cutoff_time",
    "event_time",
    "quote_event_time",
    "local_receive_time",
    "update_id",
    "bid",
    "ask",
    "bid_qty",
    "ask_qty",
    "mid",
    "spread",
    "relative_spread",
    "top_imbalance",
    "microprice",
    "microprice_deviation",
    "quote_updates_in_bucket",
    *QUOTE_CONTEXT_COLUMNS,
    "trade_count",
    "buy_qty",
    "sell_qty",
    "trade_qty",
    "buy_notional",
    "sell_notional",
    "trade_notional",
    "trade_imbalance",
    "large_trade_count",
    *DEPTH_FEATURE_COLUMNS,
    "execution_latency_ms",
    "holding_horizon_ms",
    "entry_target_time",
    "future_target_time",
    "entry_lag_ms",
    "future_lag_ms",
    "entry_event_time",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "future_event_time",
    "future_bid",
    "future_ask",
    "future_mid",
    "future_spread",
    *EXECUTION_PATH_COLUMNS,
    "delta_mid",
    "label",
]


@dataclass
class QuoteBucket:
    bucket_start_ms: int
    event_time: int
    update_id: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    update_count: int = 1
    decision_time_ms: int | None = None
    local_receive_time: int | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def decision_time(self) -> int:
        return self.decision_time_ms if self.decision_time_ms is not None else self.event_time


@dataclass
class TradeBucket:
    trade_count: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    large_trade_count: int = 0

    @property
    def trade_qty(self) -> float:
        return self.buy_qty + self.sell_qty

    @property
    def trade_notional(self) -> float:
        return self.buy_notional + self.sell_notional

    @property
    def trade_imbalance(self) -> float:
        total = self.trade_qty
        return (self.buy_qty - self.sell_qty) / total if total else 0.0


@dataclass(frozen=True)
class TradeEvent:
    timestamp_ms: int
    bucket_start_ms: int
    quantity: float
    notional: float
    is_sell_initiated: bool
    is_large: bool


@dataclass(frozen=True)
class TradeBucketIndex:
    bucket_ms: int
    events_by_bucket: dict[int, list[TradeEvent]]

    def aggregate_until(self, *, bucket_start_ms: int, decision_time_ms: int) -> TradeBucket:
        bucket = TradeBucket()
        for event in self.events_by_bucket.get(bucket_start_ms, []):
            if event.timestamp_ms > decision_time_ms:
                break
            bucket.trade_count += 1
            if event.is_sell_initiated:
                bucket.sell_qty += event.quantity
                bucket.sell_notional += event.notional
            else:
                bucket.buy_qty += event.quantity
                bucket.buy_notional += event.notional
            if event.is_large:
                bucket.large_trade_count += 1
        return bucket


@dataclass
class DepthSnapshot:
    snapshot_time_ms: int
    bid_depth_by_pct: dict[int, float] = field(default_factory=dict)
    ask_depth_by_pct: dict[int, float] = field(default_factory=dict)
    bid_notional_by_pct: dict[int, float] = field(default_factory=dict)
    ask_notional_by_pct: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteContext:
    quote_ofi: float = 0.0
    quote_ofi_normalized: float = 0.0
    quote_ofi_5: float = 0.0
    quote_ofi_5_normalized: float = 0.0
    mid_return_1: float = 0.0
    mid_return_5: float = 0.0
    realized_volatility_5: float = 0.0
    spread_mean_5: float = 0.0
    top_imbalance_mean_5: float = 0.0


@dataclass(frozen=True)
class ExecutionPath:
    horizon_min_ask: float
    horizon_max_bid: float
    maker_long_fillable: bool
    maker_short_fillable: bool
    maker_long_fill_event_time: int | None = None
    maker_short_fill_event_time: int | None = None


@dataclass(frozen=True)
class FeatureSummary:
    rows: int
    label_counts: dict[str, int]
    rows_with_trades: int
    rows_with_depth: int
    spread_min: float
    spread_max: float
    delta_mid_min: float
    delta_mid_max: float


def build_quote_trade_dataset(
    *,
    book_ticker_zip: Path | str,
    output_csv: Path | str,
    agg_trades_zip: Path | str | None = None,
    book_depth_zip: Path | str | None = None,
    bucket_ms: int = 1000,
    horizon_ms: int = 1000,
    execution_latency_ms: int = 0,
    threshold: str = "half_spread",
    min_tick: float = 0.0,
    large_trade_notional: float = 10_000.0,
    max_quote_buckets: int | None = None,
    max_feature_build_memory_gb: float = 0.0,
    memory_estimate_multiplier: float = 12.0,
) -> Path:
    """Build a compact feature/label CSV from Binance Vision ZIP archives.

    This intentionally avoids pandas so the first pipeline works on a clean
    Python install. It aggregates best-quote updates to fixed time buckets and
    labels each bucket by future mid-price movement.
    """
    if bucket_ms <= 0:
        raise ValueError("bucket_ms must be positive")
    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    if execution_latency_ms < 0:
        raise ValueError("execution_latency_ms must be >= 0")
    assert_feature_build_budget(
        [book_ticker_zip, agg_trades_zip, book_depth_zip],
        max_memory_gb=max_feature_build_memory_gb,
        multiplier=memory_estimate_multiplier,
    )

    quote_buckets = list(
        iter_quote_buckets(
            Path(book_ticker_zip),
            bucket_ms=bucket_ms,
            max_quote_buckets=max_quote_buckets,
        )
    )
    if not quote_buckets:
        raise ValueError("no quote buckets produced")

    first_bucket = quote_buckets[0].bucket_start_ms
    last_decision_time = quote_buckets[-1].decision_time
    trade_index: TradeBucketIndex | None = None
    if agg_trades_zip is not None:
        trade_index = build_trade_bucket_index(
            Path(agg_trades_zip),
            bucket_ms=bucket_ms,
            start_bucket_ms=first_bucket,
            end_time_ms=last_decision_time,
            large_trade_notional=large_trade_notional,
        )
    depth_snapshots: list[DepthSnapshot] = []
    depth_snapshot_times: list[int] = []
    if book_depth_zip is not None:
        depth_snapshots = load_depth_snapshots(
            Path(book_depth_zip),
            start_time_ms=first_bucket,
            end_time_ms=last_decision_time,
        )
        depth_snapshot_times = [snapshot.snapshot_time_ms for snapshot in depth_snapshots]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quote_event_times = [quote.event_time for quote in quote_buckets]
    quote_contexts = build_quote_contexts(quote_buckets, rolling_window=5)
    rows_written = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for idx, quote in enumerate(quote_buckets):
            decision_time = quote.decision_time
            entry_target_time = decision_time + execution_latency_ms
            future_target_time = decision_time + execution_latency_ms + horizon_ms
            entry_idx = bisect_left(quote_event_times, entry_target_time)
            if entry_idx >= len(quote_buckets):
                break
            entry = quote_buckets[entry_idx]
            future_idx = bisect_left(quote_event_times, future_target_time)
            if future_idx >= len(quote_buckets):
                break
            future = quote_buckets[future_idx]
            execution_path = summarize_execution_path(
                quote_buckets[entry_idx + 1 : future_idx + 1],
                entry=entry,
            )
            row = build_feature_row(
                quote=quote,
                entry=entry,
                future=future,
                execution_path=execution_path,
                trade=(
                    trade_index.aggregate_until(
                        bucket_start_ms=quote.bucket_start_ms,
                        decision_time_ms=decision_time,
                    )
                    if trade_index is not None
                    else TradeBucket()
                ),
                depth=latest_depth_snapshot(
                    depth_snapshots,
                    depth_snapshot_times,
                    decision_time,
                ),
                quote_context=quote_contexts[idx],
                execution_latency_ms=execution_latency_ms,
                holding_horizon_ms=horizon_ms,
                decision_time=decision_time,
                entry_target_time=entry_target_time,
                future_target_time=future_target_time,
                threshold=threshold,
                min_tick=min_tick,
            )
            writer.writerow(row)
            rows_written += 1

    if rows_written == 0:
        raise ValueError("no labeled rows written; reduce horizon or collect more quote buckets")

    return output_path


def combine_feature_csvs(
    inputs: list[tuple[Path | str, dict[str, str]]],
    output_csv: Path | str,
) -> Path:
    """Concatenate feature CSVs while adding stable source metadata columns."""
    if not inputs:
        raise ValueError("no input files provided")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    output_columns: list[str] | None = None
    with output_path.open("w", newline="") as output_handle:
        writer: csv.DictWriter | None = None
        for input_path_value, metadata in inputs:
            input_path = Path(input_path_value)
            with input_path.open(newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                if reader.fieldnames is None:
                    raise ValueError(f"missing header in {input_path}")

                metadata_columns = list(metadata.keys())
                current_columns = metadata_columns + list(reader.fieldnames)
                if not header_written:
                    output_columns = current_columns
                    writer = csv.DictWriter(output_handle, fieldnames=output_columns)
                    writer.writeheader()
                    header_written = True
                elif current_columns != output_columns:
                    raise ValueError(f"schema mismatch while combining {input_path}")

                assert writer is not None
                for row in reader:
                    writer.writerow({**metadata, **row})

    return output_path


def summarize_feature_csv(path: Path | str) -> FeatureSummary:
    label_counts: dict[str, int] = {}
    rows = 0
    rows_with_trades = 0
    rows_with_depth = 0
    spread_min = float("inf")
    spread_max = float("-inf")
    delta_mid_min = float("inf")
    delta_mid_max = float("-inf")

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            label = row["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            if int(row["trade_count"]):
                rows_with_trades += 1
            if int(row.get("depth_snapshot_age_ms") or -1) >= 0:
                rows_with_depth += 1
            spread = float(row["spread"])
            delta_mid = float(row["delta_mid"])
            spread_min = min(spread_min, spread)
            spread_max = max(spread_max, spread)
            delta_mid_min = min(delta_mid_min, delta_mid)
            delta_mid_max = max(delta_mid_max, delta_mid)

    if rows == 0:
        raise ValueError(f"No rows found in {path}")

    return FeatureSummary(
        rows=rows,
        label_counts=label_counts,
        rows_with_trades=rows_with_trades,
        rows_with_depth=rows_with_depth,
        spread_min=spread_min,
        spread_max=spread_max,
        delta_mid_min=delta_mid_min,
        delta_mid_max=delta_mid_max,
    )


def iter_quote_buckets(
    book_ticker_zip: Path,
    *,
    bucket_ms: int,
    max_quote_buckets: int | None = None,
) -> Iterator[QuoteBucket]:
    current: QuoteBucket | None = None
    emitted = 0
    for row in iter_zip_dict_rows(book_ticker_zip, BOOK_TICKER_COLUMNS):
        event_time = int(row["event_time"])
        bucket_start = event_time - (event_time % bucket_ms)
        quote = QuoteBucket(
            bucket_start_ms=bucket_start,
            decision_time_ms=event_time,
            event_time=event_time,
            update_id=int(row["update_id"]),
            bid=float(row["best_bid_price"]),
            ask=float(row["best_ask_price"]),
            bid_qty=float(row["best_bid_qty"]),
            ask_qty=float(row["best_ask_qty"]),
            local_receive_time=None,
        )
        if current is None:
            current = quote
            continue
        if bucket_start == current.bucket_start_ms:
            quote.update_count = current.update_count + 1
            current = quote
            continue
        yield current
        emitted += 1
        if max_quote_buckets is not None and emitted >= max_quote_buckets:
            return
        current = quote

    if current is not None:
        yield current


def aggregate_agg_trades(
    agg_trades_zip: Path,
    *,
    bucket_ms: int,
    start_bucket_ms: int | None = None,
    end_bucket_ms: int | None = None,
    large_trade_notional: float = 10_000.0,
) -> dict[int, TradeBucket]:
    buckets: dict[int, TradeBucket] = {}
    for row in iter_zip_dict_rows(agg_trades_zip, AGG_TRADE_COLUMNS):
        timestamp = int(row["transact_time"])
        bucket_start = timestamp - (timestamp % bucket_ms)
        if start_bucket_ms is not None and bucket_start < start_bucket_ms:
            continue
        if end_bucket_ms is not None and bucket_start > end_bucket_ms:
            break

        price = float(row["price"])
        qty = float(row["quantity"])
        notional = price * qty
        is_buyer_maker = row["is_buyer_maker"].strip().lower() == "true"

        bucket = buckets.setdefault(bucket_start, TradeBucket())
        bucket.trade_count += 1
        if is_buyer_maker:
            bucket.sell_qty += qty
            bucket.sell_notional += notional
        else:
            bucket.buy_qty += qty
            bucket.buy_notional += notional
        if notional >= large_trade_notional:
            bucket.large_trade_count += 1

    return buckets


def build_trade_bucket_index(
    agg_trades_zip: Path,
    *,
    bucket_ms: int,
    start_bucket_ms: int | None = None,
    end_time_ms: int | None = None,
    large_trade_notional: float = 10_000.0,
) -> TradeBucketIndex:
    events_by_bucket: dict[int, list[TradeEvent]] = {}
    for row in iter_zip_dict_rows(agg_trades_zip, AGG_TRADE_COLUMNS):
        timestamp = int(row["transact_time"])
        bucket_start = timestamp - (timestamp % bucket_ms)
        if start_bucket_ms is not None and bucket_start < start_bucket_ms:
            continue
        if end_time_ms is not None and timestamp > end_time_ms:
            break

        price = float(row["price"])
        qty = float(row["quantity"])
        notional = price * qty
        is_buyer_maker = row["is_buyer_maker"].strip().lower() == "true"
        events_by_bucket.setdefault(bucket_start, []).append(
            TradeEvent(
                timestamp_ms=timestamp,
                bucket_start_ms=bucket_start,
                quantity=qty,
                notional=notional,
                is_sell_initiated=is_buyer_maker,
                is_large=notional >= large_trade_notional,
            )
        )

    for events in events_by_bucket.values():
        events.sort(key=lambda event: event.timestamp_ms)
    return TradeBucketIndex(bucket_ms=bucket_ms, events_by_bucket=events_by_bucket)


def load_depth_snapshots(
    book_depth_zip: Path,
    *,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> list[DepthSnapshot]:
    snapshots: list[DepthSnapshot] = []
    current: DepthSnapshot | None = None

    for row in iter_zip_dict_rows(book_depth_zip, BOOK_DEPTH_COLUMNS):
        snapshot_time_ms = parse_book_depth_timestamp_ms(row["timestamp"])
        if end_time_ms is not None and snapshot_time_ms > end_time_ms:
            break

        if current is None or snapshot_time_ms != current.snapshot_time_ms:
            if current is not None and _should_keep_depth_snapshot(current, start_time_ms, end_time_ms):
                snapshots.append(current)
            current = DepthSnapshot(snapshot_time_ms=snapshot_time_ms)

        percentage = int(abs(float(row["percentage"])))
        depth = float(row["depth"])
        notional = float(row["notional"])
        if float(row["percentage"]) < 0:
            current.bid_depth_by_pct[percentage] = depth
            current.bid_notional_by_pct[percentage] = notional
        else:
            current.ask_depth_by_pct[percentage] = depth
            current.ask_notional_by_pct[percentage] = notional

    if current is not None and _should_keep_depth_snapshot(current, start_time_ms, end_time_ms):
        snapshots.append(current)

    return snapshots


def build_quote_contexts(
    quote_buckets: list[QuoteBucket],
    *,
    rolling_window: int = 5,
) -> list[QuoteContext]:
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")

    ofi_values: list[float] = []
    mid_returns: list[float] = []
    contexts: list[QuoteContext] = []

    for idx, quote in enumerate(quote_buckets):
        if idx == 0:
            quote_ofi = 0.0
            mid_return_1 = 0.0
        else:
            previous = quote_buckets[idx - 1]
            quote_ofi = compute_quote_ofi(previous, quote)
            mid_return_1 = (quote.mid - previous.mid) / previous.mid if previous.mid else 0.0

        ofi_values.append(quote_ofi)
        mid_returns.append(mid_return_1)
        window_start = max(0, idx - rolling_window + 1)
        quote_window = quote_buckets[window_start : idx + 1]
        ofi_window = ofi_values[window_start : idx + 1]
        return_window = mid_returns[window_start : idx + 1]
        size_sum = quote.bid_qty + quote.ask_qty
        ofi_5 = sum(ofi_window)
        mid_return_5 = (
            (quote.mid - quote_buckets[idx - rolling_window].mid) / quote_buckets[idx - rolling_window].mid
            if idx >= rolling_window and quote_buckets[idx - rolling_window].mid
            else 0.0
        )

        contexts.append(
            QuoteContext(
                quote_ofi=quote_ofi,
                quote_ofi_normalized=quote_ofi / size_sum if size_sum else 0.0,
                quote_ofi_5=ofi_5,
                quote_ofi_5_normalized=ofi_5 / size_sum if size_sum else 0.0,
                mid_return_1=mid_return_1,
                mid_return_5=mid_return_5,
                realized_volatility_5=_root_sum_squares(return_window),
                spread_mean_5=sum(item.spread for item in quote_window) / len(quote_window),
                top_imbalance_mean_5=sum(_top_imbalance(item) for item in quote_window) / len(quote_window),
            )
        )

    return contexts


def compute_quote_ofi(previous: QuoteBucket, current: QuoteBucket) -> float:
    bid_contribution = 0.0
    ask_contribution = 0.0
    if current.bid >= previous.bid:
        bid_contribution += current.bid_qty
    if current.bid <= previous.bid:
        bid_contribution -= previous.bid_qty
    if current.ask <= previous.ask:
        ask_contribution -= current.ask_qty
    if current.ask >= previous.ask:
        ask_contribution += previous.ask_qty
    return bid_contribution + ask_contribution


def summarize_execution_path(
    path_quotes: list[QuoteBucket],
    *,
    entry: QuoteBucket,
) -> ExecutionPath:
    if not path_quotes:
        return ExecutionPath(
            horizon_min_ask=entry.ask,
            horizon_max_bid=entry.bid,
            maker_long_fillable=False,
            maker_short_fillable=False,
        )

    horizon_min_ask = min(quote.ask for quote in path_quotes)
    horizon_max_bid = max(quote.bid for quote in path_quotes)
    long_fill_time: int | None = None
    short_fill_time: int | None = None
    for quote in path_quotes:
        if long_fill_time is None and quote.ask <= entry.bid:
            long_fill_time = quote.event_time
        if short_fill_time is None and quote.bid >= entry.ask:
            short_fill_time = quote.event_time
        if long_fill_time is not None and short_fill_time is not None:
            break

    return ExecutionPath(
        horizon_min_ask=horizon_min_ask,
        horizon_max_bid=horizon_max_bid,
        maker_long_fillable=long_fill_time is not None,
        maker_short_fillable=short_fill_time is not None,
        maker_long_fill_event_time=long_fill_time,
        maker_short_fill_event_time=short_fill_time,
    )


def latest_depth_snapshot(
    snapshots: list[DepthSnapshot],
    snapshot_times: list[int],
    event_time_ms: int,
) -> DepthSnapshot | None:
    if not snapshots:
        return None
    idx = bisect_right(snapshot_times, event_time_ms) - 1
    if idx < 0:
        return None
    return snapshots[idx]


def parse_book_depth_timestamp_ms(value: str) -> int:
    raw = value.strip()
    if raw.isdigit():
        return int(raw)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"unsupported bookDepth timestamp: {value!r}")


def build_feature_row(
    *,
    quote: QuoteBucket,
    entry: QuoteBucket,
    future: QuoteBucket,
    execution_path: ExecutionPath,
    trade: TradeBucket,
    depth: DepthSnapshot | None,
    quote_context: QuoteContext,
    execution_latency_ms: int,
    holding_horizon_ms: int,
    decision_time: int,
    entry_target_time: int,
    future_target_time: int,
    threshold: str,
    min_tick: float,
) -> dict[str, str | int]:
    mid = quote.mid
    spread = quote.spread
    entry_mid = entry.mid
    future_mid = future.mid
    delta_mid = future_mid - entry_mid

    size_sum = quote.bid_qty + quote.ask_qty
    top_imbalance = (quote.bid_qty - quote.ask_qty) / size_sum if size_sum else 0.0
    microprice = (quote.ask * quote.bid_qty + quote.bid * quote.ask_qty) / size_sum if size_sum else mid
    microprice_deviation = (microprice - mid) / spread if spread else 0.0

    theta = label_threshold(spread=entry.spread, min_tick=min_tick, mode=threshold)
    label = 1 if delta_mid > theta else -1 if delta_mid < -theta else 0

    depth_features = build_depth_feature_values(depth, event_time_ms=decision_time)

    return {
        "bucket_start_ms": quote.bucket_start_ms,
        "decision_time": decision_time,
        "feature_cutoff_time": decision_time,
        "event_time": decision_time,
        "quote_event_time": quote.event_time,
        "local_receive_time": quote.local_receive_time or "",
        "update_id": quote.update_id,
        "bid": _fmt(quote.bid),
        "ask": _fmt(quote.ask),
        "bid_qty": _fmt(quote.bid_qty),
        "ask_qty": _fmt(quote.ask_qty),
        "mid": _fmt(mid),
        "spread": _fmt(spread),
        "relative_spread": _fmt(spread / mid if mid else 0.0),
        "top_imbalance": _fmt(top_imbalance),
        "microprice": _fmt(microprice),
        "microprice_deviation": _fmt(microprice_deviation),
        "quote_updates_in_bucket": quote.update_count,
        "quote_ofi": _fmt(quote_context.quote_ofi),
        "quote_ofi_normalized": _fmt(quote_context.quote_ofi_normalized),
        "quote_ofi_5": _fmt(quote_context.quote_ofi_5),
        "quote_ofi_5_normalized": _fmt(quote_context.quote_ofi_5_normalized),
        "mid_return_1": _fmt(quote_context.mid_return_1),
        "mid_return_5": _fmt(quote_context.mid_return_5),
        "realized_volatility_5": _fmt(quote_context.realized_volatility_5),
        "spread_mean_5": _fmt(quote_context.spread_mean_5),
        "top_imbalance_mean_5": _fmt(quote_context.top_imbalance_mean_5),
        "trade_count": trade.trade_count,
        "buy_qty": _fmt(trade.buy_qty),
        "sell_qty": _fmt(trade.sell_qty),
        "trade_qty": _fmt(trade.trade_qty),
        "buy_notional": _fmt(trade.buy_notional),
        "sell_notional": _fmt(trade.sell_notional),
        "trade_notional": _fmt(trade.trade_notional),
        "trade_imbalance": _fmt(trade.trade_imbalance),
        "large_trade_count": trade.large_trade_count,
        **depth_features,
        "execution_latency_ms": execution_latency_ms,
        "holding_horizon_ms": holding_horizon_ms,
        "entry_target_time": entry_target_time,
        "future_target_time": future_target_time,
        "entry_lag_ms": entry.event_time - entry_target_time,
        "future_lag_ms": future.event_time - future_target_time,
        "entry_event_time": entry.event_time,
        "entry_bid": _fmt(entry.bid),
        "entry_ask": _fmt(entry.ask),
        "entry_mid": _fmt(entry_mid),
        "entry_spread": _fmt(entry.spread),
        "future_event_time": future.event_time,
        "future_bid": _fmt(future.bid),
        "future_ask": _fmt(future.ask),
        "future_mid": _fmt(future_mid),
        "future_spread": _fmt(future.spread),
        "horizon_min_ask": _fmt(execution_path.horizon_min_ask),
        "horizon_max_bid": _fmt(execution_path.horizon_max_bid),
        "maker_long_fillable": int(execution_path.maker_long_fillable),
        "maker_short_fillable": int(execution_path.maker_short_fillable),
        "maker_long_fill_event_time": execution_path.maker_long_fill_event_time or "",
        "maker_short_fill_event_time": execution_path.maker_short_fill_event_time or "",
        "delta_mid": _fmt(delta_mid),
        "label": label,
    }


def build_depth_feature_values(
    depth: DepthSnapshot | None,
    *,
    event_time_ms: int,
) -> dict[str, str | int]:
    if depth is None:
        values: dict[str, str | int] = {
            "depth_snapshot_time": "",
            "depth_snapshot_age_ms": -1,
        }
        for band in DEPTH_BANDS:
            values.update(_empty_depth_band_values(band))
        return values

    values = {
        "depth_snapshot_time": depth.snapshot_time_ms,
        "depth_snapshot_age_ms": event_time_ms - depth.snapshot_time_ms,
    }
    for band in DEPTH_BANDS:
        bid_depth = depth.bid_depth_by_pct.get(band, 0.0)
        ask_depth = depth.ask_depth_by_pct.get(band, 0.0)
        bid_notional = depth.bid_notional_by_pct.get(band, 0.0)
        ask_notional = depth.ask_notional_by_pct.get(band, 0.0)
        values.update(
            {
                f"bid_depth_{band}pct": _fmt(bid_depth),
                f"ask_depth_{band}pct": _fmt(ask_depth),
                f"bid_notional_{band}pct": _fmt(bid_notional),
                f"ask_notional_{band}pct": _fmt(ask_notional),
                f"depth_imbalance_{band}pct": _fmt(_imbalance(bid_depth, ask_depth)),
                f"notional_imbalance_{band}pct": _fmt(_imbalance(bid_notional, ask_notional)),
            }
        )
    return values


def label_threshold(*, spread: float, min_tick: float, mode: str) -> float:
    if mode == "half_spread":
        return max(0.5 * spread, min_tick)
    if mode == "one_tick":
        return min_tick
    if mode == "zero":
        return 0.0
    raise ValueError("threshold must be one of: half_spread, one_tick, zero")


def iter_zip_dict_rows(path: Path, default_columns: list[str]) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in {path}")
        with zf.open(csv_names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text)
            header: list[str] | None = None
            for row in reader:
                if not row:
                    continue
                if header is None:
                    if _is_header(row):
                        header = row
                        continue
                    header = default_columns
                if len(row) != len(header):
                    raise ValueError(f"Unexpected row width in {path}: expected {len(header)}, got {len(row)}")
                yield dict(zip(header, row))


def _is_header(row: list[str]) -> bool:
    return any(not _looks_numeric(cell) for cell in row)


def _looks_numeric(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return True
    try:
        float(lowered)
    except ValueError:
        return False
    return True


def _should_keep_depth_snapshot(
    snapshot: DepthSnapshot,
    start_time_ms: int | None,
    end_time_ms: int | None,
) -> bool:
    if start_time_ms is not None and snapshot.snapshot_time_ms < start_time_ms:
        return False
    if end_time_ms is not None and snapshot.snapshot_time_ms > end_time_ms:
        return False
    return True


def _empty_depth_band_values(band: int) -> dict[str, str]:
    return {
        f"bid_depth_{band}pct": _fmt(0.0),
        f"ask_depth_{band}pct": _fmt(0.0),
        f"bid_notional_{band}pct": _fmt(0.0),
        f"ask_notional_{band}pct": _fmt(0.0),
        f"depth_imbalance_{band}pct": _fmt(0.0),
        f"notional_imbalance_{band}pct": _fmt(0.0),
    }


def _imbalance(bid_value: float, ask_value: float) -> float:
    total = bid_value + ask_value
    return (bid_value - ask_value) / total if total else 0.0


def _top_imbalance(quote: QuoteBucket) -> float:
    total = quote.bid_qty + quote.ask_qty
    return (quote.bid_qty - quote.ask_qty) / total if total else 0.0


def _root_sum_squares(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def _fmt(value: float) -> str:
    return f"{value:.12g}"
