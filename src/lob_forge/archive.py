from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path


TRADE_COLUMNS = [
    "id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
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

BOOK_TICKER_COLUMNS = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]

BOOK_DEPTH_COLUMNS = [
    "timestamp",
    "percentage",
    "depth",
    "notional",
]


@dataclass(frozen=True)
class ZipCsvSample:
    archive: Path
    inner_name: str
    row_count_sampled: int
    header: list[str] | None
    rows: list[list[str]]


def sample_zip_csv(path: Path | str, rows: int = 5) -> ZipCsvSample:
    """Read the first CSV member in a Binance Vision ZIP archive."""
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as zf:
        csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in {archive_path}")
        inner_name = csv_names[0]
        with zf.open(inner_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text)
            first_rows: list[list[str]] = []
            for idx, row in enumerate(reader):
                if idx >= rows + 1:
                    break
                first_rows.append(row)

    if not first_rows:
        return ZipCsvSample(archive_path, inner_name, 0, None, [])

    first = first_rows[0]
    has_header = any(not _looks_numeric(cell) for cell in first)
    if has_header:
        header = first
        data_rows = first_rows[1 : rows + 1]
    else:
        header = _default_header(inner_name, len(first))
        data_rows = first_rows[:rows]

    return ZipCsvSample(
        archive=archive_path,
        inner_name=inner_name,
        row_count_sampled=len(data_rows),
        header=header,
        rows=data_rows,
    )


def _looks_numeric(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return True
    try:
        float(lowered)
    except ValueError:
        return False
    return True


def _default_header(inner_name: str, width: int) -> list[str] | None:
    if "-trades-" in inner_name and width == len(TRADE_COLUMNS):
        return TRADE_COLUMNS
    if "-aggTrades-" in inner_name and width == len(AGG_TRADE_COLUMNS):
        return AGG_TRADE_COLUMNS
    if "-bookTicker-" in inner_name and width == len(BOOK_TICKER_COLUMNS):
        return BOOK_TICKER_COLUMNS
    if "-bookDepth-" in inner_name and width == len(BOOK_DEPTH_COLUMNS):
        return BOOK_DEPTH_COLUMNS
    return None
