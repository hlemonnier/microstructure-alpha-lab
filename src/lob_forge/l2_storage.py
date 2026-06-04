from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

from lob_forge.data_sources import NormalizedL2Row


NORMALIZED_L2_COLUMNS = [
    "event_type",
    "exchange_timestamp",
    "local_timestamp",
    "side",
    "price",
    "size",
    "sequence",
    "update_id",
    "venue",
    "symbol",
]


def normalized_l2_parquet_path(
    root: Path | str,
    *,
    venue: str,
    symbol: str,
    session_date: str | date,
) -> Path:
    day = session_date.isoformat() if isinstance(session_date, date) else str(session_date)
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return Path(root) / "normalized_l2" / venue / safe_symbol / f"{day}.parquet"


def normalized_l2_csv_path(
    root: Path | str,
    *,
    venue: str,
    symbol: str,
    session_date: str | date,
) -> Path:
    return normalized_l2_parquet_path(
        root,
        venue=venue,
        symbol=symbol,
        session_date=session_date,
    ).with_suffix(".csv")


def write_normalized_l2_csv(rows: Iterable[NormalizedL2Row], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_L2_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: asdict(row).get(column) for column in NORMALIZED_L2_COLUMNS})
    return path


def write_normalized_l2_parquet(rows: Iterable[NormalizedL2Row], path: Path | str) -> Path:
    """Write normalized L2 rows to Parquet when pyarrow is installed.

    The base project intentionally has no heavy dependencies; this function is
    the durable Parquet storage hook used by the research extras environment.
    """
    path = Path(path)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Parquet storage; install the research extras") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{column: asdict(row).get(column) for column in NORMALIZED_L2_COLUMNS} for row in rows]
    table = pa.Table.from_pylist(payload, schema=_parquet_schema(pa))
    pq.write_table(table, path)
    return path


def _parquet_schema(pa):
    return pa.schema(
        [
            ("event_type", pa.string()),
            ("exchange_timestamp", pa.int64()),
            ("local_timestamp", pa.int64()),
            ("side", pa.string()),
            ("price", pa.float64()),
            ("size", pa.float64()),
            ("sequence", pa.int64()),
            ("update_id", pa.int64()),
            ("venue", pa.string()),
            ("symbol", pa.string()),
        ]
    )
