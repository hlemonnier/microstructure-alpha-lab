import csv
import gzip
from pathlib import Path

from lob_forge.data_sources import (
    get_source,
    list_sources,
    normalize_l2_row,
    source_priority_markdown,
    validate_l2_columns,
    validate_l2_csv,
    validate_l2_row,
)


def test_source_priority_keeps_okx_first_and_fi2010_last() -> None:
    sources = list_sources()

    assert sources[0].source_id == "okx"
    assert sources[-1].source_id == "fi2010"
    assert get_source("coinbase").free_historical_l2 is False
    assert "OKX Historical Market Data" in source_priority_markdown()


def test_l2_schema_validation_accepts_aliases_and_requires_core_fields() -> None:
    columns = ["type", "timestamp", "side", "price", "amount", "seq"]

    assert validate_l2_columns(columns, source_id="tardis") == ()
    assert "size" in validate_l2_columns(["type", "timestamp", "side", "price"], source_id="okx")


def test_l2_row_validation_normalizes_snapshot_delta_and_side() -> None:
    row = {
        "type": "snapshot",
        "timestamp": "1700000000000",
        "local_timestamp": "1700000000001",
        "side": "buy",
        "price": "100.5",
        "amount": "2.25",
        "seq": "42",
        "exchange": "tardis",
        "symbol": "BTC-USDT",
    }

    validation = validate_l2_row(row, source_id="tardis")
    normalized = normalize_l2_row(row, source_id="tardis")

    assert validation.ok
    assert normalized.event_type == "snapshot"
    assert normalized.side == "bid"
    assert normalized.price == 100.5
    assert normalized.sequence == 42
    assert normalized.venue == "tardis"


def test_bybit_replay_grade_requires_sequence_or_update_id() -> None:
    row = {
        "type": "delta",
        "timestamp": "1700000000000",
        "side": "ask",
        "price": "101",
        "qty": "1",
    }

    validation = validate_l2_row(row, source_id="bybit")

    assert not validation.ok
    assert validation.errors[-1].field == "sequence"


def test_validate_l2_csv_handles_gzip(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv.gz"
    rows = [
        {
            "type": "delta",
            "timestamp": "1700000000000",
            "side": "ask",
            "price": "101",
            "amount": "1",
            "seq": "7",
        }
    ]
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = validate_l2_csv(path, source_id="tardis")

    assert result.ok
    assert result.rows_checked == 1
