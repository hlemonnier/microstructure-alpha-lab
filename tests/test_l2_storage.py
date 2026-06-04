from pathlib import Path

from lob_forge.data_sources import NormalizedL2Row
from lob_forge.l2_storage import (
    normalized_l2_csv_path,
    normalized_l2_parquet_path,
    write_normalized_l2_csv,
    write_normalized_l2_parquet,
)


def test_l2_storage_paths_are_partitioned() -> None:
    path = normalized_l2_parquet_path("data", venue="okx", symbol="BTC/USDT:SWAP", session_date="2026-06-03")

    assert path == Path("data/normalized_l2/okx/BTC-USDT-SWAP/2026-06-03.parquet")
    assert normalized_l2_csv_path("data", venue="okx", symbol="BTC-USDT", session_date="2026-06-03").suffix == ".csv"


def test_write_normalized_l2_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = [
        NormalizedL2Row(
            event_type="snapshot",
            exchange_timestamp=1700000000000,
            local_timestamp=1700000000001,
            side="bid",
            price=100.0,
            size=2.0,
            sequence=1,
            update_id=10,
            venue="okx",
            symbol="BTC-USDT",
        )
    ]

    write_normalized_l2_csv(rows, path)

    assert "exchange_timestamp" in path.read_text()
    assert "BTC-USDT" in path.read_text()


def test_parquet_writer_is_dependency_gated(tmp_path: Path) -> None:
    rows = [
        NormalizedL2Row(
            event_type="delta",
            exchange_timestamp=1,
            local_timestamp=None,
            side="ask",
            price=101.0,
            size=0.0,
        )
    ]

    try:
        write_normalized_l2_parquet(rows, tmp_path / "rows.parquet")
    except RuntimeError as exc:
        assert "pyarrow" in str(exc)
