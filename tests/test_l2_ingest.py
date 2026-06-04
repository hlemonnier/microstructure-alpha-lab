import csv
import io
import json
import zipfile
from pathlib import Path

from lob_forge.data_sources import iter_bybit_orderbook_data_zip, validate_bybit_orderbook_data_zip, validate_l2_csv
from lob_forge.l2_ingest import (
    HistoricalL2ManifestEntry,
    build_bybit_list_files_query,
    build_okx_download_link_payload,
    build_historical_l2_manifest,
    download_historical_l2_manifest,
    import_historical_l2_file,
    import_historical_l2_manifest,
    read_historical_l2_manifest,
    resolve_bybit_historical_l2_manifest,
    resolve_okx_historical_l2_manifest,
    write_historical_l2_manifest,
)


def test_validate_l2_csv_handles_zip_archives(tmp_path: Path) -> None:
    archive_path = tmp_path / "bybit_orderbook.zip"
    payload = io.StringIO()
    writer = csv.DictWriter(
        payload,
        fieldnames=["type", "timestamp", "side", "price", "qty", "seq", "symbol"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "type": "delta",
            "timestamp": "1700000000000",
            "side": "sell",
            "price": "101.25",
            "qty": "0.75",
            "seq": "42",
            "symbol": "BTCUSDT",
        }
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("orderbook.csv", payload.getvalue())

    result = validate_l2_csv(
        archive_path,
        source_id="bybit",
        require_sequence=True,
    )

    assert result.ok
    assert result.rows_checked == 1


def test_build_historical_l2_manifest_expands_symbols_and_dates() -> None:
    entries = build_historical_l2_manifest(
        source_id="okx",
        symbols=["BTC-USDT-SWAP", " ETH-USDT-SWAP "],
        start="2023-05-16",
        end="2023-05-17",
    )

    assert len(entries) == 4
    assert entries[0].source_id == "okx"
    assert entries[0].symbol == "BTC-USDT-SWAP"
    assert entries[0].session_date == "2023-05-16"
    assert entries[0].status == "url_required"
    assert entries[0].source_page_url == "https://www.okx.com/en-gb/historical-data"
    assert entries[-1].symbol == "ETH-USDT-SWAP"
    assert entries[-1].session_date == "2023-05-17"


def test_historical_l2_manifest_download_and_import(tmp_path: Path) -> None:
    source_file = tmp_path / "okx_l2.csv"
    _write_csv(
        source_file,
        [
            {
                "type": "snapshot",
                "timestamp": "1700000000000",
                "side": "buy",
                "price": "100.00",
                "amount": "1.5",
                "seq": "1",
            },
            {
                "type": "delta",
                "timestamp": "1700000000001",
                "side": "sell",
                "price": "100.25",
                "amount": "0.4",
                "seq": "2",
            },
        ],
    )
    manifest_path = tmp_path / "manifests" / "okx_manifest.csv"
    write_historical_l2_manifest(
        [
            HistoricalL2ManifestEntry(
                source_id="okx",
                symbol="BTC-USDT-SWAP",
                session_date="2023-05-16",
                dataset="order_book_l2",
                market="swap",
                source_page_url="https://www.okx.com/en-gb/historical-data",
                direct_url=str(source_file),
            )
        ],
        manifest_path,
    )

    downloaded = download_historical_l2_manifest(
        manifest_path,
        raw_root=tmp_path / "raw",
    )
    imported = import_historical_l2_manifest(
        manifest_path,
        output_root=tmp_path / "store",
    )
    manifest_entries = read_historical_l2_manifest(manifest_path)

    assert downloaded[0].status == "downloaded"
    assert Path(downloaded[0].local_path).exists()
    assert downloaded[0].bytes > 0
    assert downloaded[0].sha256
    assert len(imported) == 1
    assert imported[0].rows_written == 2
    assert imported[0].validation.ok
    assert imported[0].output_path == tmp_path / "store" / "normalized_l2" / "okx" / "BTC-USDT-SWAP" / "2023-05-16.csv"
    assert manifest_entries[0].status == "normalized"
    assert manifest_entries[0].normalized_path == str(imported[0].output_path)

    with imported[0].output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["symbol"] == "BTC-USDT-SWAP"
    assert rows[0]["side"] == "bid"
    assert rows[1]["side"] == "ask"


def test_okx_download_link_payload_uses_swap_instrument_family() -> None:
    entry = HistoricalL2ManifestEntry(
        source_id="okx",
        symbol="BTC-USDT-SWAP",
        session_date="2023-05-16",
        dataset="order_book_l2",
        market="swap",
        source_page_url="https://www.okx.com/en-gb/historical-data",
    )

    payload = build_okx_download_link_payload(entry, module="4")

    assert payload == {
        "module": "4",
        "instType": "SWAP",
        "instQueryParam": {"instFamilyList": ["BTC-USDT"]},
        "dateQuery": {
            "dateAggrType": "daily",
            "begin": "1684195200000",
            "end": "1684281599999",
        },
    }


def test_resolve_okx_manifest_records_download_links(tmp_path: Path) -> None:
    manifest_path = tmp_path / "okx_manifest.csv"
    write_historical_l2_manifest(
        [
            HistoricalL2ManifestEntry(
                source_id="okx",
                symbol="BTC-USDT-SWAP",
                session_date="2023-05-16",
                dataset="order_book_l2",
                market="swap",
                source_page_url="https://www.okx.com/en-gb/historical-data",
            )
        ],
        manifest_path,
    )

    def fake_fetcher(entry: HistoricalL2ManifestEntry, module: str, timeout_seconds: float) -> dict:
        assert entry.symbol == "BTC-USDT-SWAP"
        assert module == "4"
        assert timeout_seconds == 30.0
        return {
            "details": [
                {
                    "instFamily": "BTC-USDT",
                    "url": "https://static.okx.com/example/btc-orderbook.zip",
                    "fileSizeMB": "12.5",
                }
            ],
            "exportTime": "1700000000000",
            "totalSizeMB": "12.5",
        }

    entries = resolve_okx_historical_l2_manifest(
        manifest_path,
        depth="400",
        throttle_seconds=0,
        fetcher=fake_fetcher,
    )
    reread = read_historical_l2_manifest(manifest_path)

    assert entries[0].status == "url_resolved"
    assert entries[0].direct_url == "https://static.okx.com/example/btc-orderbook.zip"
    assert "OKX depth=400" in entries[0].notes
    assert reread[0].status == entries[0].status
    assert reread[0].direct_url == entries[0].direct_url
    assert reread[0].notes == entries[0].notes


def test_resolve_okx_manifest_records_no_url_found(tmp_path: Path) -> None:
    manifest_path = tmp_path / "okx_manifest.csv"
    write_historical_l2_manifest(
        [
            HistoricalL2ManifestEntry(
                source_id="okx",
                symbol="ETH-USDT-SWAP",
                session_date="2023-05-16",
                dataset="order_book_l2",
                market="swap",
                source_page_url="https://www.okx.com/en-gb/historical-data",
            )
        ],
        manifest_path,
    )

    entries = resolve_okx_historical_l2_manifest(
        manifest_path,
        depth="400",
        throttle_seconds=0,
        fetcher=lambda entry, module, timeout_seconds: {"details": [], "totalSizeMB": "0"},
    )

    assert entries[0].status == "no_url_found"
    assert entries[0].direct_url == ""
    assert "details=0" in entries[0].notes


def test_bybit_list_files_query_maps_swap_manifest_to_contract_orderbook() -> None:
    entry = HistoricalL2ManifestEntry(
        source_id="bybit",
        symbol="BTCUSDT",
        session_date="2023-05-16",
        dataset="order_book_l2",
        market="swap",
        source_page_url="https://www.bybit.com/derivatives/en/history-data",
    )

    query = build_bybit_list_files_query(entry)

    assert query == {
        "bizType": "contract",
        "productId": "orderbook",
        "symbols": "BTCUSDT",
        "interval": "daily",
        "startDay": "2023-05-16",
        "endDay": "2023-05-16",
    }


def test_resolve_bybit_manifest_records_orderbook_url(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bybit_manifest.csv"
    write_historical_l2_manifest(
        [
            HistoricalL2ManifestEntry(
                source_id="bybit",
                symbol="BTCUSDT",
                session_date="2023-05-16",
                dataset="order_book_l2",
                market="contract",
                source_page_url="https://www.bybit.com/derivatives/en/history-data",
            )
        ],
        manifest_path,
    )

    def fake_fetcher(entry: HistoricalL2ManifestEntry, timeout_seconds: float) -> dict:
        assert entry.symbol == "BTCUSDT"
        assert timeout_seconds == 30.0
        return {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "date": "2023-05-16",
                    "filename": "2023-05-16_BTCUSDT_ob500.data.zip",
                    "size": "107156124",
                    "url": "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/2023-05-16_BTCUSDT_ob500.data.zip",
                    "period": "",
                }
            ]
        }

    entries = resolve_bybit_historical_l2_manifest(
        manifest_path,
        throttle_seconds=0,
        fetcher=fake_fetcher,
    )

    assert entries[0].status == "url_resolved"
    assert entries[0].direct_url.endswith("2023-05-16_BTCUSDT_ob500.data.zip")
    assert "remote_size_bytes=107156124" in entries[0].notes


def test_bybit_orderbook_data_zip_imports_to_normalized_rows(tmp_path: Path) -> None:
    archive_path = tmp_path / "2023-05-16_BTCUSDT_ob500.data.zip"
    payloads = [
        {
            "topic": "orderbook.500.BTCUSDT",
            "type": "snapshot",
            "ts": 1684195201451,
            "data": {
                "s": "BTCUSDT",
                "b": [["27143.70", "1.223"]],
                "a": [["27144.10", "0.500"]],
                "seq": 10,
                "u": 1,
                "cts": 1684195201400,
            },
        },
        {
            "topic": "orderbook.500.BTCUSDT",
            "type": "delta",
            "ts": 1684195202451,
            "data": {
                "s": "BTCUSDT",
                "b": [["27143.70", "0"]],
                "a": [],
                "seq": 11,
                "u": 2,
                "cts": 1684195202400,
            },
        },
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("2023-05-16_BTCUSDT_ob500.data", "\n".join(json.dumps(item) for item in payloads) + "\n")

    rows = list(iter_bybit_orderbook_data_zip(archive_path, default_symbol="BTCUSDT"))
    validation = validate_bybit_orderbook_data_zip(archive_path)
    imported = import_historical_l2_file(
        archive_path,
        source_id="bybit",
        symbol="BTCUSDT",
        session_date="2023-05-16",
        output_root=tmp_path / "store",
    )

    assert len(rows) == 3
    assert rows[0].event_type == "snapshot"
    assert rows[0].exchange_timestamp == 1684195201400
    assert rows[0].sequence == 10
    assert rows[0].update_id == 1
    assert rows[1].side == "ask"
    assert rows[2].event_type == "delta"
    assert rows[2].size == 0.0
    assert validation.ok
    assert validation.rows_checked == 3
    assert imported.rows_written == 3

    with imported.output_path.open(newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert output_rows[0]["venue"] == "bybit"
    assert output_rows[0]["symbol"] == "BTCUSDT"


def test_bybit_orderbook_import_can_be_row_capped(tmp_path: Path) -> None:
    archive_path = tmp_path / "2023-05-16_BTCUSDT_ob500.data.zip"
    payloads = [
        {
            "type": "snapshot",
            "ts": 1684195201451,
            "data": {
                "s": "BTCUSDT",
                "b": [["27143.70", "1.223"], ["27143.60", "0.300"]],
                "a": [["27144.10", "0.500"]],
                "seq": 10,
                "u": 1,
                "cts": 1684195201400,
            },
        },
        {
            "type": "delta",
            "ts": 1684195202451,
            "data": {
                "s": "BTCUSDT",
                "b": [["27143.70", "0"]],
                "a": [["27144.20", "0.100"]],
                "seq": 11,
                "u": 2,
                "cts": 1684195202400,
            },
        }
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("2023-05-16_BTCUSDT_ob500.data", "\n".join(json.dumps(item) for item in payloads) + "\n")

    imported = import_historical_l2_file(
        archive_path,
        source_id="bybit",
        symbol="BTCUSDT",
        session_date="2023-05-16",
        output_root=tmp_path / "store",
        max_import_rows=4,
    )

    assert imported.rows_written == 3
    with imported.output_path.open(newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == 3
    assert {row["event_type"] for row in output_rows} == {"snapshot"}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
