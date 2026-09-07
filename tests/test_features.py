import csv
import zipfile
from pathlib import Path

from lob_forge.features import (
    QuoteBucket,
    build_quote_contexts,
    build_quote_trade_dataset,
    combine_feature_csvs,
    compute_quote_ofi,
    load_depth_snapshots,
    parse_book_depth_timestamp_ms,
)


def test_build_quote_trade_dataset(tmp_path: Path) -> None:
    book_zip = tmp_path / "bookTicker.zip"
    trades_zip = tmp_path / "aggTrades.zip"
    output = tmp_path / "features.csv"

    _write_zip_csv(
        book_zip,
        "BTCUSDT-bookTicker-2023-05-16.csv",
        [
            [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time",
            ],
            ["1", "100.0", "5.0", "100.2", "2.0", "1000", "1000"],
            ["2", "100.1", "4.0", "100.3", "3.0", "1500", "1500"],
            ["3", "100.4", "6.0", "100.6", "2.0", "2000", "2000"],
            ["4", "100.5", "6.0", "100.7", "2.0", "3000", "3000"],
            ["5", "100.6", "6.0", "100.8", "2.0", "4000", "4000"],
        ],
    )
    _write_zip_csv(
        trades_zip,
        "BTCUSDT-aggTrades-2023-05-16.csv",
        [
            [
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "transact_time",
                "is_buyer_maker",
            ],
            ["1", "100.2", "1.0", "10", "10", "1001", "false"],
            ["2", "100.1", "2.0", "11", "11", "1002", "true"],
        ],
    )

    build_quote_trade_dataset(
        book_ticker_zip=book_zip,
        agg_trades_zip=trades_zip,
        output_csv=output,
        bucket_ms=1000,
        horizon_ms=1000,
        threshold="zero",
    )

    with output.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["label"] == "1"
    assert rows[0]["trade_count"] == "2"
    assert float(rows[0]["trade_imbalance"]) < 0
    assert rows[0]["decision_time"] == "2000"
    assert rows[0]["event_time"] == rows[0]["decision_time"]
    assert rows[0]["quote_event_time"] == "1500"
    assert rows[0]["local_receive_time"] == ""
    assert rows[0]["entry_mid"] == "100.5"
    assert rows[0]["future_mid"] == "100.6"
    assert rows[0]["depth_snapshot_age_ms"] == "-1"
    assert rows[0]["quote_ofi"] == "6"
    assert rows[1]["quote_ofi"] == "9"
    assert rows[1]["quote_ofi_normalized"] == "1.125"
    assert float(rows[1]["mid_return_1"]) > 0
    assert rows[0]["maker_long_fillable"] == "0"
    assert rows[0]["maker_short_fillable"] == "0"
    assert rows[0]["horizon_max_bid"] == "100.5"


def test_trade_after_decision_time_does_not_change_features(tmp_path: Path) -> None:
    book_zip = tmp_path / "bookTicker.zip"
    clean_trades_zip = tmp_path / "aggTrades-clean.zip"
    adversarial_trades_zip = tmp_path / "aggTrades-adversarial.zip"
    clean_output = tmp_path / "clean.csv"
    adversarial_output = tmp_path / "adversarial.csv"

    _write_zip_csv(
        book_zip,
        "BTCUSDT-bookTicker-2023-05-16.csv",
        [
            [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time",
            ],
            ["1", "100.0", "5.0", "100.2", "2.0", "1000", "1000"],
            ["2", "100.1", "4.0", "100.3", "3.0", "1500", "1500"],
            ["3", "100.4", "6.0", "100.6", "2.0", "2500", "2500"],
            ["4", "100.5", "6.0", "100.7", "2.0", "3500", "3500"],
        ],
    )
    header = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
    base_rows = [
        header,
        ["1", "100.2", "1.0", "10", "10", "1400", "false"],
    ]
    _write_zip_csv(clean_trades_zip, "clean.csv", base_rows)
    _write_zip_csv(
        adversarial_trades_zip,
        "adversarial.csv",
        base_rows + [["2", "101.0", "99.0", "11", "11", "2001", "true"]],
    )

    for trades_zip, output in [(clean_trades_zip, clean_output), (adversarial_trades_zip, adversarial_output)]:
        build_quote_trade_dataset(
            book_ticker_zip=book_zip,
            agg_trades_zip=trades_zip,
            output_csv=output,
            bucket_ms=1000,
            horizon_ms=1000,
            threshold="zero",
        )

    with clean_output.open() as handle:
        clean_rows = list(csv.DictReader(handle))
    with adversarial_output.open() as handle:
        adversarial_rows = list(csv.DictReader(handle))

    comparable_fields = [
        "decision_time",
        "trade_count",
        "buy_qty",
        "sell_qty",
        "trade_qty",
        "trade_imbalance",
        "large_trade_count",
        "label",
    ]
    assert clean_rows[0]["decision_time"] == "2000"
    assert {field: clean_rows[0][field] for field in comparable_fields} == {
        field: adversarial_rows[0][field] for field in comparable_fields
    }


def test_build_quote_trade_dataset_resolves_execution_on_raw_quotes(tmp_path: Path) -> None:
    book_zip = tmp_path / "bookTicker.zip"
    trades_zip = tmp_path / "aggTrades.zip"
    output = tmp_path / "features.csv"

    _write_zip_csv(
        book_zip,
        "BTCUSDT-bookTicker-2023-05-16.csv",
        [
            [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time",
            ],
            ["1", "100.0", "5.0", "100.2", "2.0", "900", "900"],
            ["2", "100.1", "4.0", "100.3", "3.0", "1050", "1050"],
            ["3", "101.0", "6.0", "101.2", "2.0", "1480", "1480"],
            ["4", "99.0", "6.0", "99.2", "2.0", "1910", "1910"],
            ["5", "102.0", "6.0", "102.2", "2.0", "2200", "2200"],
        ],
    )
    _write_zip_csv(
        trades_zip,
        "BTCUSDT-aggTrades-2023-05-16.csv",
        [
            ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"],
        ],
    )

    build_quote_trade_dataset(
        book_ticker_zip=book_zip,
        agg_trades_zip=trades_zip,
        output_csv=output,
        bucket_ms=1000,
        horizon_ms=1000,
        execution_latency_ms=200,
        threshold="zero",
    )

    with output.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["decision_time"] == "1000"
    assert rows[0]["entry_target_time"] == "1200"
    assert rows[0]["entry_event_time"] == "1480"
    assert rows[0]["entry_lag_ms"] == "280"
    assert rows[0]["entry_mid"] == "101.1"
    assert rows[0]["future_event_time"] == "2200"
    assert rows[0]["future_lag_ms"] == "0"
    assert rows[0]["future_mid"] == "102.1"
    assert rows[0]["horizon_min_ask"] == "99.2"
    assert rows[0]["maker_long_fillable"] == "1"
    assert rows[0]["maker_long_fill_event_time"] == "1910"


def test_build_quote_trade_dataset_rejects_noncausal_bucket_execution(tmp_path: Path) -> None:
    try:
        build_quote_trade_dataset(
            book_ticker_zip=tmp_path / "unused.zip", output_csv=tmp_path / "unused.csv",
            execution_quote_resolution="bucket",
        )
    except ValueError as exc:
        assert "not causal" in str(exc)
    else:
        raise AssertionError("retained last quote is not an executable future stopping time")


def test_build_quote_trade_dataset_refuses_feature_memory_budget(tmp_path: Path) -> None:
    book_zip = tmp_path / "bookTicker.zip"
    trades_zip = tmp_path / "aggTrades.zip"
    output = tmp_path / "features.csv"
    _write_zip_csv(
        book_zip,
        "BTCUSDT-bookTicker-2023-05-16.csv",
        [
            [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time",
            ],
            ["1", "100.0", "5.0", "100.2", "2.0", "1000", "1000"],
        ],
    )
    _write_zip_csv(
        trades_zip,
        "BTCUSDT-aggTrades-2023-05-16.csv",
        [
            ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"],
            ["1", "100.2", "1.0", "10", "10", "1001", "false"],
        ],
    )

    try:
        build_quote_trade_dataset(
            book_ticker_zip=book_zip,
            agg_trades_zip=trades_zip,
            output_csv=output,
            max_feature_build_memory_gb=0.000000001,
            memory_estimate_multiplier=100.0,
        )
    except ValueError as exc:
        assert "estimated feature-build memory exceeds budget" in str(exc)
    else:
        raise AssertionError("expected feature-build memory budget rejection")


def test_build_quote_trade_dataset_with_depth_bands(tmp_path: Path) -> None:
    book_zip = tmp_path / "bookTicker.zip"
    trades_zip = tmp_path / "aggTrades.zip"
    depth_zip = tmp_path / "bookDepth.zip"
    output = tmp_path / "features.csv"

    _write_zip_csv(
        book_zip,
        "BTCUSDT-bookTicker-2023-05-16.csv",
        [
            [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time",
            ],
            ["1", "100.0", "5.0", "100.2", "2.0", "1684195200000", "1684195200000"],
            ["2", "100.1", "4.0", "100.3", "3.0", "1684195201000", "1684195201000"],
            ["3", "100.4", "6.0", "100.6", "2.0", "1684195202000", "1684195202000"],
            ["4", "100.5", "6.0", "100.7", "2.0", "1684195203000", "1684195203000"],
        ],
    )
    _write_zip_csv(
        trades_zip,
        "BTCUSDT-aggTrades-2023-05-16.csv",
        [
            [
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "transact_time",
                "is_buyer_maker",
            ],
            ["1", "100.2", "1.0", "10", "10", "1684195200001", "false"],
        ],
    )
    _write_zip_csv(
        depth_zip,
        "BTCUSDT-bookDepth-2023-05-16.csv",
        [
            ["timestamp", "percentage", "depth", "notional"],
            ["2023-05-16 00:00:00", "-1", "10.0", "1000.0"],
            ["2023-05-16 00:00:00", "1", "5.0", "500.0"],
            ["2023-05-16 00:00:00", "-2", "20.0", "2000.0"],
            ["2023-05-16 00:00:00", "2", "10.0", "1000.0"],
            ["2023-05-16 00:00:00", "-5", "50.0", "5000.0"],
            ["2023-05-16 00:00:00", "5", "25.0", "2500.0"],
        ],
    )

    build_quote_trade_dataset(
        book_ticker_zip=book_zip,
        agg_trades_zip=trades_zip,
        book_depth_zip=depth_zip,
        output_csv=output,
        bucket_ms=1000,
        horizon_ms=1000,
        threshold="zero",
    )

    with output.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["depth_snapshot_age_ms"] == "1000"
    assert rows[0]["bid_depth_1pct"] == "10"
    assert rows[0]["ask_depth_1pct"] == "5"
    assert rows[0]["depth_imbalance_1pct"] == "0.333333333333"
    assert rows[0]["notional_imbalance_5pct"] == "0.333333333333"


def test_load_depth_snapshots_and_timestamp_parser(tmp_path: Path) -> None:
    depth_zip = tmp_path / "bookDepth.zip"
    _write_zip_csv(
        depth_zip,
        "BTCUSDT-bookDepth-2023-05-16.csv",
        [
            ["timestamp", "percentage", "depth", "notional"],
            ["2023-05-16 00:00:00", "-1", "10.0", "1000.0"],
            ["2023-05-16 00:00:00", "1", "5.0", "500.0"],
            ["2023-05-16 00:00:01.250", "-1", "11.0", "1100.0"],
            ["2023-05-16 00:00:01.250", "1", "6.0", "600.0"],
        ],
    )

    snapshots = load_depth_snapshots(depth_zip)

    assert parse_book_depth_timestamp_ms("2023-05-16 00:00:00") == 1684195200000
    assert parse_book_depth_timestamp_ms("2023-05-16 00:00:01.250") == 1684195201250
    assert len(snapshots) == 2
    assert snapshots[0].bid_depth_by_pct[1] == 10.0
    assert snapshots[0].ask_depth_by_pct[1] == 5.0


def test_quote_ofi_and_rolling_contexts() -> None:
    quotes = [
        QuoteBucket(
            bucket_start_ms=0,
            decision_time_ms=0,
            event_time=0,
            update_id=1,
            bid=100.0,
            ask=100.2,
            bid_qty=5.0,
            ask_qty=4.0,
        ),
        QuoteBucket(
            bucket_start_ms=1000,
            decision_time_ms=1000,
            event_time=1000,
            update_id=2,
            bid=100.0,
            ask=100.2,
            bid_qty=7.0,
            ask_qty=3.0,
        ),
        QuoteBucket(
            bucket_start_ms=2000,
            decision_time_ms=2000,
            event_time=2000,
            update_id=3,
            bid=100.1,
            ask=100.3,
            bid_qty=6.0,
            ask_qty=2.0,
        ),
    ]

    assert compute_quote_ofi(quotes[0], quotes[1]) == 3.0
    assert compute_quote_ofi(quotes[1], quotes[2]) == 9.0

    contexts = build_quote_contexts(quotes, rolling_window=2)

    assert contexts[0].quote_ofi == 0.0
    assert contexts[1].quote_ofi == 3.0
    assert contexts[2].quote_ofi_5 == 12.0
    assert contexts[2].mid_return_1 > 0
    assert contexts[2].realized_volatility_5 > 0


def _write_zip_csv(path: Path, inner_name: str, rows: list[list[str]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        content = "\n".join(",".join(row) for row in rows) + "\n"
        zf.writestr(inner_name, content)


def test_combine_feature_csvs(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    combined = tmp_path / "combined.csv"
    first.write_text("label,mid\n1,100\n")
    second.write_text("label,mid\n-1,99\n")

    combine_feature_csvs(
        [
            (first, {"source_symbol": "BTCUSDT", "source_date": "2023-05-16"}),
            (second, {"source_symbol": "BTCUSDT", "source_date": "2023-05-17"}),
        ],
        combined,
    )

    with combined.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["source_date"] == "2023-05-16"
    assert rows[1]["label"] == "-1"
