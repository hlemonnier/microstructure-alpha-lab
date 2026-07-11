import json
from pathlib import Path

from lob_forge.experiments import (
    expected_feature_build_config,
    write_combined_feature_manifest,
    write_feature_build_marker,
)
from lob_forge.study_features import (
    evaluate_expected_edge_feature_status,
    format_expected_edge_feature_status,
)


def test_expected_edge_feature_status_passes_complete_features(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    plan_path = _write_plan(
        tmp_path,
        processed_root=processed_root,
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        start_date="2023-05-16",
        end_date="2023-05-17",
        latency_ms=250,
    )
    symbol_dir = processed_root / "btcusdt_5000ms_latency_250"
    daily_features = [
        _write_valid_daily(symbol_dir, symbol="BTCUSDT", date_value=date_value, horizon_ms=5000, latency_ms=250)
        for date_value in ("2023-05-16", "2023-05-17")
    ]
    combined = symbol_dir / "BTCUSDT-2023-05-16_2023-05-17-combined-features.csv"
    _touch(combined, payload="x\n")
    write_combined_feature_manifest(combined, daily_features)

    status = evaluate_expected_edge_feature_status(plan_path=plan_path)

    assert status.complete
    assert status.expected_feature_jobs == 1
    assert status.complete_feature_jobs == 1
    assert status.expected_daily_markers == 2
    assert status.present_daily_markers == 2
    assert not status.invalid_daily_markers
    assert not status.invalid_combined_files


def test_expected_edge_feature_status_reports_missing_coverage(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    plan_path = _write_plan(
        tmp_path,
        processed_root=processed_root,
        symbols=["BTCUSDT", "ETHUSDT"],
        horizons_ms=[5000],
        start_date="2023-05-16",
        end_date="2023-05-17",
        latency_ms=1000,
    )
    _touch(processed_root / "btcusdt_5000ms_latency_1000" / "BTCUSDT-2023-05-16-quote-trade-features.csv.done")

    status = evaluate_expected_edge_feature_status(plan_path=plan_path)
    formatted = format_expected_edge_feature_status(status)

    assert not status.complete
    assert status.expected_feature_jobs == 2
    assert status.complete_feature_jobs == 0
    assert status.expected_daily_markers == 4
    assert status.present_daily_markers == 0
    assert len(status.missing_combined_files) == 2
    assert len(status.missing_daily_markers) == 3
    assert len(status.invalid_daily_markers) == 1
    assert "complete=0" in formatted
    assert "missing_combined_files=2" in formatted


def test_expected_edge_feature_status_rejects_tampered_combined_file(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    plan_path = _write_plan(
        tmp_path,
        processed_root=processed_root,
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        start_date="2023-05-16",
        end_date="2023-05-16",
        latency_ms=1000,
    )
    symbol_dir = processed_root / "btcusdt_5000ms_latency_1000"
    daily = _write_valid_daily(
        symbol_dir,
        symbol="BTCUSDT",
        date_value="2023-05-16",
        horizon_ms=5000,
        latency_ms=1000,
    )
    combined = symbol_dir / "BTCUSDT-2023-05-16_2023-05-16-combined-features.csv"
    _touch(combined, payload="original\n")
    write_combined_feature_manifest(combined, [daily])
    combined.write_text("tampered\n")

    status = evaluate_expected_edge_feature_status(plan_path=plan_path)

    assert not status.complete
    assert status.invalid_combined_files == [str(combined)]


def _write_plan(
    root: Path,
    *,
    processed_root: Path,
    symbols: list[str],
    horizons_ms: list[int],
    start_date: str,
    end_date: str,
    latency_ms: int,
) -> Path:
    path = root / "run_plan.json"
    path.write_text(
        json.dumps(
            {
                "profile": "test",
                "processed_root": str(processed_root),
                "symbols": symbols,
                "horizons_ms": horizons_ms,
                "start_date": start_date,
                "end_date": end_date,
                "latency_ms": latency_ms,
            }
        )
    )
    return path


def _touch(path: Path, *, payload: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _write_valid_daily(
    symbol_dir: Path,
    *,
    symbol: str,
    date_value: str,
    horizon_ms: int,
    latency_ms: int,
) -> Path:
    feature = symbol_dir / f"{symbol}-{date_value}-quote-trade-features.csv"
    _touch(feature, payload="event_time,label\n1,0\n")
    write_feature_build_marker(
        feature,
        feature.with_suffix(feature.suffix + ".done"),
        build_config=expected_feature_build_config(
            symbol=symbol,
            date_value=date_value,
            bucket_ms=1000,
            horizon_ms=horizon_ms,
            execution_latency_ms=latency_ms,
            threshold="half_spread",
            min_tick=0.1,
            large_trade_notional=10_000.0,
            max_quote_buckets=None,
            with_book_depth=True,
            execution_quote_resolution="raw",
        ),
        input_hashes={"book_ticker_sha256": "book", "agg_trades_sha256": "trades", "book_depth_sha256": None},
    )
    return feature
