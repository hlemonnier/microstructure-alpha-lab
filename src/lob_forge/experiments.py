from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lob_forge.binance_vision import download_archive, iter_dates
from lob_forge.features import build_quote_trade_dataset, combine_feature_csvs


@dataclass(frozen=True)
class DailyBuildResult:
    date: str
    book_ticker_zip: Path
    agg_trades_zip: Path
    book_depth_zip: Path | None
    feature_csv: Path


def build_daily_feature_range(
    *,
    market: str,
    symbol: str,
    start: str,
    end: str,
    raw_root: Path | str,
    output_dir: Path | str,
    combined_output: Path | str | None = None,
    bucket_ms: int = 1000,
    horizon_ms: int = 1000,
    execution_latency_ms: int = 0,
    threshold: str = "half_spread",
    min_tick: float = 0.0,
    large_trade_notional: float = 10_000.0,
    max_quote_buckets: int | None = None,
    with_book_depth: bool = False,
    overwrite_download: bool = False,
    verify_checksum: bool = True,
    max_feature_build_memory_gb: float = 0.0,
    memory_estimate_multiplier: float = 12.0,
) -> tuple[list[DailyBuildResult], Path | None]:
    """Download/build feature samples for a daily date range."""
    results: list[DailyBuildResult] = []
    output_root = Path(output_dir)
    for date_value in iter_dates(start, end):
        book_ticker_zip = download_archive(
            market=market,
            frequency="daily",
            dataset="bookTicker",
            symbol=symbol,
            date_value=date_value,
            root=raw_root,
            overwrite=overwrite_download,
            verify_checksum=verify_checksum,
        )
        agg_trades_zip = download_archive(
            market=market,
            frequency="daily",
            dataset="aggTrades",
            symbol=symbol,
            date_value=date_value,
            root=raw_root,
            overwrite=overwrite_download,
            verify_checksum=verify_checksum,
        )
        book_depth_zip: Path | None = None
        if with_book_depth:
            book_depth_zip = download_archive(
                market=market,
                frequency="daily",
                dataset="bookDepth",
                symbol=symbol,
                date_value=date_value,
                root=raw_root,
                overwrite=overwrite_download,
                verify_checksum=verify_checksum,
            )
        feature_csv = output_root / f"{symbol.upper()}-{date_value}-quote-trade-features.csv"
        done_marker = feature_csv.with_suffix(feature_csv.suffix + ".done")
        if not _feature_build_done(feature_csv, done_marker):
            build_quote_trade_dataset(
                book_ticker_zip=book_ticker_zip,
                agg_trades_zip=agg_trades_zip,
                book_depth_zip=book_depth_zip,
                output_csv=feature_csv,
                bucket_ms=bucket_ms,
                horizon_ms=horizon_ms,
                execution_latency_ms=execution_latency_ms,
                threshold=threshold,
                min_tick=min_tick,
                large_trade_notional=large_trade_notional,
                max_quote_buckets=max_quote_buckets,
                max_feature_build_memory_gb=max_feature_build_memory_gb,
                memory_estimate_multiplier=memory_estimate_multiplier,
            )
            done_marker.write_text("ok\n")
        results.append(
            DailyBuildResult(
                date=date_value,
                book_ticker_zip=book_ticker_zip,
                agg_trades_zip=agg_trades_zip,
                book_depth_zip=book_depth_zip,
                feature_csv=feature_csv,
            )
        )

    combined_path: Path | None = None
    if combined_output is not None:
        combined_path = combine_feature_csvs(
            [
                (
                    result.feature_csv,
                    {"source_symbol": symbol.upper(), "source_date": result.date},
                )
                for result in results
            ],
            combined_output,
        )

    return results, combined_path


def _feature_build_done(feature_csv: Path, done_marker: Path) -> bool:
    return done_marker.exists() and feature_csv.exists() and feature_csv.stat().st_size > 0
