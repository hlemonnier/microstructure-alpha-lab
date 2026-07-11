from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from lob_forge.binance_vision import download_archive, iter_dates
from lob_forge.features import build_quote_trade_dataset, combine_feature_csvs
from lob_forge.holdout import sha256_file


FEATURE_BUILD_MARKER_VERSION = 1
COMBINED_FEATURE_MANIFEST_VERSION = 1
DEFAULT_SYMBOL_MIN_TICKS = {
    "BTCUSDT": 0.1,
    "ETHUSDT": 0.01,
    "BNBUSDT": 0.01,
    "SOLUSDT": 0.001,
    "XRPUSDT": 0.0001,
}


@dataclass(frozen=True)
class DailyBuildResult:
    date: str
    book_ticker_zip: Path
    agg_trades_zip: Path
    book_depth_zip: Path | None
    feature_csv: Path


def default_min_tick_for_symbol(symbol: str) -> float:
    return DEFAULT_SYMBOL_MIN_TICKS.get(symbol.upper(), 0.01)


def expected_feature_build_config(
    *,
    symbol: str,
    date_value: str,
    bucket_ms: int,
    horizon_ms: int,
    execution_latency_ms: int,
    threshold: str,
    min_tick: float,
    large_trade_notional: float,
    max_quote_buckets: int | None,
    with_book_depth: bool,
    execution_quote_resolution: str,
) -> dict[str, object]:
    return {
        "symbol": symbol.upper(),
        "date": date_value,
        "bucket_ms": bucket_ms,
        "horizon_ms": horizon_ms,
        "execution_latency_ms": execution_latency_ms,
        "threshold": threshold,
        "min_tick": min_tick,
        "large_trade_notional": large_trade_notional,
        "max_quote_buckets": max_quote_buckets,
        "with_book_depth": with_book_depth,
        "execution_quote_resolution": execution_quote_resolution,
    }


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
    execution_quote_resolution: str = "raw",
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
        build_config = expected_feature_build_config(
            symbol=symbol,
            date_value=date_value,
            bucket_ms=bucket_ms,
            horizon_ms=horizon_ms,
            execution_latency_ms=execution_latency_ms,
            threshold=threshold,
            min_tick=min_tick,
            large_trade_notional=large_trade_notional,
            max_quote_buckets=max_quote_buckets,
            with_book_depth=with_book_depth,
            execution_quote_resolution=execution_quote_resolution,
        )
        input_hashes = {
            "book_ticker_sha256": sha256_file(book_ticker_zip),
            "agg_trades_sha256": sha256_file(agg_trades_zip),
            "book_depth_sha256": sha256_file(book_depth_zip) if book_depth_zip is not None else None,
        }
        if not feature_build_is_current(
            feature_csv,
            done_marker,
            build_config=build_config,
            input_hashes=input_hashes,
        ):
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
                execution_quote_resolution=execution_quote_resolution,
            )
            write_feature_build_marker(
                feature_csv,
                done_marker,
                build_config=build_config,
                input_hashes=input_hashes,
            )
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
        write_combined_feature_manifest(
            combined_path,
            [result.feature_csv for result in results],
        )

    return results, combined_path


def feature_build_is_current(
    feature_csv: Path,
    done_marker: Path,
    *,
    build_config: dict[str, object],
    input_hashes: dict[str, str | None] | None,
) -> bool:
    """Return true only when a cached feature file is bound to this exact build.

    Legacy ``ok`` markers intentionally fail closed.  Reusing a feature CSV
    built with a different row cap, horizon, latency, source archive, or other
    feature parameter would invalidate every downstream research artifact.
    """
    if not done_marker.exists() or not feature_csv.exists() or feature_csv.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(done_marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if payload.get("marker_version") != FEATURE_BUILD_MARKER_VERSION:
        return False
    if payload.get("builder_source_sha256") != _feature_builder_source_sha256():
        return False
    if payload.get("build_config") != build_config:
        return False
    if input_hashes is not None and payload.get("input_hashes") != input_hashes:
        return False
    return payload.get("output_sha256") == sha256_file(feature_csv)


def write_feature_build_marker(
    feature_csv: Path,
    done_marker: Path,
    *,
    build_config: dict[str, object],
    input_hashes: dict[str, str | None],
) -> Path:
    if not feature_csv.exists() or feature_csv.stat().st_size <= 0:
        raise ValueError(f"cannot mark missing or empty feature CSV complete: {feature_csv}")
    payload = {
        "marker_version": FEATURE_BUILD_MARKER_VERSION,
        "builder_source_sha256": _feature_builder_source_sha256(),
        "build_config": build_config,
        "input_hashes": input_hashes,
        "output_sha256": sha256_file(feature_csv),
    }
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return done_marker


def combined_feature_manifest_path(combined_csv: Path | str) -> Path:
    return Path(f"{Path(combined_csv)}.manifest.json")


def write_combined_feature_manifest(
    combined_csv: Path,
    daily_feature_csvs: list[Path],
    *,
    manifest_path: Path | None = None,
) -> Path:
    if not combined_csv.exists() or combined_csv.stat().st_size <= 0:
        raise ValueError(f"cannot mark missing or empty combined feature CSV complete: {combined_csv}")
    if not daily_feature_csvs:
        raise ValueError("combined feature manifest requires at least one daily feature CSV")
    missing = [path for path in daily_feature_csvs if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise ValueError(f"cannot write combined feature manifest with missing inputs: {missing}")
    payload = {
        "manifest_version": COMBINED_FEATURE_MANIFEST_VERSION,
        "builder_source_sha256": _feature_builder_source_sha256(),
        "daily_features": [{"path": str(path), "sha256": sha256_file(path)} for path in daily_feature_csvs],
        "output_path": str(combined_csv),
        "output_sha256": sha256_file(combined_csv),
    }
    output = manifest_path or combined_feature_manifest_path(combined_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def combined_feature_is_current(
    combined_csv: Path,
    daily_feature_csvs: list[Path],
    *,
    manifest_path: Path | None = None,
) -> bool:
    manifest = manifest_path or combined_feature_manifest_path(combined_csv)
    if not combined_csv.exists() or combined_csv.stat().st_size <= 0 or not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if payload.get("manifest_version") != COMBINED_FEATURE_MANIFEST_VERSION:
        return False
    if payload.get("builder_source_sha256") != _feature_builder_source_sha256():
        return False
    expected_inputs = []
    for path in daily_feature_csvs:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        expected_inputs.append({"path": str(path), "sha256": sha256_file(path)})
    if payload.get("daily_features") != expected_inputs:
        return False
    if payload.get("output_path") != str(combined_csv):
        return False
    return payload.get("output_sha256") == sha256_file(combined_csv)


def _feature_builder_source_sha256() -> str:
    hasher = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("features.py")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()
