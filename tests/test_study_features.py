import json
from pathlib import Path

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
    _touch(symbol_dir / "BTCUSDT-2023-05-16-quote-trade-features.csv.done")
    _touch(symbol_dir / "BTCUSDT-2023-05-17-quote-trade-features.csv.done")
    _touch(symbol_dir / "BTCUSDT-2023-05-16_2023-05-17-combined-features.csv", payload="x\n")

    status = evaluate_expected_edge_feature_status(plan_path=plan_path)

    assert status.complete
    assert status.expected_feature_jobs == 1
    assert status.complete_feature_jobs == 1
    assert status.expected_daily_markers == 2
    assert status.present_daily_markers == 2


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
    assert status.present_daily_markers == 1
    assert len(status.missing_combined_files) == 2
    assert len(status.missing_daily_markers) == 3
    assert "complete=0" in formatted
    assert "missing_combined_files=2" in formatted


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
