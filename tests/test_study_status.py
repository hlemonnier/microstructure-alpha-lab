import csv
import json
from pathlib import Path

from lob_forge.study_status import (
    evaluate_expected_edge_study_status,
    format_expected_edge_study_status,
)


def test_expected_edge_study_status_passes_complete_study(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    plan_path = _write_plan(
        result_dir,
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0, 0.05],
    )
    _write_edge_result(result_dir / "BTCUSDT_5000ms_fee_0_edge.csv")
    _write_edge_audit(result_dir / "BTCUSDT_5000ms_fee_0_edge_audit.csv")
    _write_edge_result(result_dir / "BTCUSDT_5000ms_fee_0p05_edge.csv")
    _write_edge_audit(result_dir / "BTCUSDT_5000ms_fee_0p05_edge_audit.csv")
    _write_pvalues(result_dir / "pvalues.csv")
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv")

    status = evaluate_expected_edge_study_status(plan_path=plan_path)

    assert status.complete
    assert status.expected_edge_jobs == 2
    assert status.present_result_files == 2
    assert status.present_audit_files == 2
    assert not status.missing_result_files
    assert not status.missing_audit_files
    assert status.artifact_verifier_passed


def test_expected_edge_study_status_reports_missing_outputs(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    plan_path = _write_plan(
        result_dir,
        symbols=["BTCUSDT", "ETHUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0],
    )
    _write_edge_result(result_dir / "BTCUSDT_5000ms_fee_0_edge.csv")

    status = evaluate_expected_edge_study_status(plan_path=plan_path)
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert status.expected_edge_jobs == 2
    assert status.present_result_files == 1
    assert status.present_audit_files == 0
    assert "ETHUSDT_5000ms_fee_0_edge.csv" in status.missing_result_files
    assert "BTCUSDT_5000ms_fee_0_edge_audit.csv" in status.missing_audit_files
    assert "pvalues.csv" in status.missing_required_files
    assert "complete=0" in formatted
    assert "missing_result=ETHUSDT_5000ms_fee_0_edge.csv" in formatted


def test_expected_edge_study_status_can_skip_pvalue_requirement(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    plan_path = _write_plan(
        result_dir,
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0],
    )
    _write_edge_result(result_dir / "BTCUSDT_5000ms_fee_0_edge.csv")
    _write_edge_audit(result_dir / "BTCUSDT_5000ms_fee_0_edge_audit.csv")

    status = evaluate_expected_edge_study_status(
        plan_path=plan_path,
        require_pvalues=False,
    )

    assert status.complete
    assert status.artifact_verifier_passed
    assert not status.missing_required_files


def test_expected_edge_study_status_can_require_min_audit_folds(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    plan_path = _write_plan(
        result_dir,
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0],
    )
    _write_edge_result(result_dir / "BTCUSDT_5000ms_fee_0_edge.csv")
    _write_edge_audit(result_dir / "BTCUSDT_5000ms_fee_0_edge_audit.csv")
    _write_pvalues(result_dir / "pvalues.csv")
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv")

    status = evaluate_expected_edge_study_status(
        plan_path=plan_path,
        min_audit_fold_count=20,
    )
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "fold_count 1 < required 20" in formatted


def _write_plan(
    result_dir: Path,
    *,
    symbols: list[str],
    horizons_ms: list[int],
    fees_bps: list[float],
) -> Path:
    path = result_dir / "run_plan.json"
    path.write_text(
        json.dumps(
            {
                "profile": "test",
                "out_dir": str(result_dir),
                "symbols": symbols,
                "horizons_ms": horizons_ms,
                "fees_bps": fees_bps,
            }
        )
    )
    return path


def _write_edge_result(path: Path) -> None:
    _write_csv(
        path,
        ["fold", "test_net_pnl"],
        [{"fold": "0", "test_net_pnl": "1.0"}],
    )


def _write_edge_audit(path: Path) -> None:
    _write_csv(
        path,
        [
            "inference_grain",
            "fold_count",
            "total_test_rows",
            "total_test_trades",
            "total_test_net_pnl",
            "acceptance_passed",
            "rejection_reasons",
        ],
        [
            {
                "inference_grain": "fold_summary",
                "fold_count": "1",
                "total_test_rows": "10",
                "total_test_trades": "2",
                "total_test_net_pnl": "1.0",
                "acceptance_passed": "1",
                "rejection_reasons": "",
            }
        ],
    )


def _write_pvalues(path: Path) -> None:
    _write_csv(
        path,
        ["hypothesis_id", "metric", "p_value"],
        [{"hypothesis_id": "BTCUSDT_5000ms_fee_0_edge", "metric": "fold_mean_net_pnl", "p_value": "0.1"}],
    )


def _write_pvalue_corrections(path: Path) -> None:
    _write_csv(
        path,
        ["hypothesis_id", "p_value", "bonferroni_p_value", "bh_adjusted_p_value", "bh_accept"],
        [
            {
                "hypothesis_id": "BTCUSDT_5000ms_fee_0_edge",
                "p_value": "0.1",
                "bonferroni_p_value": "0.1",
                "bh_adjusted_p_value": "0.1",
                "bh_accept": "0",
            }
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
