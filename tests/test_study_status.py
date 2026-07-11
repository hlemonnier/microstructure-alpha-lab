import csv
import json
from pathlib import Path

from lob_forge.experiments import (
    expected_feature_build_config,
    write_combined_feature_manifest,
    write_feature_build_marker,
)
from lob_forge.holdout import sha256_file
from lob_forge.study_status import (
    evaluate_expected_edge_study_status,
    format_expected_edge_study_status,
)
from lob_forge.study_provenance import write_expected_edge_provenance
from lob_forge.study_registry import build_expected_edge_candidate_registry, write_expected_edge_candidate_registry
from lob_forge.study_registry import write_expected_edge_candidate_pvalues


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
    _write_candidate_registry(plan_path, result_dir)
    _write_pvalues_from_registry(plan_path, result_dir)
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", result_dir / "pvalues.csv")

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
    assert "candidate_registry.jsonl" in status.missing_required_files
    assert "pvalues.csv" in status.missing_required_files
    assert "complete=0" in formatted
    assert "missing_result=ETHUSDT_5000ms_fee_0_edge.csv" in formatted


def test_expected_edge_study_status_rejects_candidate_corrections_without_config_hash(tmp_path: Path) -> None:
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
    _write_candidate_registry(plan_path, result_dir)
    _write_pvalues_from_registry(plan_path, result_dir)
    _write_artifact_level_pvalue_corrections(result_dir / "pvalue_corrections.csv")

    status = evaluate_expected_edge_study_status(plan_path=plan_path)
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "pvalue_corrections.csv: missing candidate-link columns ['procedure_sha256']" in formatted


def test_expected_edge_study_status_rejects_pvalue_that_differs_from_audit(tmp_path: Path) -> None:
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
    _write_candidate_registry(plan_path, result_dir)
    _write_pvalues_from_registry(plan_path, result_dir)
    pvalues_path = result_dir / "pvalues.csv"
    with pvalues_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["p_value"] = "0.2"
    _write_csv(pvalues_path, fieldnames, rows)
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", pvalues_path)

    status = evaluate_expected_edge_study_status(plan_path=plan_path)

    assert not status.complete
    assert any("p_value does not match procedure audit" in error for error in status.artifact_verifier_errors)


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
    _write_candidate_registry(plan_path, result_dir)

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
    _write_candidate_registry(plan_path, result_dir)
    _write_pvalues_from_registry(plan_path, result_dir)
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", result_dir / "pvalues.csv")

    status = evaluate_expected_edge_study_status(
        plan_path=plan_path,
        min_audit_fold_count=20,
    )
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "fold_count 1 < required 20" in formatted


def test_expected_edge_study_status_rejects_stale_planned_candidate_registry(tmp_path: Path) -> None:
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
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", result_dir / "pvalues.csv")
    (result_dir / "candidate_registry.jsonl").write_text(
        json.dumps(
            {
                "run_id": "fixture",
                "family": "expected_edge_threshold_grid",
                "symbol": "BTCUSDT",
                "horizon_ms": 5000,
                "taker_fee_bps": 0.0,
                "latency_ms": 0,
                "edge_threshold_bps": 0.0,
                "model_class": "ridge_expected_edge",
                "feature_set": "default_microstructure",
                "selection_metric": "validation_net_pnl",
                "train_size": 0,
                "validation_size": 0,
                "test_size": 0,
                "step_size": 0,
                "status": "planned",
                "selected": False,
                "selected_fold_count": 0,
                "fold_count": 0,
                "validation_net_pnl": None,
                "test_net_pnl": None,
                "audit_acceptance_passed": None,
                "audit_rejection_reasons": "",
                "artifact_path": "",
                "audit_path": "",
                "failure_reason": "",
                "config_sha256": "a" * 64,
                "config_json": "{}",
            },
            sort_keys=True,
        )
        + "\n"
    )

    status = evaluate_expected_edge_study_status(plan_path=plan_path)
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "candidate_registry.jsonl: status_planned=1" in formatted
    assert "candidate_registry.jsonl: selected_candidates=0" in formatted


def test_expected_edge_study_status_rejects_registry_missing_plan_candidates(tmp_path: Path) -> None:
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
    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    write_expected_edge_candidate_registry(attempts[:1], result_dir / "candidate_registry.jsonl")
    write_expected_edge_candidate_pvalues(attempts[:1], result_dir / "pvalues.csv")
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", result_dir / "pvalues.csv")

    status = evaluate_expected_edge_study_status(plan_path=plan_path)
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "candidate_registry.jsonl: missing expected candidate config_sha256 rows" in formatted


def test_expected_edge_study_status_rejects_registry_stale_artifact_references(tmp_path: Path) -> None:
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
    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    payload = attempts[0].__dict__.copy()
    payload["artifact_path"] = "results/stale.csv"
    with (result_dir / "candidate_registry.jsonl").open("w") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        for attempt in attempts[1:]:
            handle.write(json.dumps(attempt.__dict__, sort_keys=True) + "\n")
    _write_pvalues_from_registry(plan_path, result_dir)
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", result_dir / "pvalues.csv")

    status = evaluate_expected_edge_study_status(plan_path=plan_path)
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "candidate_registry.jsonl: artifact_path mismatch" in formatted


def test_expected_edge_study_status_rejects_registry_stale_derived_fields(tmp_path: Path) -> None:
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
    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    payload = attempts[0].__dict__.copy()
    payload["selected_fold_count"] = 99
    payload["validation_net_pnl"] = -123.0
    payload["audit_p_value"] = 0.99
    with (result_dir / "candidate_registry.jsonl").open("w") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        for attempt in attempts[1:]:
            handle.write(json.dumps(attempt.__dict__, sort_keys=True) + "\n")
    _write_pvalues_from_registry(plan_path, result_dir)
    _write_pvalue_corrections(result_dir / "pvalue_corrections.csv", result_dir / "pvalues.csv")

    status = evaluate_expected_edge_study_status(plan_path=plan_path)
    formatted = format_expected_edge_study_status(status)

    assert not status.complete
    assert not status.artifact_verifier_passed
    assert "candidate_registry.jsonl: selected_fold_count mismatch" in formatted
    assert "candidate_registry.jsonl: validation_net_pnl mismatch" in formatted
    assert "candidate_registry.jsonl: audit_p_value mismatch" in formatted


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
                "start_date": "2023-05-16",
                "end_date": "2023-05-16",
                "symbols": symbols,
                "horizons_ms": horizons_ms,
                "fees_bps": fees_bps,
                "latency_ms": 1000,
                "bucket_ms": 1000,
                "execution_quote_resolution": "raw",
                "max_quote_buckets": 1200,
                "with_book_depth": False,
                "feature_threshold": "half_spread",
                "large_trade_notional": 10000.0,
                "symbol_min_ticks": {symbol: 0.1 if symbol == "BTCUSDT" else 0.01 for symbol in symbols},
                "train_size": 900,
                "validation_size": 450,
                "test_size": 450,
                "step_size": 450,
                "edge_streaming": True,
                "edge_thresholds_bps": [0.0, 0.1],
                "model_classes": ["ridge_expected_edge"],
                "feature_sets": ["default_microstructure"],
                "selection_metric": "validation_net_pnl",
            }
        )
    )
    return path


def _write_edge_result(path: Path) -> None:
    _write_csv(
        path,
        [
            "fold",
            "train_rows",
            "validation_rows",
            "test_rows",
            "purged_train_rows",
            "purged_validation_rows",
            "edge_threshold_bps",
            "val_net_pnl",
            "test_trades",
            "test_net_pnl",
        ],
        [
            {
                "fold": "1",
                "train_rows": "894",
                "validation_rows": "444",
                "test_rows": "450",
                "purged_train_rows": "6",
                "purged_validation_rows": "6",
                "edge_threshold_bps": "0.0",
                "val_net_pnl": "2.0",
                "test_trades": "2",
                "test_net_pnl": "1.0",
            }
        ],
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
            "one_sided_p_value_mean_le_zero",
            "acceptance_passed",
            "rejection_reasons",
        ],
        [
            {
                "inference_grain": "fold_summary",
                "fold_count": "1",
                "total_test_rows": "450",
                "total_test_trades": "2",
                "total_test_net_pnl": "1.0",
                "one_sided_p_value_mean_le_zero": "0.1",
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


def _write_pvalue_corrections(path: Path, pvalues_path: Path) -> None:
    with pvalues_path.open(newline="") as handle:
        pvalue_rows = list(csv.DictReader(handle))
    metadata_columns = [
        column
        for column in (pvalue_rows[0].keys() if pvalue_rows else [])
        if column not in {"hypothesis_id", "metric", "p_value"}
    ]
    _write_csv(
        path,
        [
            "hypothesis_id",
            "p_value",
            "bonferroni_p_value",
            "bh_adjusted_p_value",
            "bh_accept",
            *metadata_columns,
        ],
        [
            {
                "hypothesis_id": row["hypothesis_id"],
                "p_value": row["p_value"],
                "bonferroni_p_value": row["p_value"],
                "bh_adjusted_p_value": row["p_value"],
                "bh_accept": "0",
                **{column: row.get(column, "") for column in metadata_columns},
            }
            for row in pvalue_rows
        ],
    )


def _write_artifact_level_pvalue_corrections(path: Path) -> None:
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


def _write_candidate_registry(plan_path: Path, result_dir: Path) -> None:
    _write_all_provenance(plan_path, result_dir)
    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    write_expected_edge_candidate_registry(attempts, result_dir / "candidate_registry.jsonl")


def _write_pvalues_from_registry(plan_path: Path, result_dir: Path) -> None:
    _write_all_provenance(plan_path, result_dir)
    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    write_expected_edge_candidate_pvalues(attempts, result_dir / "pvalues.csv")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_all_provenance(plan_path: Path, result_dir: Path) -> None:
    plan = json.loads(plan_path.read_text())
    holdout = result_dir / "fixture_holdout.json"
    for symbol in plan.get("symbols", []):
        for horizon_ms in plan.get("horizons_ms", []):
            feature_dir = result_dir / f"{str(symbol).lower()}_{horizon_ms}ms_latency_{plan['latency_ms']}"
            feature_dir.mkdir(parents=True, exist_ok=True)
            daily = feature_dir / f"{symbol}-2023-05-16-quote-trade-features.csv"
            daily.write_text("event_time,label\n1,0\n")
            write_feature_build_marker(
                daily,
                daily.with_suffix(daily.suffix + ".done"),
                build_config=expected_feature_build_config(
                    symbol=str(symbol),
                    date_value="2023-05-16",
                    bucket_ms=int(plan["bucket_ms"]),
                    horizon_ms=int(horizon_ms),
                    execution_latency_ms=int(plan["latency_ms"]),
                    threshold=str(plan["feature_threshold"]),
                    min_tick=float(plan["symbol_min_ticks"][symbol]),
                    large_trade_notional=float(plan["large_trade_notional"]),
                    max_quote_buckets=int(plan["max_quote_buckets"]),
                    with_book_depth=bool(plan["with_book_depth"]),
                    execution_quote_resolution=str(plan["execution_quote_resolution"]),
                ),
                input_hashes={
                    "book_ticker_sha256": "book",
                    "agg_trades_sha256": "trades",
                    "book_depth_sha256": None,
                },
            )
            feature = feature_dir / f"{symbol}-2023-05-16_2023-05-16-combined-features.csv"
            feature.write_text("event_time,label\n1,0\n")
            write_combined_feature_manifest(feature, [daily])
            feature_sha256 = sha256_file(feature)
            holdout.write_text(
                json.dumps(
                    {
                        "split_column": "source_date",
                        "holdout_values": ["2023-05-16"],
                        "source_sha256": feature_sha256,
                        "dataset_fingerprint": feature_sha256,
                    }
                )
                + "\n"
            )
            for fee_bps in plan.get("fees_bps", []):
                fee_token = (
                    str(int(fee_bps)) if float(fee_bps).is_integer() else format(float(fee_bps), "g").replace(".", "p")
                )
                result = result_dir / f"{symbol}_{horizon_ms}ms_fee_{fee_token}_edge.csv"
                audit = result_dir / f"{symbol}_{horizon_ms}ms_fee_{fee_token}_edge_audit.csv"
                if not result.exists() or not audit.exists():
                    continue
                write_expected_edge_provenance(
                    plan_path=plan_path,
                    feature_path=feature,
                    holdout_manifest_path=holdout,
                    result_path=result,
                    audit_path=audit,
                    symbol=symbol,
                    horizon_ms=int(horizon_ms),
                    taker_fee_bps=float(fee_bps),
                    working_tree_dirty=False,
                )
