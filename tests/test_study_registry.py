from __future__ import annotations

import csv
import json
from pathlib import Path

from lob_forge.experiments import (
    expected_feature_build_config,
    write_combined_feature_manifest,
    write_feature_build_marker,
)
from lob_forge.holdout import sha256_file
from lob_forge.study_plan import build_expected_edge_run_plan, write_expected_edge_run_plan
from lob_forge.study_provenance import write_expected_edge_provenance
from lob_forge.study_registry import (
    build_expected_edge_candidate_registry,
    main as study_registry_main,
    read_expected_edge_candidate_registry,
    write_expected_edge_candidate_registry,
    write_expected_edge_candidate_pvalues,
)


def test_expected_edge_candidate_registry_expands_run_plan_grid(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json", symbols=["BTCUSDT", "ETHUSDT"])

    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=tmp_path / "results")

    assert len(attempts) == 4
    assert {attempt.symbol for attempt in attempts} == {"BTCUSDT", "ETHUSDT"}
    assert {attempt.edge_threshold_bps for attempt in attempts} == {0.0, 0.1}
    assert {attempt.status for attempt in attempts} == {"planned"}
    assert all(attempt.config_sha256 for attempt in attempts)


def test_expected_edge_candidate_registry_marks_selected_artifact_rows(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _write_valid_artifacts(plan_path, result_dir)

    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    selected = next(attempt for attempt in attempts if attempt.edge_threshold_bps == 0.1)
    unselected = next(attempt for attempt in attempts if attempt.edge_threshold_bps == 0.0)

    assert selected.status == "selected_in_artifact"
    assert selected.selected
    assert selected.selected_fold_count == 2
    assert selected.fold_count == 2
    assert selected.validation_net_pnl == 4.0
    assert selected.test_net_pnl == -0.25
    assert selected.audit_acceptance_passed is False
    assert selected.audit_rejection_reasons == "total OOS net PnL is not positive"
    assert selected.audit_p_value is None
    assert selected.artifact_path.endswith("BTCUSDT_5000ms_fee_0p1_edge.csv")
    assert unselected.status == "evaluated_unselected"
    assert not unselected.selected
    assert unselected.audit_p_value is None


def test_expected_edge_candidate_pvalues_cover_completed_attempt_grid(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _write_valid_artifacts(plan_path, result_dir)

    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    pvalues_path = write_expected_edge_candidate_pvalues(attempts, result_dir / "pvalues.csv")

    with pvalues_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["procedure_sha256"] == rows[0]["config_sha256"]
    assert rows[0]["procedure_sha256"] not in {attempt.config_sha256 for attempt in attempts}
    assert {row["p_value"] for row in rows} == {"0.04"}
    assert rows[0]["metric"] == "validation_selected_procedure_fold_mean_p_value"
    assert rows[0]["inference_method"] == "legacy_hac_z_proxy"


def test_expected_edge_candidate_registry_cli_writes_jsonl(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json")
    output = tmp_path / "candidate_registry.jsonl"

    pvalues_output = tmp_path / "pvalues.csv"
    exit_code = study_registry_main(
        [
            "--plan",
            str(plan_path),
            "--result-dir",
            str(tmp_path / "results"),
            "--output",
            str(output),
            "--pvalues-output",
            str(pvalues_output),
        ]
    )
    attempts = read_expected_edge_candidate_registry(output)

    assert exit_code == 0
    assert len(attempts) == 2
    assert output.read_text().count("\n") == 2
    assert pvalues_output.exists()
    assert write_expected_edge_candidate_registry(attempts, tmp_path / "roundtrip.jsonl").exists()


def _write_plan(path: Path, *, symbols: list[str] | None = None) -> Path:
    plan = build_expected_edge_run_plan(
        profile="laptop_tiny",
        start_date="2023-05-16",
        end_date="2023-05-16",
        symbols=symbols or ["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.1],
        latency_ms=1000,
        max_quote_buckets=1200,
        train_size=900,
        validation_size=450,
        test_size=450,
        step_size=450,
        edge_streaming=True,
        min_ram_gb=4,
        max_csv_load_memory_gb=4,
        out_dir=str(path.parent / "results"),
        processed_root=str(path.parent / "processed"),
        raw_root=str(path.parent / "raw"),
        physical_ram_gb_value=16,
        with_book_depth=False,
        max_feature_build_memory_gb=4,
        edge_thresholds_bps=[0.0, 0.1],
    )
    return write_expected_edge_run_plan(plan, path)


def _write_valid_artifacts(plan_path: Path, result_dir: Path) -> None:
    result = result_dir / "BTCUSDT_5000ms_fee_0p1_edge.csv"
    audit = result_dir / "BTCUSDT_5000ms_fee_0p1_edge_audit.csv"
    plan = json.loads(plan_path.read_text())
    feature_dir = Path(plan["processed_root"]) / "btcusdt_5000ms_latency_1000"
    feature_dir.mkdir(parents=True, exist_ok=True)
    daily = feature_dir / "BTCUSDT-2023-05-16-quote-trade-features.csv"
    daily.write_text("event_time,label\n1,0\n")
    write_feature_build_marker(
        daily,
        daily.with_suffix(daily.suffix + ".done"),
        build_config=expected_feature_build_config(
            symbol="BTCUSDT",
            date_value="2023-05-16",
            bucket_ms=1000,
            horizon_ms=5000,
            execution_latency_ms=1000,
            threshold="half_spread",
            min_tick=0.1,
            large_trade_notional=10_000.0,
            max_quote_buckets=1200,
            with_book_depth=False,
            execution_quote_resolution="raw",
        ),
        input_hashes={"book_ticker_sha256": "book", "agg_trades_sha256": "trades", "book_depth_sha256": None},
    )
    feature = feature_dir / "BTCUSDT-2023-05-16_2023-05-16-combined-features.csv"
    holdout = result_dir / "holdout.json"
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
    with result.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "fold": "1",
                    "train_rows": "894",
                    "validation_rows": "444",
                    "test_rows": "450",
                    "purged_train_rows": "6",
                    "purged_validation_rows": "6",
                    "edge_threshold_bps": "0.1",
                    "val_net_pnl": "2.5",
                    "test_trades": "1",
                    "test_net_pnl": "-0.5",
                },
                {
                    "fold": "2",
                    "train_rows": "894",
                    "validation_rows": "444",
                    "test_rows": "450",
                    "purged_train_rows": "6",
                    "purged_validation_rows": "6",
                    "edge_threshold_bps": "0.1",
                    "val_net_pnl": "1.5",
                    "test_trades": "1",
                    "test_net_pnl": "0.25",
                },
            ]
        )
    with audit.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold_count",
                "total_test_rows",
                "total_test_trades",
                "total_test_net_pnl",
                "acceptance_passed",
                "rejection_reasons",
                "one_sided_p_value_mean_le_zero",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fold_count": "2",
                "total_test_rows": "900",
                "total_test_trades": "2",
                "total_test_net_pnl": "-0.25",
                "acceptance_passed": "0",
                "rejection_reasons": "total OOS net PnL is not positive",
                "one_sided_p_value_mean_le_zero": "0.04",
            }
        )
    write_expected_edge_provenance(
        plan_path=plan_path,
        feature_path=feature,
        holdout_manifest_path=holdout,
        result_path=result,
        audit_path=audit,
        symbol="BTCUSDT",
        horizon_ms=5000,
        taker_fee_bps=0.1,
        working_tree_dirty=False,
    )
