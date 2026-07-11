import csv
import json
from pathlib import Path

from lob_forge.experiments import (
    expected_feature_build_config,
    write_combined_feature_manifest,
    write_feature_build_marker,
)
from lob_forge.holdout import sha256_file
from lob_forge.study_provenance import (
    provenance_path_for,
    verify_expected_edge_provenance,
    verify_recorded_expected_edge_provenance,
    write_expected_edge_provenance,
)


def test_expected_edge_provenance_binds_plan_sources_and_artifacts(tmp_path: Path) -> None:
    paths = _write_study_fixture(tmp_path)
    provenance = write_expected_edge_provenance(**paths, working_tree_dirty=False)

    report = verify_expected_edge_provenance(
        provenance,
        plan_path=paths["plan_path"],
        result_path=paths["result_path"],
        audit_path=paths["audit_path"],
        expected_symbol="BTCUSDT",
        expected_horizon_ms=5000,
        expected_taker_fee_bps=0.1,
        require_source_files=True,
    )

    assert provenance == provenance_path_for(paths["result_path"])
    assert report.passed
    assert verify_recorded_expected_edge_provenance(
        provenance,
        result_path=paths["result_path"],
        audit_path=paths["audit_path"],
        require_source_files=True,
    ).passed

    Path(paths["result_path"]).write_text(Path(paths["result_path"]).read_text().replace("1.0", "2.0"))
    report = verify_expected_edge_provenance(
        provenance,
        plan_path=paths["plan_path"],
        result_path=paths["result_path"],
        audit_path=paths["audit_path"],
        expected_symbol="BTCUSDT",
        expected_horizon_ms=5000,
        expected_taker_fee_bps=0.1,
    )
    assert not report.passed
    assert "result SHA-256 mismatch" in report.errors


def test_provenance_rejects_fold_sizes_from_an_old_plan(tmp_path: Path) -> None:
    paths = _write_study_fixture(tmp_path)
    plan = json.loads(Path(paths["plan_path"]).read_text())
    plan["train_size"] = 1800
    Path(paths["plan_path"]).write_text(json.dumps(plan))

    try:
        write_expected_edge_provenance(**paths, working_tree_dirty=False)
    except ValueError as exc:
        assert "raw train rows 900 != planned 1800" in str(exc)
    else:
        raise AssertionError("stale result unexpectedly received valid provenance")


def test_provenance_rejects_result_generated_from_dirty_worktree(tmp_path: Path) -> None:
    paths = _write_study_fixture(tmp_path)
    provenance = write_expected_edge_provenance(**paths, working_tree_dirty=True)

    report = verify_recorded_expected_edge_provenance(
        provenance,
        result_path=paths["result_path"],
        audit_path=paths["audit_path"],
    )

    assert not report.passed
    assert "result was generated from a dirty working tree" in report.errors


def test_provenance_rejects_holdout_values_not_declared_by_plan(tmp_path: Path) -> None:
    paths = _write_study_fixture(tmp_path)
    holdout = Path(paths["holdout_manifest_path"])
    payload = json.loads(holdout.read_text())
    payload["holdout_values"] = ["2023-05-16"]
    holdout.write_text(json.dumps(payload))

    try:
        write_expected_edge_provenance(**paths, working_tree_dirty=False)
    except ValueError as exc:
        assert "holdout_values" in str(exc)
    else:
        raise AssertionError("unplanned holdout values unexpectedly received valid provenance")


def _write_study_fixture(tmp_path: Path) -> dict[str, object]:
    plan = tmp_path / "run_plan.json"
    feature_dir = tmp_path / "processed" / "btcusdt_5000ms_latency_1000"
    feature_dir.mkdir(parents=True)
    feature = feature_dir / "BTCUSDT-2023-05-16_2023-05-17-combined-features.csv"
    holdout = tmp_path / "holdout.json"
    result = tmp_path / "BTCUSDT_5000ms_fee_0p1_edge.csv"
    audit = tmp_path / "BTCUSDT_5000ms_fee_0p1_edge_audit.csv"
    plan.write_text(
        json.dumps(
            {
                "profile": "test",
                "start_date": "2023-05-16",
                "end_date": "2023-05-17",
                "symbols": ["BTCUSDT"],
                "horizons_ms": [5000],
                "fees_bps": [0.1],
                "latency_ms": 1000,
                "bucket_ms": 1000,
                "execution_quote_resolution": "raw",
                "max_quote_buckets": 1200,
                "with_book_depth": False,
                "feature_threshold": "half_spread",
                "large_trade_notional": 10000.0,
                "symbol_min_ticks": {"BTCUSDT": 0.1},
                "train_size": 900,
                "validation_size": 450,
                "test_size": 450,
                "step_size": 450,
                "edge_streaming": True,
                "edge_thresholds_bps": [0.0, 0.1],
                "model_classes": ["ridge_expected_edge"],
                "feature_sets": ["default_microstructure"],
                "selection_metric": "validation_net_pnl",
            },
            sort_keys=True,
        )
    )
    daily_features = []
    for date_value in ("2023-05-16", "2023-05-17"):
        daily = feature_dir / f"BTCUSDT-{date_value}-quote-trade-features.csv"
        daily.write_text("event_time,label\n1,0\n")
        write_feature_build_marker(
            daily,
            daily.with_suffix(daily.suffix + ".done"),
            build_config=expected_feature_build_config(
                symbol="BTCUSDT",
                date_value=date_value,
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
            input_hashes={
                "book_ticker_sha256": "book",
                "agg_trades_sha256": "trades",
                "book_depth_sha256": None,
            },
        )
        daily_features.append(daily)
    feature.write_text("event_time,label\n1,0\n")
    write_combined_feature_manifest(feature, daily_features)
    feature_sha256 = sha256_file(feature)
    holdout.write_text(
        json.dumps(
            {
                "split_column": "source_date",
                "holdout_values": ["2023-05-17"],
                "source_sha256": feature_sha256,
                "dataset_fingerprint": feature_sha256,
            }
        )
        + "\n"
    )
    _write_csv(
        result,
        [
            "fold",
            "train_rows",
            "validation_rows",
            "test_rows",
            "purged_train_rows",
            "purged_validation_rows",
            "edge_threshold_bps",
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
                "edge_threshold_bps": "0.1",
                "test_trades": "2",
                "test_net_pnl": "1.0",
            }
        ],
    )
    _write_csv(
        audit,
        ["fold_count", "total_test_rows", "total_test_trades", "total_test_net_pnl"],
        [{"fold_count": "1", "total_test_rows": "450", "total_test_trades": "2", "total_test_net_pnl": "1.0"}],
    )
    return {
        "plan_path": plan,
        "feature_path": feature,
        "holdout_manifest_path": holdout,
        "result_path": result,
        "audit_path": audit,
        "symbol": "BTCUSDT",
        "horizon_ms": 5000,
        "taker_fee_bps": 0.1,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
