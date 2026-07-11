import pytest

from lob_forge.edge_model import DEFAULT_EDGE_THRESHOLDS_BPS
from lob_forge.study_plan import build_expected_edge_run_plan, format_expected_edge_run_plan


def test_laptop_tiny_plan_disables_depth_and_reduces_archives() -> None:
    plan = build_expected_edge_run_plan(
        profile="laptop_tiny",
        start_date="2023-05-16",
        end_date="2023-05-17",
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0, 0.1, 0.5],
        latency_ms=1000,
        max_quote_buckets=1200,
        train_size=900,
        validation_size=450,
        test_size=450,
        step_size=450,
        edge_streaming=True,
        min_ram_gb=4,
        max_csv_load_memory_gb=4,
        out_dir="results/tiny",
        processed_root="data/processed/tiny",
        raw_root="data/raw",
        physical_ram_gb_value=16,
        with_book_depth=False,
        max_feature_build_memory_gb=4,
    )

    assert plan.risk_level == "laptop_safe"
    assert plan.total_days == 2
    assert plan.binance_daily_archives == 4
    assert plan.edge_eval_jobs == 3
    assert plan.edge_thresholds_bps == list(DEFAULT_EDGE_THRESHOLDS_BPS)
    assert plan.threshold_candidate_attempts == plan.edge_eval_jobs * len(DEFAULT_EDGE_THRESHOLDS_BPS)
    assert plan.model_classes == ["ridge_expected_edge"]
    assert plan.feature_sets == ["default_microstructure"]
    assert plan.selection_metric == "validation_net_pnl"
    assert plan.max_combined_rows_per_symbol_horizon == 2400
    assert not plan.with_book_depth
    assert any("proof-of-pipeline" in recommendation for recommendation in plan.recommendations)


def test_laptop_quick_plan_counts_archives_and_jobs() -> None:
    plan = build_expected_edge_run_plan(
        profile="laptop_quick",
        start_date="2023-05-16",
        end_date="2023-05-22",
        symbols=["BTCUSDT", "ETHUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0, 0.05, 0.1, 0.25, 0.5],
        latency_ms=1000,
        max_quote_buckets=3600,
        train_size=7200,
        validation_size=3600,
        test_size=3600,
        step_size=3600,
        edge_streaming=True,
        min_ram_gb=8,
        max_csv_load_memory_gb=8,
        out_dir="results/quick",
        processed_root="data/processed/quick",
        raw_root="data/raw",
        physical_ram_gb_value=16,
        edge_thresholds_bps=[0.0, 0.25],
        model_classes=["ridge_expected_edge", "lasso_expected_edge"],
        feature_sets=["default_microstructure", "depth_bands"],
    )

    assert plan.risk_level == "laptop_safe"
    assert plan.total_days == 7
    assert plan.binance_daily_archives == 42
    assert plan.feature_build_jobs == 2
    assert plan.edge_eval_jobs == 10
    assert plan.threshold_candidate_attempts == 10 * 2 * 2 * 2
    assert plan.max_combined_rows_per_symbol_horizon == 25_200
    assert plan.can_start_on_current_machine


def test_cloud_full_plan_rejects_sixteen_gb_machine() -> None:
    plan = build_expected_edge_run_plan(
        profile="cloud_full",
        start_date="2023-05-16",
        end_date="2023-07-14",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        horizons_ms=[5000, 10000],
        fees_bps=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        latency_ms=1000,
        max_quote_buckets=None,
        train_size=72000,
        validation_size=18000,
        test_size=18000,
        step_size=18000,
        edge_streaming=True,
        min_ram_gb=64,
        max_csv_load_memory_gb=48,
        out_dir="results/full",
        processed_root="data/processed/full",
        raw_root="data/raw",
        physical_ram_gb_value=16,
    )

    assert plan.risk_level == "cloud_full_heavy"
    assert plan.total_days == 60
    assert plan.binance_daily_archives == 900
    assert plan.feature_build_jobs == 10
    assert plan.edge_eval_jobs == 80
    assert plan.max_combined_rows_per_symbol_horizon is None
    assert not plan.ram_preflight_passed
    assert not plan.can_start_on_current_machine
    assert any("16.0GB RAM" in recommendation for recommendation in plan.recommendations)


def test_study_plan_text_surfaces_risk_and_uncapped_state() -> None:
    plan = build_expected_edge_run_plan(
        profile="full",
        start_date="2023-05-16",
        end_date="2023-05-16",
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0],
        latency_ms=1000,
        max_quote_buckets=None,
        train_size=1,
        validation_size=1,
        test_size=1,
        step_size=1,
        edge_streaming=False,
        min_ram_gb=64,
        max_csv_load_memory_gb=48,
        out_dir="results/full",
        processed_root="data/processed/full",
        raw_root="data/raw",
        physical_ram_gb_value=128,
    )

    formatted = format_expected_edge_run_plan(plan)

    assert "profile=cloud_full" in formatted
    assert "risk=cloud_full_heavy" in formatted
    assert "threshold_candidate_attempts=" in formatted
    assert "edge_thresholds_bps=" in formatted
    assert "max_combined_rows_per_symbol_horizon=uncapped" in formatted
    assert "EDGE_STREAMING=0" in formatted


def test_study_plan_rejects_overlapping_confirmatory_test_windows() -> None:
    with pytest.raises(ValueError, match="OOS windows do not overlap"):
        build_expected_edge_run_plan(
            profile="laptop_tiny",
            start_date="2023-05-16",
            end_date="2023-05-17",
            symbols=["BTCUSDT"],
            horizons_ms=[5000],
            fees_bps=[0.0],
            latency_ms=1000,
            max_quote_buckets=1200,
            train_size=900,
            validation_size=450,
            test_size=450,
            step_size=225,
            edge_streaming=True,
            min_ram_gb=4,
            max_csv_load_memory_gb=4,
            out_dir="results/tiny",
            processed_root="data/processed/tiny",
            raw_root="data/raw",
            physical_ram_gb_value=16,
        )
