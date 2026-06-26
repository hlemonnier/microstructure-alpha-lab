import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local16_profile_requires_confirmation_for_non_plan_run() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "STUDY_PROFILE": "local16_60day",
            "PREFLIGHT": "0",
        }
    )
    env.pop("CONFIRM_LOCAL16", None)

    result = subprocess.run(
        ["bash", "scripts/run_60day_expected_edge_study.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "CONFIRM_LOCAL16=1" in result.stderr


def test_local16_plan_only_stays_memory_scaled_without_confirmation(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "STUDY_PROFILE": "local16_60day",
            "PLAN_ONLY": "1",
            "PREFLIGHT": "0",
            "OUT_DIR": str(tmp_path / "results"),
            "PROCESSED_ROOT": str(tmp_path / "processed"),
            "RAW_ROOT": str(tmp_path / "raw"),
        }
    )
    env.pop("CONFIRM_LOCAL16", None)

    result = subprocess.run(
        ["bash", "scripts/run_60day_expected_edge_study.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "runtime_memory_limit_gb=8" in result.stdout
    assert "with_book_depth=0" in result.stdout
    assert 'max_quote_buckets="3600"' in result.stdout
    assert "threshold_candidate_attempts=50" in result.stdout
    assert "candidate_registry=" in result.stdout
    assert (tmp_path / "results" / "candidate_registry.jsonl").exists()


def test_l2_sequence_runner_dry_run_propagates_holdout_manifest(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "MODELS": "sequence_tcn",
            "SEEDS": "7",
            "OUT_DIR": str(tmp_path / "model_experiments"),
            "HOLDOUT_MANIFEST_PATH": str(tmp_path / "l2_holdout.json"),
            "DEVELOPMENT_L2_DIR": str(tmp_path / "development_l2"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_l2_sequence_experiments.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "holdout_manifest=" in result.stdout
    assert "--holdout-manifest" in result.stdout
    assert "--development-l2-output" in result.stdout
    assert "sequence_tcn_development_l2.csv" in result.stdout


def test_l2_sequence_runner_dry_run_emits_ablation_matrix(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "MODELS": "sequence_tcn",
            "SEEDS": "7,11",
            "ABLATIONS": "baseline,no_class_weighting,short_window",
            "OUT_DIR": str(tmp_path / "model_experiments"),
            "WINDOW": "16",
            "ABLATION_SHORT_WINDOW": "5",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_l2_sequence_experiments.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert 'ablations="baseline,no_class_weighting,short_window"' in result.stdout
    assert result.stdout.count("would_run model=sequence_tcn") == 6
    assert "sequence_tcn_ablation_no_class_weighting_seed_7_results.csv" in result.stdout
    assert "--class-weighting none" in result.stdout
    assert "sequence_tcn_ablation_short_window_seed_11_results.csv" in result.stdout
    assert "--window 5" in result.stdout


def test_l2_sequence_runner_rejects_unknown_ablation(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "MODELS": "sequence_tcn",
            "ABLATIONS": "baseline,unknown_variant",
            "OUT_DIR": str(tmp_path / "model_experiments"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_l2_sequence_experiments.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "unknown ablation=unknown_variant" in result.stderr


def test_shadow_fill_session_helper_dry_run_emits_order_plans(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow.csv"
    shadow.write_text(
        "decision_id,timestamp_ms,venue,symbol,model_name,predicted_side,predicted_edge_bps,"
        "order_type,intended_price,intended_size,observed_fill_price,observed_fill_size,realized_pnl,notes\n"
        "d1,1700000000000,bybit,BTCUSDT,edge,1,0.8,paper_taker,100.0,1.0,,,,\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "SHADOW_PATH": str(shadow),
            "OUTPUT_PATH": str(tmp_path / "observed_fills_template.csv"),
            "ORDER_PLAN_DIR": str(tmp_path),
            "ORDER_PLAN_PROVIDERS": "bybit binance",
            "LIMIT": "7",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/prepare_shadow_fill_validation_session.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "observed-fill-template" in result.stdout
    assert "paper-order-plan --shadow" in result.stdout
    assert "--provider bybit" in result.stdout
    assert "--provider binance" in result.stdout
    assert "binance_usdm_order_plan.jsonl" in result.stdout


def test_shadow_fill_observation_runner_dry_run_emits_fetch_import_validate(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow.csv"
    simulated = tmp_path / "simulated.csv"
    shadow.write_text(
        "decision_id,timestamp_ms,venue,symbol,model_name,predicted_side,predicted_edge_bps,"
        "order_type,intended_price,intended_size,observed_fill_price,observed_fill_size,realized_pnl,notes\n"
        "d1,1700000000000,bybit,BTCUSDT,edge,1,0.8,paper_taker,100.0,1.0,,,,\n"
    )
    simulated.write_text("decision_id,simulated_fill_price,simulated_fill_size\nd1,100.0,1.0\n")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "PROVIDER": "okx",
            "SHADOW_PATH": str(shadow),
            "SIMULATED_PATH": str(simulated),
            "OUT_DIR": str(tmp_path),
            "LIMIT": "5",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_shadow_fill_observation_session.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "provider=okx symbol=BTC-USDT-SWAP" in result.stdout
    assert "required_env=OKX_DEMO_API_KEY OKX_DEMO_API_SECRET OKX_DEMO_API_PASSPHRASE" in result.stdout
    assert "missing_env=OKX_DEMO_API_KEY OKX_DEMO_API_SECRET OKX_DEMO_API_PASSPHRASE" in result.stdout
    assert "order_submission_skipped=1" in result.stdout
    assert "fetch-observed-fills" in result.stdout
    assert "normalize-observed-fills" in result.stdout
    assert "import-observed-fills" in result.stdout
    assert "validate-shadow-fills" in result.stdout
    assert "shadow_fill_validation.txt" in result.stdout


def test_shadow_fill_observation_runner_dry_run_can_preview_order_submission(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow.csv"
    simulated = tmp_path / "simulated.csv"
    order_plan = tmp_path / "bybit_order_plan.jsonl"
    shadow.write_text(
        "decision_id,timestamp_ms,venue,symbol,model_name,predicted_side,predicted_edge_bps,"
        "order_type,intended_price,intended_size,observed_fill_price,observed_fill_size,realized_pnl,notes\n"
        "d1,1700000000000,bybit,BTCUSDT,edge,1,0.8,paper_taker,100.0,1.0,,,,\n"
    )
    simulated.write_text("decision_id,simulated_fill_price,simulated_fill_size\nd1,100.0,1.0\n")
    order_plan.write_text(
        '{"decision_id":"d1","endpoint":"/v5/order/create","method":"POST","notes":"","payload":{"category":"linear","orderLinkId":"d1","orderType":"Market","qty":"1","side":"Buy","symbol":"BTCUSDT"},"provider":"bybit","symbol":"BTCUSDT"}\n'
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "SUBMIT_ORDERS": "1",
            "PROVIDER": "bybit",
            "SHADOW_PATH": str(shadow),
            "SIMULATED_PATH": str(simulated),
            "OUT_DIR": str(tmp_path),
            "ORDER_PLAN_PATH": str(order_plan),
            "LIMIT": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_shadow_fill_observation_session.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "submit_orders=1" in result.stdout
    assert "submit-paper-orders" in result.stdout
    assert "--plan" in result.stdout
    assert "--execute" not in result.stdout


def test_shadow_fill_observation_runner_requires_credentials_for_execution(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow.csv"
    simulated = tmp_path / "simulated.csv"
    shadow.write_text(
        "decision_id,timestamp_ms,venue,symbol,model_name,predicted_side,predicted_edge_bps,"
        "order_type,intended_price,intended_size,observed_fill_price,observed_fill_size,realized_pnl,notes\n"
        "d1,1700000000000,bybit,BTCUSDT,edge,1,0.8,paper_taker,100.0,1.0,,,,\n"
    )
    simulated.write_text("decision_id,simulated_fill_price,simulated_fill_size\nd1,100.0,1.0\n")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "0",
            "PROVIDER": "binance",
            "SHADOW_PATH": str(shadow),
            "SIMULATED_PATH": str(simulated),
            "OUT_DIR": str(tmp_path),
        }
    )
    env.pop("BINANCE_USDM_TESTNET_API_KEY", None)
    env.pop("BINANCE_USDM_TESTNET_API_SECRET", None)

    result = subprocess.run(
        ["bash", "scripts/run_shadow_fill_observation_session.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "missing provider credentials" in result.stderr
    assert "BINANCE_USDM_TESTNET_API_KEY" in result.stderr
    assert "BINANCE_USDM_TESTNET_API_SECRET" in result.stderr


def test_expected_edge_helper_scripts_refresh_candidate_registry() -> None:
    helper_scripts = [
        ROOT / "scripts" / "run_complete_feature_edge_jobs.sh",
        ROOT / "scripts" / "run_local16_existing_feature_edge_jobs.sh",
    ]
    for script in helper_scripts:
        text = script.read_text()
        assert "lob_forge.study_registry" in text
        assert 'candidate_registry.jsonl"' in text


def test_final_holdout_preparation_dry_run_derives_candidate_flow(tmp_path: Path) -> None:
    result_dir, plan_path, registry_path, feature_csv, artifact_path = _write_final_holdout_fixture(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "1",
            "REQUIRE_STUDY_COMPLETE": "0",
            "RESULT_DIR": str(result_dir),
            "PLAN_PATH": str(plan_path),
            "CANDIDATE_REGISTRY": str(registry_path),
            "SELECTION_OUTPUT": str(tmp_path / "selection.json"),
            "HOLDOUT_VALUES": "2023-05-17",
            "CANDIDATE_OUTPUT": str(tmp_path / "final" / "candidate.json"),
            "DEVELOPMENT_MANIFEST_OUTPUT": str(tmp_path / "final" / "development_manifest.json"),
            "FINAL_MANIFEST_OUTPUT": str(tmp_path / "final" / "final_manifest.json"),
            "FINAL_OUTPUT": str(tmp_path / "final" / "final_holdout_edge_result.json"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/prepare_final_holdout_from_expected_edge_study.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "selected_candidate=BTCUSDT horizon_ms=5000 taker_fee_bps=0.05" in result.stdout
    assert f"feature_csv={feature_csv} exists=1" in result.stdout
    assert f"walk_forward_artifact={artifact_path} exists=1" in result.stdout
    assert "step=create_development_holdout_manifest" in result.stdout
    assert "step=freeze_edge_candidate" in result.stdout
    assert "step=create_candidate_locked_final_manifest" in result.stdout
    assert "step=run_final_holdout_edge" in result.stdout
    assert "--walk-forward-artifact" in result.stdout
    assert "--candidate-json" in result.stdout
    assert "--explicit-final-evaluation" in result.stdout
    assert "--holdout-values 2023-05-17" in result.stdout


def test_final_holdout_preparation_requires_holdout_values_for_actual_run(tmp_path: Path) -> None:
    result_dir, plan_path, registry_path, _, _ = _write_final_holdout_fixture(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "DRY_RUN": "0",
            "REQUIRE_STUDY_COMPLETE": "0",
            "RESULT_DIR": str(result_dir),
            "PLAN_PATH": str(plan_path),
            "CANDIDATE_REGISTRY": str(registry_path),
            "SELECTION_OUTPUT": str(tmp_path / "selection.json"),
            "CANDIDATE_OUTPUT": str(tmp_path / "final" / "candidate.json"),
            "DEVELOPMENT_MANIFEST_OUTPUT": str(tmp_path / "final" / "development_manifest.json"),
            "FINAL_MANIFEST_OUTPUT": str(tmp_path / "final" / "final_manifest.json"),
            "FINAL_OUTPUT": str(tmp_path / "final" / "final_holdout_edge_result.json"),
        }
    )
    env.pop("HOLDOUT_VALUES", None)

    result = subprocess.run(
        ["bash", "scripts/prepare_final_holdout_from_expected_edge_study.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "HOLDOUT_VALUES is required when DRY_RUN=0." in result.stderr
    assert not (tmp_path / "final" / "candidate.json").exists()


def _write_final_holdout_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    result_dir = tmp_path / "results"
    processed_root = tmp_path / "processed"
    feature_dir = processed_root / "btcusdt_5000ms_latency_1000"
    feature_csv = feature_dir / "BTCUSDT-2023-05-16_2023-05-17-combined-features.csv"
    artifact_path = result_dir / "BTCUSDT_5000ms_fee_0p05_edge.csv"
    audit_path = result_dir / "BTCUSDT_5000ms_fee_0p05_edge_audit.csv"
    plan_path = result_dir / "run_plan.json"
    registry_path = result_dir / "candidate_registry.jsonl"

    feature_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    feature_csv.write_text(
        "timestamp_ms,source_date,label,bid_price,ask_price,mid_price,spread_bps,trade_count\n"
        "1,2023-05-16,1,100,101,100.5,1.0,2\n"
        "2,2023-05-17,-1,100,101,100.5,1.0,3\n"
    )
    artifact_path.write_text(
        "fold,name,edge_threshold_bps,val_net_pnl,test_net_pnl,val_break_even_fee_bps,test_break_even_fee_bps\n"
        "0,edge_long_short,0.1,1.5,1.0,0.2,0.1\n"
    )
    audit_path.write_text("acceptance_passed,rejection_reasons,fold_count,one_sided_p_value_mean_le_zero\n1,,20,0.01\n")
    plan_path.write_text(
        json.dumps(
            {
                "profile": "cloud_full",
                "start_date": "2023-05-16",
                "end_date": "2023-05-17",
                "symbols": ["BTCUSDT"],
                "horizons_ms": [5000],
                "fees_bps": [0.05],
                "latency_ms": 1000,
                "processed_root": str(processed_root),
            },
            sort_keys=True,
        )
        + "\n"
    )
    registry_path.write_text(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "audit_acceptance_passed": True,
                "audit_path": str(audit_path),
                "edge_threshold_bps": 0.1,
                "horizon_ms": 5000,
                "latency_ms": 1000,
                "selected": True,
                "selected_fold_count": 20,
                "status": "selected_in_artifact",
                "symbol": "BTCUSDT",
                "taker_fee_bps": 0.05,
                "test_net_pnl": 1.0,
                "validation_net_pnl": 1.5,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return result_dir, plan_path, registry_path, feature_csv, artifact_path
