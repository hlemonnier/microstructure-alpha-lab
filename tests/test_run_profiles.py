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
    assert "fetch-observed-fills" in result.stdout
    assert "normalize-observed-fills" in result.stdout
    assert "import-observed-fills" in result.stdout
    assert "validate-shadow-fills" in result.stdout
    assert "shadow_fill_validation.txt" in result.stdout


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
