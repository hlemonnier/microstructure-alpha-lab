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
