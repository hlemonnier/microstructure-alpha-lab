"""Run the expected-edge study on Modal with persistent data/results storage.

Usage:
    python3 -m pip install ".[cloud]"
    modal setup
    make modal-study
    MODE=run make modal-study
    MODE=sequence make modal-study

Results are written to the Modal Volume named by LOB_FORGE_MODAL_VOLUME,
or microstructure-alpha-lab-expected-edge when that environment variable is not set.
Download them with the Modal CLI, for example:
    modal volume get microstructure-alpha-lab-expected-edge /results/expected_edge_60day_20230516_20230714 ./expected_edge_results
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "microstructure-alpha-lab-expected-edge-study"
REMOTE_PROJECT_DIR = Path("/workspace/microstructure-alpha-lab")
VOLUME_MOUNT = Path("/mnt/microstructure-alpha-lab")
DEFAULT_VOLUME_NAME = os.environ.get("LOB_FORGE_MODAL_VOLUME", "microstructure-alpha-lab-expected-edge")
MODAL_CPU_CORES = float(os.environ.get("LOB_FORGE_MODAL_CPU_CORES", "8"))
MODAL_MEMORY_REQUEST_GB = int(os.environ.get("LOB_FORGE_MODAL_MEMORY_REQUEST_GB", "64"))
MODAL_MEMORY_LIMIT_GB = int(os.environ.get("LOB_FORGE_MODAL_MEMORY_LIMIT_GB", "128"))
MODAL_EPHEMERAL_DISK_GB = int(os.environ.get("LOB_FORGE_MODAL_EPHEMERAL_DISK_GB", "300"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_IGNORE = [
    ".git/**",
    ".venv/**",
    "venv/**",
    ".next/**",
    "node_modules/**",
    "data/**",
    "results/**",
    "dist/**",
    "__pycache__/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "*.pyc",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bash", "curl", "git", "rsync", "unzip", "zip")
    .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_PROJECT_DIR), copy=True, ignore=PACKAGE_IGNORE)
    .run_commands(
        f"cd {REMOTE_PROJECT_DIR} && "
        "python -m pip install --upgrade pip && "
        "python -m pip install -r requirements-research.txt && "
        "python -m pip install -e . --no-deps"
    )
)

volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={str(VOLUME_MOUNT): volume},
    cpu=MODAL_CPU_CORES,
    memory=(MODAL_MEMORY_REQUEST_GB * 1024, MODAL_MEMORY_LIMIT_GB * 1024),
    ephemeral_disk=MODAL_EPHEMERAL_DISK_GB * 1024,
    timeout=24 * 60 * 60,
)
def run_expected_edge(
    mode: str = "plan",
    study_profile: str = "cloud_full",
    run_tests: bool = True,
    confirm_heavy: bool = True,
    holdout_manifest_path: str = "",
) -> dict[str, str]:
    if mode not in {"plan", "run", "verify", "sequence"}:
        raise ValueError("mode must be one of: plan, run, verify, sequence")

    _prepare_persistent_mounts()
    env = os.environ.copy()
    env.update(
        {
            "MODE": mode,
            "STUDY_PROFILE": study_profile,
            "CONFIRM_HEAVY": "1" if confirm_heavy else "0",
            "SKIP_INSTALL": "1",
            "RUN_TESTS": "1" if run_tests else "0",
            "PYTHONPATH": "src",
        }
    )
    if holdout_manifest_path:
        env["HOLDOUT_MANIFEST_PATH"] = holdout_manifest_path
    if mode == "sequence" and not holdout_manifest_path:
        raise ValueError("holdout_manifest_path is required for sequence mode")
    subprocess.run(
        ["bash", "scripts/bootstrap_cloud_expected_edge.sh"],
        cwd=REMOTE_PROJECT_DIR,
        env=env,
        check=True,
    )
    volume.commit()
    return {
        "mode": mode,
        "study_profile": study_profile,
        "volume_name": DEFAULT_VOLUME_NAME,
        "results_path": f"{VOLUME_MOUNT}/results",
        "data_path": f"{VOLUME_MOUNT}/data",
    }


@app.local_entrypoint()
def main(
    mode: str = "plan",
    study_profile: str = "cloud_full",
    run_tests: bool = True,
    confirm_heavy: bool = True,
    holdout_manifest_path: str = "",
) -> None:
    summary = run_expected_edge.remote(
        mode=mode,
        study_profile=study_profile,
        run_tests=run_tests,
        confirm_heavy=confirm_heavy,
        holdout_manifest_path=holdout_manifest_path,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


def _prepare_persistent_mounts() -> None:
    for name in ("data", "results"):
        volume_path = VOLUME_MOUNT / name
        project_path = REMOTE_PROJECT_DIR / name
        volume_path.mkdir(parents=True, exist_ok=True)
        if project_path.is_symlink():
            continue
        if project_path.exists():
            raise RuntimeError(f"expected {project_path} to be absent or a symlink")
        project_path.symlink_to(volume_path, target_is_directory=True)
