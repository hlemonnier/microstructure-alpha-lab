import json
import os
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_modal_expected_edge_job_persists_data_and_results() -> None:
    script = Path("scripts/modal_expected_edge_job.py").read_text()

    assert "modal.Volume.from_name" in script
    assert "create_if_missing=True" in script
    assert "data" in script
    assert "results" in script
    assert "volume.commit()" in script


def test_modal_expected_edge_job_excludes_local_heavy_paths() -> None:
    script = Path("scripts/modal_expected_edge_job.py").read_text()

    for pattern in ["data/**", "results/**", ".venv/**", ".next/**", "node_modules/**"]:
        assert pattern in script


def test_modal_expected_edge_job_has_high_ram_limit() -> None:
    script = Path("scripts/modal_expected_edge_job.py").read_text()

    assert "LOB_FORGE_MODAL_MEMORY_REQUEST_GB" in script
    assert "LOB_FORGE_MODAL_MEMORY_LIMIT_GB" in script
    assert '"64"' in script
    assert '"128"' in script


def test_cloud_handoff_package_includes_locked_requirements() -> None:
    package_script = Path("scripts/package_cloud_handoff.sh").read_text()
    bootstrap_script = Path("scripts/bootstrap_cloud_expected_edge.sh").read_text()
    modal_script = Path("scripts/modal_expected_edge_job.py").read_text()

    assert "--include='/requirements-ci.txt'" in package_script
    assert "--include='/requirements-research.txt'" in package_script
    assert "pip install -r requirements-research.txt" in bootstrap_script
    assert "pip install -e . --no-deps" in bootstrap_script
    assert "pip install -r requirements-research.txt" in modal_script
    assert "pip install -e . --no-deps" in modal_script
    assert 'pip install -e ".[research]"' not in bootstrap_script
    assert "pip install -e '.[research]'" not in modal_script


def test_cloud_handoff_package_includes_reduced_e2e_fixtures_and_source_provenance() -> None:
    script = Path("scripts/package_cloud_handoff.sh").read_text()

    assert "--include='/examples/***'" in script
    assert "source scripts/source_provenance.sh" in script
    assert "source_git_rev" in script
    assert ".source-git-commit" in script


def test_cloud_handoff_archive_runs_reduced_e2e_without_git(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    package_name = "microstructure-alpha-lab-cloud-handoff-test.zip"
    env = os.environ.copy()
    env.update(
        {
            "DIST_DIR": str(dist_dir),
            "PACKAGE_NAME": package_name,
        }
    )

    package_result = subprocess.run(
        ["bash", "scripts/package_cloud_handoff.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    package_path = dist_dir / package_name
    assert package_result.returncode == 0, package_result.stderr
    assert package_path.exists()
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        assert "microstructure-alpha-lab/.source-git-commit" in names
        assert "microstructure-alpha-lab/requirements-research.txt" in names
        bootstrap = archive.read("microstructure-alpha-lab/scripts/bootstrap_cloud_expected_edge.sh").decode("utf-8")
        modal_job = archive.read("microstructure-alpha-lab/scripts/modal_expected_edge_job.py").decode("utf-8")
        assert "pip install -r requirements-research.txt" in bootstrap
        assert "pip install -e . --no-deps" in bootstrap
        assert "pip install -r requirements-research.txt" in modal_job
        assert "pip install -e . --no-deps" in modal_job
        assert 'pip install -e ".[research]"' not in bootstrap
        assert "pip install -e '.[research]'" not in modal_job
        assert not any(name.startswith("microstructure-alpha-lab/.git/") for name in names)
        archive.extractall(tmp_path / "unpacked")

    project_dir = tmp_path / "unpacked" / "microstructure-alpha-lab"
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = "src"
    run_result = subprocess.run(
        ["python3", "scripts/run_reduced_e2e.py"],
        cwd=project_dir,
        env=run_env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert run_result.returncode == 0, run_result.stderr
    research_manifest = json.loads((project_dir / "artifacts" / "research_manifest.json").read_text())
    assert research_manifest["git_commit"] == (project_dir / ".source-git-commit").read_text().strip()
    assert research_manifest["working_tree_dirty"] is None

    verify_result = subprocess.run(
        ["make", "verify-results"],
        cwd=project_dir,
        env=run_env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert verify_result.returncode == 0, verify_result.stderr
    assert "source_package=1" in verify_result.stdout
    assert "present=1 errors=0" in verify_result.stdout
