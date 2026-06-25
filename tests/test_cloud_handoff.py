from pathlib import Path


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
    script = Path("scripts/package_cloud_handoff.sh").read_text()

    assert "--include='/requirements-ci.txt'" in script
    assert "--include='/requirements-research.txt'" in script
