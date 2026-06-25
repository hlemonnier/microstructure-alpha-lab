import subprocess
import shlex
from pathlib import Path


SCRIPT = Path("scripts/source_provenance.sh").resolve()


def test_source_git_rev_uses_explicit_env_hash(tmp_path: Path) -> None:
    result = _source_git_rev(tmp_path, extra_env=f"LOB_FORGE_SOURCE_GIT_COMMIT={'a' * 40}")

    assert result.returncode == 0
    assert result.stdout.strip() == "a" * 40


def test_source_git_rev_uses_expanded_archive_hash_without_git(tmp_path: Path) -> None:
    (tmp_path / ".source-git-commit").write_text("b" * 40 + "\n")

    result = _source_git_rev(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == "b" * 40


def test_source_git_rev_rejects_unexpanded_archive_placeholder_without_git(tmp_path: Path) -> None:
    (tmp_path / ".source-git-commit").write_text("$Format:%H$\n")

    result = _source_git_rev(tmp_path)

    assert result.returncode == 2
    assert "source provenance requires Git metadata" in result.stderr


def test_result_artifact_script_does_not_emit_invalid_package_git_sentinel() -> None:
    script = Path("scripts/reproduce_result_artifacts.sh").read_text()

    assert "package-no-git" not in script
    assert "source_git_rev" in script


def _source_git_rev(root: Path, *, extra_env: str = "") -> subprocess.CompletedProcess[str]:
    parts = [
        "set -euo pipefail",
        f"ROOT_DIR={shlex.quote(str(root))}",
    ]
    if extra_env:
        parts.append(extra_env)
    parts.extend([f"source {shlex.quote(str(SCRIPT))}", "source_git_rev"])
    command = "; ".join(parts)
    return subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        cwd=root,
    )
