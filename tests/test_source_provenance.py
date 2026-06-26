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


def test_source_worktree_dirty_reports_clean_git_repo(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    result = _source_worktree_dirty(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == "false"


def test_source_worktree_dirty_reports_tracked_changes(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("dirty\n")

    result = _source_worktree_dirty(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == "true"


def test_source_worktree_dirty_returns_null_without_git(tmp_path: Path) -> None:
    result = _source_worktree_dirty(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == "null"


def test_result_artifact_script_does_not_emit_invalid_package_git_sentinel() -> None:
    script = Path("scripts/reproduce_result_artifacts.sh").read_text()

    assert "package-no-git" not in script
    assert "source_git_rev" in script
    assert "source_worktree_dirty" in script


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


def _source_worktree_dirty(root: Path, *, extra_env: str = "") -> subprocess.CompletedProcess[str]:
    parts = [
        "set -euo pipefail",
        f"ROOT_DIR={shlex.quote(str(root))}",
    ]
    if extra_env:
        parts.append(extra_env)
    parts.extend([f"source {shlex.quote(str(SCRIPT))}", "source_worktree_dirty"])
    command = "; ".join(parts)
    return subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        cwd=root,
    )


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True)
    (path / "README.md").write_text("fixture\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "fixture"], check=True, capture_output=True, text=True)
