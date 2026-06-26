import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_reduced_e2e_artifacts.py"


def test_reduced_e2e_artifact_verifier_skips_missing_manifest(tmp_path: Path) -> None:
    result = _run_verifier(tmp_path)

    assert result.returncode == 0
    assert "present=0 skipped=1" in result.stdout


def test_reduced_e2e_artifact_verifier_rejects_stale_git_commit(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    _write_reduced_artifacts(tmp_path, git_commit="b" * 40)

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert f"manifest={'b' * 12} head={head[:12]}" in result.stdout


def test_reduced_e2e_artifact_verifier_accepts_current_clean_artifacts(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    _write_reduced_artifacts(tmp_path, git_commit=head)

    result = _run_verifier(tmp_path)

    assert result.returncode == 0
    assert "present=1 errors=0" in result.stdout


def test_reduced_e2e_artifact_verifier_rejects_untagged_current_artifacts(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path, tag_head=False)
    _write_reduced_artifacts(tmp_path, git_commit=head)

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert f"git_commit has no local tag: {head[:12]}" in result.stdout


def _run_verifier(project_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(project_root)],
        text=True,
        capture_output=True,
        timeout=10,
    )


def _init_git_repo(path: Path, *, tag_head: bool = True) -> str:
    (path / "README.md").write_text("# temp\n")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if tag_head:
        subprocess.run(["git", "tag", "fixture-reduced-e2e"], cwd=path, check=True, capture_output=True, text=True)
    return head


def _write_reduced_artifacts(path: Path, *, git_commit: str) -> None:
    artifact_dir = path / "artifacts" / "reduced_e2e"
    artifact_dir.mkdir(parents=True)
    result_manifest = artifact_dir / "result_manifest.json"
    result_manifest.write_text(
        json.dumps(
            {
                "claim_scope": "Synthetic reduced E2E fixture only; no live or historical profitability claim.",
            }
        )
        + "\n"
    )
    stateful_orders = artifact_dir / "stateful_orders.csv"
    stateful_orders.write_text("order_id\n1\n")
    research_manifest = path / "artifacts" / "research_manifest.json"
    research_manifest.write_text(
        json.dumps(
            {
                "git_commit": git_commit,
                "working_tree_dirty": False,
                "reduced_e2e_manifest": "artifacts/reduced_e2e/result_manifest.json",
                "reduced_e2e_artifacts": {
                    "stateful_orders": "artifacts/reduced_e2e/stateful_orders.csv",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
