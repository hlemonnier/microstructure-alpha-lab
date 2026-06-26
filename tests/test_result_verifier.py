from __future__ import annotations

import subprocess
from pathlib import Path

from lob_forge.result_verifier import verify_result_artifacts


FIXTURE_GIT_REV = "a" * 40


def test_result_verifier_accepts_rejected_audit_with_reason(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev=FIXTURE_GIT_REV,
        command="python3 -m lob_forge.cli walk-forward data.csv --holdout-manifest manifest.json",
        holdout_manifest_path="manifest.json",
        holdout_manifest_sha256="a" * 64,
    )
    _write_pvalue_files(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=0, rejection_reasons="fold count too low"))

    report = verify_result_artifacts(tmp_path)

    assert report.passed
    assert report.audit_files == 1


def test_result_verifier_rejects_missing_reason(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev=FIXTURE_GIT_REV,
        command="python3 -m lob_forge.cli walk-forward data.csv --holdout-manifest manifest.json",
        holdout_manifest_path="manifest.json",
        holdout_manifest_sha256="a" * 64,
    )
    _write_pvalue_files(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=0, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert "rejection_reasons" in report.errors[0]


def test_result_verifier_rejects_unknown_inference_grain(tmp_path: Path) -> None:
    _write_minimal_result_dir(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(
        _audit_csv(acceptance_passed=1, rejection_reasons="", inference_grain="normal_pvalue_trade_claim")
    )

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("unsupported inference_grain" in error for error in report.errors)


def test_result_verifier_requires_ledger_artifact_for_ledger_grain(tmp_path: Path) -> None:
    _write_minimal_result_dir(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(
        _audit_csv(acceptance_passed=1, rejection_reasons="", inference_grain="trade_ledger")
    )

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("trade_ledger inference requires" in error for error in report.errors)


def test_result_verifier_accepts_ledger_grain_with_present_ledger(tmp_path: Path) -> None:
    _write_minimal_result_dir(tmp_path)
    (tmp_path / "fills.csv").write_text("fill_time,price,quantity\n1000,100.0,1.0\n")
    (tmp_path / "sample_audit.csv").write_text(
        _audit_csv(
            acceptance_passed=1,
            rejection_reasons="",
            inference_grain="trade_ledger",
            extra_header=",trade_ledger_path",
            extra_values=",fills.csv",
        )
    )

    report = verify_result_artifacts(tmp_path)

    assert report.passed


def test_result_verifier_rejects_pvalues_missing_audit_artifact_id(tmp_path: Path) -> None:
    _write_minimal_result_dir(tmp_path, pvalue_id="other_result")
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("missing p-value rows for audit artifacts: sample" in error for error in report.errors)
    assert any("p-value IDs without matching audit artifact: other_result" in error for error in report.errors)


def test_result_verifier_defers_config_hash_pvalue_join_to_study_registry(tmp_path: Path) -> None:
    _write_minimal_result_dir(tmp_path, pvalue_id="grid:config:abc", config_sha256="a" * 64)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert report.passed


def test_result_verifier_rejects_in_repo_ledger_from_different_head(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    result_dir = repo / "results"
    result_dir.mkdir()
    _write_minimal_result_dir(result_dir, git_rev="a" * 40)
    (result_dir / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(result_dir)

    assert not report.passed
    assert any("does not match current HEAD" in error for error in report.errors)


def test_result_verifier_accepts_in_repo_ledger_from_current_head(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result_dir = repo / "results"
    result_dir.mkdir()
    _write_minimal_result_dir(result_dir, git_rev=head)
    (result_dir / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(result_dir)

    assert report.passed


def test_result_verifier_rejects_in_repo_ledger_from_untagged_head(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo", tag_head=False)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result_dir = repo / "results"
    result_dir.mkdir()
    _write_minimal_result_dir(result_dir, git_rev=head)
    (result_dir / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(result_dir)

    assert not report.passed
    assert any("is not tagged" in error for error in report.errors)


def test_result_verifier_rejects_in_repo_ledger_without_worktree_provenance(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result_dir = repo / "results"
    result_dir.mkdir()
    _write_minimal_result_dir(result_dir, git_rev=head, working_tree_dirty=None)
    (result_dir / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(result_dir)

    assert not report.passed
    assert any("missing working_tree_dirty provenance" in error for error in report.errors)


def test_result_verifier_rejects_in_repo_ledger_from_dirty_worktree(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result_dir = repo / "results"
    result_dir.mkdir()
    _write_minimal_result_dir(result_dir, git_rev=head, working_tree_dirty=True)
    (result_dir / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(result_dir)

    assert not report.passed
    assert any("working_tree_dirty must be false" in error for error in report.errors)


def test_result_verifier_rejects_stale_protocol_ledger_metadata(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev="no-git-commit",
        command="python3 -m lob_forge.cli walk-forward data.csv --sort-by validation_net_pnl",
        holdout_manifest_path="",
        holdout_manifest_sha256="",
    )
    _write_pvalue_files(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("invalid git_rev" in error for error in report.errors)
    assert any("lacks --holdout-manifest" in error for error in report.errors)
    assert any("missing holdout_manifest_path" in error for error in report.errors)
    assert any("holdout_manifest_sha256" in error for error in report.errors)


def test_result_verifier_rejects_duplicate_experiment_ids(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev=FIXTURE_GIT_REV,
        command="python3 -m lob_forge.cli fill-diagnostics data.csv",
        holdout_manifest_path="",
        holdout_manifest_sha256="",
    )
    with (tmp_path / "experiment_ledger.jsonl").open("a") as handle:
        handle.write((tmp_path / "experiment_ledger.jsonl").read_text())
    _write_pvalue_files(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("duplicate experiment_id" in error for error in report.errors)


def test_result_verifier_rejects_unresolved_short_git_rev(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev="0000000",
        command="python3 -m lob_forge.cli fill-diagnostics data.csv",
        holdout_manifest_path="",
        holdout_manifest_sha256="",
    )
    _write_pvalue_files(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("invalid git_rev '0000000'" in error for error in report.errors)


def test_result_verifier_study_mode_does_not_require_ledger(tmp_path: Path) -> None:
    _write_pvalue_files(tmp_path)
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path, strict_metadata=False)

    assert report.passed


def _audit_csv(
    *,
    acceptance_passed: int,
    rejection_reasons: str,
    inference_grain: str = "fold_summary",
    extra_header: str = "",
    extra_values: str = "",
) -> str:
    return (
        "inference_grain,fold_count,total_test_rows,total_test_trades,total_test_net_pnl,"
        f"acceptance_passed,rejection_reasons{extra_header}\n"
        f"{inference_grain},4,100,10,1.0,{acceptance_passed},{rejection_reasons}{extra_values}\n"
    )


def _write_minimal_result_dir(
    tmp_path: Path,
    *,
    pvalue_id: str = "sample",
    config_sha256: str = "",
    git_rev: str = FIXTURE_GIT_REV,
    working_tree_dirty: bool | None = False,
) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev=git_rev,
        command="python3 -m lob_forge.cli walk-forward data.csv --holdout-manifest manifest.json",
        holdout_manifest_path="manifest.json",
        holdout_manifest_sha256="a" * 64,
        working_tree_dirty=working_tree_dirty,
    )
    _write_pvalue_files(tmp_path, hypothesis_id=pvalue_id, config_sha256=config_sha256)


def _write_pvalue_files(tmp_path: Path, *, hypothesis_id: str = "sample", config_sha256: str = "") -> None:
    pvalue_header = "hypothesis_id,metric,p_value"
    pvalue_row = f"{hypothesis_id},fold_mean_net_pnl,0.1"
    correction_header = "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept"
    correction_row = f"{hypothesis_id},0.1,0.1,0.1,0"
    if config_sha256:
        pvalue_header += ",config_sha256"
        pvalue_row += f",{config_sha256}"
        correction_header += ",config_sha256"
        correction_row += f",{config_sha256}"
    (tmp_path / "pvalues.csv").write_text(f"{pvalue_header}\n{pvalue_row}\n")
    (tmp_path / "pvalue_corrections.csv").write_text(f"{correction_header}\n{correction_row}\n")


def _init_git_repo(path: Path, *, tag_head: bool = True) -> Path:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True)
    (path / "README.md").write_text("fixture\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "fixture"], check=True, capture_output=True, text=True)
    if tag_head:
        subprocess.run(["git", "-C", str(path), "tag", "fixture-clean-classical"], check=True)
    return path


def _write_ledger(
    path: Path,
    *,
    git_rev: str,
    command: str,
    holdout_manifest_path: str,
    holdout_manifest_sha256: str,
    working_tree_dirty: bool | None = False,
) -> None:
    working_tree_dirty_text = (
        "" if working_tree_dirty is None else f',"working_tree_dirty":{str(working_tree_dirty).lower()}'
    )
    path.write_text(
        "{"
        '"experiment_id":"e1",'
        '"hypothesis_id":"h1",'
        f'"command":"{command}",'
        '"artifact_path":"artifact.csv",'
        '"data_path":"data.csv",'
        '"candidate_count":1,'
        f'"git_rev":"{git_rev}",'
        '"created_at_utc":"2026-06-25T00:00:00+00:00",'
        f'"holdout_manifest_path":"{holdout_manifest_path}",'
        f'"holdout_manifest_sha256":"{holdout_manifest_sha256}"'
        f"{working_tree_dirty_text}"
        "}\n"
    )
