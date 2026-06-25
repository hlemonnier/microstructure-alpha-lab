from pathlib import Path

from lob_forge.result_verifier import verify_result_artifacts


def test_result_verifier_accepts_rejected_audit_with_reason(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev="abc1234",
        command="python3 -m lob_forge.cli walk-forward data.csv --holdout-manifest manifest.json",
        holdout_manifest_path="manifest.json",
        holdout_manifest_sha256="a" * 64,
    )
    (tmp_path / "pvalues.csv").write_text("hypothesis_id,metric,p_value\nh1,fold_mean_net_pnl,0.1\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=0, rejection_reasons="fold count too low"))

    report = verify_result_artifacts(tmp_path)

    assert report.passed
    assert report.audit_files == 1


def test_result_verifier_rejects_missing_reason(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev="abc1234",
        command="python3 -m lob_forge.cli walk-forward data.csv --holdout-manifest manifest.json",
        holdout_manifest_path="manifest.json",
        holdout_manifest_sha256="a" * 64,
    )
    (tmp_path / "pvalues.csv").write_text("hypothesis_id,metric,p_value\nh1,fold_mean_net_pnl,0.1\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=0, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert "rejection_reasons" in report.errors[0]


def test_result_verifier_rejects_stale_protocol_ledger_metadata(tmp_path: Path) -> None:
    (tmp_path / "hypotheses.jsonl").write_text('{"hypothesis_id":"h1"}\n')
    _write_ledger(
        tmp_path / "experiment_ledger.jsonl",
        git_rev="no-git-commit",
        command="python3 -m lob_forge.cli walk-forward data.csv --sort-by validation_net_pnl",
        holdout_manifest_path="",
        holdout_manifest_sha256="",
    )
    (tmp_path / "pvalues.csv").write_text("hypothesis_id,metric,p_value\nh1,fold_mean_net_pnl,0.1\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
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
        git_rev="abc1234",
        command="python3 -m lob_forge.cli fill-diagnostics data.csv",
        holdout_manifest_path="",
        holdout_manifest_sha256="",
    )
    with (tmp_path / "experiment_ledger.jsonl").open("a") as handle:
        handle.write((tmp_path / "experiment_ledger.jsonl").read_text())
    (tmp_path / "pvalues.csv").write_text("hypothesis_id,metric,p_value\nh1,fold_mean_net_pnl,0.1\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert any("duplicate experiment_id" in error for error in report.errors)


def test_result_verifier_study_mode_does_not_require_ledger(tmp_path: Path) -> None:
    (tmp_path / "pvalues.csv").write_text("hypothesis_id,metric,p_value\nh1,fold_mean_net_pnl,0.1\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(_audit_csv(acceptance_passed=1, rejection_reasons=""))

    report = verify_result_artifacts(tmp_path, strict_metadata=False)

    assert report.passed


def _audit_csv(*, acceptance_passed: int, rejection_reasons: str, inference_grain: str = "fold_summary") -> str:
    return (
        "inference_grain,fold_count,total_test_rows,total_test_trades,total_test_net_pnl,"
        "acceptance_passed,rejection_reasons\n"
        f"{inference_grain},4,100,10,1.0,{acceptance_passed},{rejection_reasons}\n"
    )


def _write_ledger(
    path: Path,
    *,
    git_rev: str,
    command: str,
    holdout_manifest_path: str,
    holdout_manifest_sha256: str,
) -> None:
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
        "}\n"
    )
