from pathlib import Path

from lob_forge.result_verifier import verify_result_artifacts


def test_result_verifier_accepts_rejected_audit_with_reason(tmp_path: Path) -> None:
    for filename in ["hypotheses.jsonl", "experiment_ledger.jsonl", "pvalues.csv"]:
        (tmp_path / filename).write_text("x\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(
        "fold_count,total_test_rows,total_test_trades,total_test_net_pnl,acceptance_passed,rejection_reasons\n"
        "4,100,10,1.0,0,fold count too low\n"
    )

    report = verify_result_artifacts(tmp_path)

    assert report.passed
    assert report.audit_files == 1


def test_result_verifier_rejects_missing_reason(tmp_path: Path) -> None:
    for filename in ["hypotheses.jsonl", "experiment_ledger.jsonl", "pvalues.csv"]:
        (tmp_path / filename).write_text("x\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(
        "fold_count,total_test_rows,total_test_trades,total_test_net_pnl,acceptance_passed,rejection_reasons\n"
        "4,100,10,1.0,0,\n"
    )

    report = verify_result_artifacts(tmp_path)

    assert not report.passed
    assert "rejection_reasons" in report.errors[0]


def test_result_verifier_study_mode_does_not_require_ledger(tmp_path: Path) -> None:
    (tmp_path / "pvalues.csv").write_text("hypothesis_id,metric,p_value\nh1,fold_mean_net_pnl,0.1\n")
    (tmp_path / "pvalue_corrections.csv").write_text(
        "hypothesis_id,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept\nh1,0.1,0.1,0.1,0\n"
    )
    (tmp_path / "sample_audit.csv").write_text(
        "fold_count,total_test_rows,total_test_trades,total_test_net_pnl,acceptance_passed,rejection_reasons\n"
        "4,100,10,1.0,1,\n"
    )

    report = verify_result_artifacts(tmp_path, strict_metadata=False)

    assert report.passed
