from pathlib import Path

from lob_forge.alpha_factory import (
    AcceptanceCriteria,
    PValueRecord,
    audit_result_artifact,
    benjamini_hochberg_adjust,
    correct_p_values,
    evaluate_acceptance,
    format_result_audit_csv,
    one_sided_normal_p_value_mean_le_zero,
)


def test_audit_result_artifact_scores_fold_robustness(tmp_path: Path) -> None:
    path = tmp_path / "walk_forward.csv"
    path.write_text(
        "\n".join(
            [
                "fold,test_rows,test_trades,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_win_rate",
                "1,100,10,12,10,0.2,0.1,0.6",
                "2,100,10,8,6,0.1,0.1,0.5",
                "3,100,10,-2,-1,0.05,-0.1,0.4",
                "summary,,,18,15,0.12,,",
            ]
        )
    )

    audit = audit_result_artifact(path, bootstrap_samples=100, seed=1)

    assert audit.fold_count == 3
    assert audit.total_test_trades == 30
    assert audit.total_test_net_pnl == 15
    assert round(audit.positive_fold_rate, 6) == round(2 / 3, 6)
    assert audit.median_fold_net_pnl == 6
    assert audit.weighted_break_even_fee_bps == 0.12


def test_acceptance_verdict_rejects_concentrated_weak_result(tmp_path: Path) -> None:
    path = tmp_path / "walk_forward.csv"
    path.write_text(
        "\n".join(
            [
                "fold,test_rows,test_trades,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_win_rate",
                "1,100,10,105,100,0.2,0.1,0.6",
                "2,100,10,-1,-1,0.1,-0.1,0.4",
                "summary,,,104,99,0.1,,",
            ]
        )
    )

    audit = audit_result_artifact(path, bootstrap_samples=100, seed=1)
    verdict = evaluate_acceptance(
        audit,
        AcceptanceCriteria(
            min_positive_fold_rate=0.70,
            max_positive_fold_share_of_total_net=0.40,
            min_break_even_fee_bps=0.20,
        ),
    )
    output = format_result_audit_csv(audit, verdict)

    assert not verdict.passed
    assert "positive fold rate" in "; ".join(verdict.rejection_reasons)
    assert "largest positive fold share" in "; ".join(verdict.rejection_reasons)
    assert "break-even fee" in "; ".join(verdict.rejection_reasons)
    assert "acceptance_passed" in output


def test_pvalue_corrections_apply_bonferroni_and_bh() -> None:
    corrections = correct_p_values(
        [
            PValueRecord("h1", 0.001),
            PValueRecord("h2", 0.02),
            PValueRecord("h3", 0.20),
        ],
        q=0.05,
    )

    assert corrections[0].bonferroni_p_value == 0.003
    assert corrections[0].bh_accept
    assert corrections[1].bh_adjusted_p_value == 0.03
    assert not corrections[2].bh_accept
    assert benjamini_hochberg_adjust([0.03, 0.01]) == [0.03, 0.02]


def test_one_sided_p_value_rewards_positive_mean() -> None:
    strong = one_sided_normal_p_value_mean_le_zero([1.0, 1.1, 0.9, 1.2])
    weak = one_sided_normal_p_value_mean_le_zero([1.0, -1.0, 1.0, -1.0])

    assert strong < 0.01
    assert weak == 0.5
