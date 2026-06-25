from pathlib import Path

from lob_forge.alpha_factory import (
    AcceptanceCriteria,
    PValueRecord,
    audit_result_artifact,
    benjamini_hochberg_adjust,
    correct_p_values,
    evaluate_acceptance,
    format_result_audit_csv,
    format_result_audit_markdown,
    one_sided_hac_p_value_mean_le_zero,
    one_sided_normal_p_value_mean_le_zero,
)
import lob_forge.alpha_factory as alpha_factory
from lob_forge.statistics import MeanUncertainty


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


def test_audit_result_artifact_uses_seeded_5_95_bootstrap_and_hac_wording(tmp_path: Path) -> None:
    path = tmp_path / "walk_forward.csv"
    path.write_text(
        "\n".join(
            [
                "fold,test_rows,test_trades,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_win_rate",
                "1,100,10,12,10,0.2,0.1,0.6",
                "2,100,10,8,6,0.1,0.1,0.5",
                "3,100,10,-2,-1,0.05,-0.1,0.4",
                "4,100,10,3,2,0.05,0.1,0.6",
                "summary,,,21,17,0.12,,",
            ]
        )
    )

    calls: list[tuple[int, float]] = []

    def fake_interval(values, *, expected_block_size, samples=500, confidence=0.95, seed=29):
        calls.append((seed, confidence))
        return MeanUncertainty(
            n=len(values),
            mean=sum(values) / len(values),
            standard_error=0.0,
            lower=1.0,
            upper=2.0,
        )

    original_interval = alpha_factory.stationary_block_bootstrap_mean_interval
    try:
        alpha_factory.stationary_block_bootstrap_mean_interval = fake_interval
        audit = audit_result_artifact(path, bootstrap_samples=100, seed=123)
        markdown = format_result_audit_markdown(audit, evaluate_acceptance(audit, AcceptanceCriteria(min_fold_count=1)))
    finally:
        alpha_factory.stationary_block_bootstrap_mean_interval = original_interval

    assert calls == [(123, 0.90)]
    assert audit.bootstrap_mean_net_pnl_lower_5pct == 1.0
    assert audit.bootstrap_mean_net_pnl_upper_95pct == 2.0
    assert "5/95% interval" in markdown
    assert "HAC/Newey-West z p-value" in markdown


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
    assert "inference_grain" in output
    assert "fold_summary" in output


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


def test_hac_p_value_is_available_for_serial_fold_audits() -> None:
    p_value = one_sided_hac_p_value_mean_le_zero([1.0, 0.8, 0.9, 1.1, 1.0])

    assert 0.0 <= p_value <= 1.0
    assert p_value < 0.05


def test_result_audit_markdown_names_hac_and_interval_correctly(tmp_path: Path) -> None:
    path = tmp_path / "walk_forward.csv"
    path.write_text(
        "\n".join(
            [
                "fold,test_rows,test_trades,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_win_rate",
                "1,100,10,12,10,0.2,0.1,0.6",
                "2,100,10,8,6,0.1,0.1,0.5",
                "3,100,10,-2,-1,0.05,-0.1,0.4",
            ]
        )
    )
    audit = audit_result_artifact(path, bootstrap_samples=50, seed=1)
    verdict = evaluate_acceptance(audit, AcceptanceCriteria(min_fold_count=1))

    markdown = format_result_audit_markdown(audit, verdict)

    assert "95% interval" in markdown
    assert "Inference grain: fold_summary" in markdown
    assert "HAC/Newey-West" in markdown
    assert "One-sided normal p-value" not in markdown
