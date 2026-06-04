from lob_forge.calibration import (
    edge_reliability_by_decile,
    expected_calibration_error,
    format_calibration_bins,
    format_edge_reliability_bins,
    multiclass_brier_score,
    posterior_mean_score,
)


def test_brier_and_ece_reward_calibrated_predictions() -> None:
    labels = [1, -1, 0, 1]
    probabilities = [
        {1: 0.8, 0: 0.1, -1: 0.1},
        {1: 0.1, 0: 0.2, -1: 0.7},
        {1: 0.2, 0: 0.6, -1: 0.2},
        {1: 0.7, 0: 0.2, -1: 0.1},
    ]

    brier = multiclass_brier_score(labels, probabilities, classes=[-1, 0, 1])
    ece, bins = expected_calibration_error(labels, probabilities, bins=5)
    output = format_calibration_bins(bins)

    assert brier < 0.3
    assert ece < 0.4
    assert output.startswith("bucket,lower_bound")


def test_edge_reliability_by_decile_orders_by_predicted_edge() -> None:
    predicted_edges = [0.1, 0.2, 1.0, 1.5, 2.0, 3.0]
    realized = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]

    bins = edge_reliability_by_decile(predicted_edges, realized, deciles=3)
    output = format_edge_reliability_bins(bins)

    assert len(bins) == 3
    assert bins[0].mean_predicted_edge < bins[-1].mean_predicted_edge
    assert bins[-1].mean_realized_pnl > bins[0].mean_realized_pnl
    assert output.startswith("decile,lower_edge")


def test_posterior_mean_score_reports_probability_against_margin() -> None:
    score = posterior_mean_score([0.5, 1.0, 1.5, 2.0], cost_margin=0.25, prior_observations=0.0)

    assert score.n == 4
    assert score.p_mean_gt_zero > 0.95
    assert score.p_mean_gt_margin > 0.9
    assert score.posterior_sharpe > 0
