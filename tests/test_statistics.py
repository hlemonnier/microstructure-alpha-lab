from lob_forge.statistics import (
    _moving_block_statistic,
    autocorrelation,
    block_bootstrap_mean_interval,
    break_even_cost_interval,
    day_level_mean_interval,
    newey_west_standard_error,
    sharpe_like_interval,
    stationary_block_bootstrap_mean_interval,
    grouped_bootstrap_mean_interval,
    student_t_cdf,
)
import math
import pytest


def test_newey_west_and_block_bootstrap_return_positive_uncertainty() -> None:
    values = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]

    se = newey_west_standard_error(values, lags=2)
    interval = block_bootstrap_mean_interval(values, block_size=2, samples=50)

    assert se > 0.0
    assert interval.lower <= interval.mean <= interval.upper
    assert interval.standard_error >= 0.0


def test_autocorrelation_detects_positive_lag_relation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert autocorrelation(values) > 0.0


def test_day_level_interval_aggregates_before_bootstrap() -> None:
    interval = day_level_mean_interval([1.0, 2.0, -1.0], ["d1", "d1", "d2"], samples=20)

    assert interval.n == 2
    assert interval.mean == 1.0


def test_stationary_block_bootstrap_and_statistic_intervals() -> None:
    values = [1.0, -0.5, 0.25, 0.75, -0.25, 0.5]

    mean_interval = stationary_block_bootstrap_mean_interval(values, expected_block_size=2, samples=30)
    sharpe_interval = sharpe_like_interval(values, block_size=2, samples=30)
    cost_interval = break_even_cost_interval([1.0, -0.5, 0.25], [100.0, 100.0, 50.0], block_size=1, samples=30)

    assert mean_interval.lower <= mean_interval.upper
    assert sharpe_interval.method == "moving_block_bootstrap"
    assert cost_interval.statistic == 30.0


def test_moving_block_bootstrap_uses_overlapping_blocks() -> None:
    draws = _moving_block_statistic(
        [1, 2, 3, 4, 5],
        statistic_fn=lambda draw: float(draw[0] * 10 + draw[1]),
        block_size=2,
        samples=20,
    )

    assert 23.0 in draws or 45.0 in draws


def test_stationary_block_bootstrap_seed_controls_draws() -> None:
    values = [1.0, -5.0, 2.0, 7.0, -3.0, 4.0, 9.0, -2.0]

    first = stationary_block_bootstrap_mean_interval(values, expected_block_size=2, samples=100, seed=1)
    repeated = stationary_block_bootstrap_mean_interval(values, expected_block_size=2, samples=100, seed=1)
    different = stationary_block_bootstrap_mean_interval(values, expected_block_size=2, samples=100, seed=2)

    assert first == repeated
    assert (first.lower, first.upper, first.standard_error) != (
        different.lower,
        different.upper,
        different.standard_error,
    )


def test_power_of_two_bootstrap_matches_exact_iid_mean_variance() -> None:
    # Empirical population variance is 1.25; the mean of four iid draws has
    # exact variance 1.25/4. The old LCG returned variance zero here.
    values = [0.0, 1.0, 2.0, 3.0]
    exact_se = math.sqrt(1.25 / 4)
    fixed = block_bootstrap_mean_interval(values, block_size=1, samples=20000)
    grouped = grouped_bootstrap_mean_interval({str(i): [v] for i, v in enumerate(values)}, samples=20000)
    for result in (fixed, grouped):
        assert abs(result.standard_error - exact_se) < 0.012
        assert result.lower < result.upper


def test_stationary_bootstrap_has_correct_iid_limit() -> None:
    result = stationary_block_bootstrap_mean_interval([0.0, 1.0], expected_block_size=1, samples=20000)
    assert abs(result.standard_error - math.sqrt(0.25 / 2)) < 0.008


def test_circular_blocks_do_not_overweight_short_trailing_block_positions() -> None:
    # Even when n is not a multiple of block length, each original position
    # must have expected inclusion count one in a resampled series.
    draws = _moving_block_statistic(
        [0.0, 100.0, 0.0], statistic_fn=lambda values: sum(values) / len(values),
        block_size=2, samples=20000,
    )
    assert abs(sum(draws) / len(draws) - 100 / 3) < 0.8


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_inference_input_fails_closed(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        block_bootstrap_mean_interval([1.0, value], block_size=1)
    with pytest.raises(ValueError, match="finite"):
        newey_west_standard_error([1.0, value])
    with pytest.raises(ValueError, match="finite"):
        stationary_block_bootstrap_mean_interval([1.0, value], expected_block_size=1)


def test_uncertainty_rejects_single_observation_and_single_group() -> None:
    with pytest.raises(ValueError):
        newey_west_standard_error([1.0])
    with pytest.raises(ValueError):
        grouped_bootstrap_mean_interval({"only": [1.0, 2.0]})
    with pytest.raises(ValueError, match="zero-variance"):
        sharpe_like_interval([1.0, 1.0, 1.0])


def test_student_t_cdf_matches_cauchy_and_known_quantiles() -> None:
    for x in [-100, -1, 0, 1, 100]:
        assert student_t_cdf(x, 1) == pytest.approx(0.5 + math.atan(x) / math.pi, abs=1e-13)
    for df, quantile in [(2, 4.302652729749464), (10, 2.2281388519649385), (30, 2.0422724563012373)]:
        assert student_t_cdf(quantile, df) == pytest.approx(0.975, abs=1e-11)
    assert student_t_cdf(-math.inf, 1) == 0
    assert student_t_cdf(math.inf, 1) == 1
    with pytest.raises(ValueError):
        student_t_cdf(math.nan, 2)
