from lob_forge.statistics import (
    _moving_block_statistic,
    autocorrelation,
    block_bootstrap_mean_interval,
    break_even_cost_interval,
    day_level_mean_interval,
    newey_west_standard_error,
    sharpe_like_interval,
    stationary_block_bootstrap_mean_interval,
)


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
