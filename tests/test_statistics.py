from lob_forge.statistics import autocorrelation, block_bootstrap_mean_interval, newey_west_standard_error


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
