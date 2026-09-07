from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MeanUncertainty:
    n: int
    mean: float
    standard_error: float
    lower: float
    upper: float


@dataclass(frozen=True)
class StatisticInterval:
    n: int
    statistic: float
    lower: float
    upper: float
    method: str


def newey_west_standard_error(values: list[float], *, lags: int | None = None) -> float:
    _validate_values(values, minimum=2)
    n = len(values)
    lag_count = lags if lags is not None else int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    if lag_count < 0:
        raise ValueError("lags must be non-negative")
    mean = sum(values) / n
    centered = [value - mean for value in values]
    gamma0 = sum(value * value for value in centered) / n
    variance = gamma0
    for lag in range(1, min(lag_count, n - 1) + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        covariance = sum(centered[index] * centered[index - lag] for index in range(lag, n)) / n
        variance += 2.0 * weight * covariance
    if not math.isfinite(variance):
        raise ValueError("nonfinite HAC variance; check input scale")
    return math.sqrt(max(0.0, variance / n))


def block_bootstrap_mean_interval(
    values: list[float],
    *,
    block_size: int,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 17,
) -> MeanUncertainty:
    """Circular fixed-length blocks; all origins have equal probability.

    Circular equal-length blocks avoid unequal inclusion weights from a short
    final block. This estimates the empirical resampling distribution, not a
    guarantee of coverage for arbitrary nonstationary data.
    """
    _validate_values(values, minimum=2)
    means = _moving_block_statistic(
        values, statistic_fn=lambda draw: math.fsum(draw) / len(draw),
        block_size=block_size, samples=samples, seed=seed,
    )
    return _interval_from_draws(values, means, confidence)


def stationary_bootstrap_mean_interval(
    values: list[float],
    *,
    expected_block_size: int,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 17,
) -> MeanUncertainty:
    _validate_values(values, minimum=2)
    if expected_block_size <= 0:
        raise ValueError("expected_block_size must be positive")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    n = len(values)
    restart_probability = min(1.0, 1.0 / expected_block_size)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        index = rng.randrange(n)
        draw: list[float] = []
        while len(draw) < n:
            draw.append(values[index])
            if rng.random() < restart_probability:
                index = rng.randrange(n)
            else:
                index = (index + 1) % n
        means.append(sum(draw) / n)
    return _interval_from_draws(values, means, confidence)


def grouped_bootstrap_mean_interval(
    groups: dict[str, list[float]],
    *,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 17,
) -> MeanUncertainty:
    if not groups:
        raise ValueError("need at least one group")
    flattened = [value for values in groups.values() for value in values]
    if not flattened:
        raise ValueError("groups contain no observations")
    _validate_values(flattened, minimum=2)
    group_items = sorted((name, values) for name, values in groups.items() if values)
    if not group_items:
        raise ValueError("groups contain no observations")
    if len(group_items) < 2:
        raise ValueError("need at least two nonempty groups for grouped uncertainty")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw: list[float] = []
        for _ in range(len(group_items)):
            draw.extend(group_items[rng.randrange(len(group_items))][1])
        means.append(sum(draw) / len(draw))
    return _interval_from_draws(flattened, means, confidence)


def day_level_mean_interval(
    values: list[float],
    days: list[str],
    *,
    samples: int = 500,
    confidence: float = 0.95,
) -> MeanUncertainty:
    if len(values) != len(days):
        raise ValueError("values and days must have the same length")
    _validate_values(values, minimum=2)
    grouped: dict[str, float] = {}
    for value, day in zip(values, days):
        grouped[day] = grouped.get(day, 0.0) + value
    if not grouped:
        raise ValueError("need at least one value")
    return block_bootstrap_mean_interval(
        [grouped[day] for day in sorted(grouped)],
        block_size=1,
        samples=samples,
        confidence=confidence,
    )


def stationary_block_bootstrap_mean_interval(
    values: list[float],
    *,
    expected_block_size: int,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 29,
) -> MeanUncertainty:
    return stationary_bootstrap_mean_interval(
        values, expected_block_size=expected_block_size, samples=samples,
        confidence=confidence, seed=seed,
    )


def sharpe_like_interval(
    returns: list[float],
    *,
    block_size: int = 1,
    samples: int = 500,
    confidence: float = 0.95,
) -> StatisticInterval:
    _validate_values(returns, minimum=2)
    statistic = _sharpe_like(returns)
    draws = _moving_block_statistic(
        returns,
        statistic_fn=_sharpe_like,
        block_size=block_size,
        samples=samples,
    )
    lower, upper = _interval(draws, confidence=confidence)
    return StatisticInterval(len(returns), statistic, lower, upper, "moving_block_bootstrap")


def break_even_cost_interval(
    gross_pnl: list[float],
    turnover: list[float],
    *,
    block_size: int = 1,
    samples: int = 500,
    confidence: float = 0.95,
) -> StatisticInterval:
    if len(gross_pnl) != len(turnover):
        raise ValueError("gross_pnl and turnover must have the same length")
    _validate_values(gross_pnl, minimum=2)
    _validate_values(turnover, minimum=2)
    if any(value <= 0.0 for value in turnover):
        raise ValueError("cost uncertainty requires positive turnover for each included observation")
    pairs = list(zip(gross_pnl, turnover))
    statistic = _break_even_cost_bps(pairs)
    draws = _moving_block_statistic(
        pairs,
        statistic_fn=_break_even_cost_bps,
        block_size=block_size,
        samples=samples,
    )
    lower, upper = _interval(draws, confidence=confidence)
    return StatisticInterval(len(pairs), statistic, lower, upper, "moving_block_bootstrap")


def autocorrelation(values: list[float], *, lag: int = 1) -> float:
    _validate_values(values)
    if lag <= 0:
        raise ValueError("lag must be positive")
    if len(values) <= lag:
        return 0.0
    mean = sum(values) / len(values)
    denominator = sum((value - mean) ** 2 for value in values)
    if denominator == 0.0:
        return 0.0
    numerator = sum((values[index] - mean) * (values[index - lag] - mean) for index in range(lag, len(values)))
    return numerator / denominator


def _moving_block_statistic(
    values: list[Any],
    *,
    statistic_fn: Callable[[list[Any]], float],
    block_size: int,
    samples: int,
    seed: int = 31,
) -> list[float]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    n = len(values)
    if n < 2:
        raise ValueError("need at least two values")
    if block_size >= n:
        raise ValueError("block_size must be smaller than the observation count")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        draw: list[Any] = []
        while len(draw) < n:
            start = rng.randrange(n)
            draw.extend(values[(start + offset) % n] for offset in range(min(block_size, n - len(draw))))
        draws.append(statistic_fn(draw[:n]))
    return draws


def _interval(values: list[float], *, confidence: float) -> tuple[float, float]:
    return _quantile_interval(values, confidence)


def _sharpe_like(values: list[float]) -> float:
    mean = sum(values) / len(values)
    std = _sample_std(values, mean)
    if not std:
        raise ValueError("Sharpe-like statistic is undefined for zero-variance samples or resamples")
    return mean / std


def _break_even_cost_bps(pairs: list[tuple[float, float]]) -> float:
    total_turnover = sum(turnover for _, turnover in pairs)
    if not total_turnover:
        return 0.0
    return sum(pnl for pnl, _ in pairs) / total_turnover * 10000.0


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _interval_from_draws(values: list[float], draws: list[float], confidence: float) -> MeanUncertainty:
    _validate_values(values, minimum=2)
    _validate_values(draws, minimum=2)
    draws.sort()
    lower, upper = _quantile_interval(draws, confidence)
    mean = sum(values) / len(values)
    se = _sample_std(draws, sum(draws) / len(draws))
    return MeanUncertainty(n=len(values), mean=mean, standard_error=se, lower=lower, upper=upper)


def _quantile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    _validate_values(values, minimum=2)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    ordered = sorted(values)
    alpha = (1.0 - confidence) / 2.0
    lower = ordered[min(len(ordered) - 1, max(0, int(alpha * len(ordered))))]
    upper = ordered[min(len(ordered) - 1, max(0, int((1.0 - alpha) * len(ordered)) - 1))]
    return lower, upper


def _validate_values(values: list[float], *, minimum: int = 1) -> None:
    if len(values) < minimum:
        raise ValueError(f"need at least {minimum} observations")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("observations must be finite")


def student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    """Student-t CDF via regularized incomplete beta, without SciPy."""
    if math.isnan(value) or not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError("Student-t requires a non-NaN value and finite positive degrees_of_freedom")
    if value == 0.0:
        return 0.5
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    df = degrees_of_freedom
    x = df / (df + value * value)
    tail = 0.5 * _regularized_beta(x, df / 2.0, 0.5)
    return 1.0 - tail if value > 0 else tail


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    scale = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return scale * _beta_continued_fraction(a, b, x) / a
    return 1.0 - scale * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 1001):
        even = 2 * m
        for coefficient in (
            m * (b - m) * x / ((a + even - 1.0) * (a + even)),
            -(a + m) * (a + b + m) * x / ((a + even) * (a + even + 1.0)),
        ):
            d = 1.0 + coefficient * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + coefficient / c
            c = c if abs(c) > tiny else tiny
            delta = c * d
            result *= delta
        if abs(delta - 1.0) < 3e-14:
            return result
    raise ArithmeticError("incomplete beta failed to converge")
