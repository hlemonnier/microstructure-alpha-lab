from __future__ import annotations

import math
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
    if not values:
        raise ValueError("need at least one value")
    if len(values) == 1:
        return 0.0
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
    return math.sqrt(max(0.0, variance / n))


def block_bootstrap_mean_interval(
    values: list[float],
    *,
    block_size: int,
    samples: int = 500,
    confidence: float = 0.95,
) -> MeanUncertainty:
    if not values:
        raise ValueError("need at least one value")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    means: list[float] = []
    n = len(values)
    blocks = [values[index : index + block_size] for index in range(0, n, block_size)]
    state = 17
    for _ in range(samples):
        draw: list[float] = []
        while len(draw) < n:
            state = _lcg(state)
            draw.extend(blocks[state % len(blocks)])
        draw = draw[:n]
        means.append(sum(draw) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = means[min(len(means) - 1, max(0, int(alpha * len(means))))]
    upper = means[min(len(means) - 1, max(0, int((1.0 - alpha) * len(means)) - 1))]
    mean = sum(values) / n
    se = _sample_std(means, sum(means) / len(means))
    return MeanUncertainty(n=n, mean=mean, standard_error=se, lower=lower, upper=upper)


def stationary_bootstrap_mean_interval(
    values: list[float],
    *,
    expected_block_size: int,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 17,
) -> MeanUncertainty:
    if not values:
        raise ValueError("need at least one value")
    if expected_block_size <= 0:
        raise ValueError("expected_block_size must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    n = len(values)
    restart_probability = min(1.0, 1.0 / expected_block_size)
    state = seed
    means: list[float] = []
    for _ in range(samples):
        state = _lcg(state)
        index = state % n
        draw: list[float] = []
        while len(draw) < n:
            draw.append(values[index])
            state = _lcg(state)
            if (state / (2**31)) < restart_probability:
                state = _lcg(state)
                index = state % n
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
    group_items = sorted((name, values) for name, values in groups.items() if values)
    if not group_items:
        raise ValueError("groups contain no observations")
    state = seed
    means: list[float] = []
    for _ in range(samples):
        draw: list[float] = []
        for _ in range(len(group_items)):
            state = _lcg(state)
            draw.extend(group_items[state % len(group_items)][1])
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
    if not values:
        raise ValueError("need at least one value")
    if expected_block_size <= 0:
        raise ValueError("expected_block_size must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    n = len(values)
    probability = 1.0 / expected_block_size
    means: list[float] = []
    state = seed
    for _ in range(samples):
        state = _lcg(state)
        index = state % n
        draw: list[float] = []
        while len(draw) < n:
            draw.append(values[index])
            state = _lcg(state)
            if state / (2**31) < probability:
                state = _lcg(state)
                index = state % n
            else:
                index = (index + 1) % n
        means.append(sum(draw) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = means[min(len(means) - 1, max(0, int(alpha * len(means))))]
    upper = means[min(len(means) - 1, max(0, int((1.0 - alpha) * len(means)) - 1))]
    mean = sum(values) / n
    return MeanUncertainty(
        n=n, mean=mean, standard_error=_sample_std(means, sum(means) / len(means)), lower=lower, upper=upper
    )


def sharpe_like_interval(
    returns: list[float],
    *,
    block_size: int = 1,
    samples: int = 500,
    confidence: float = 0.95,
) -> StatisticInterval:
    if not returns:
        raise ValueError("need at least one return")
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
    if not gross_pnl:
        raise ValueError("need at least one observation")
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
) -> list[float]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    n = len(values)
    blocks: list[list[Any]] = [[values[(start + offset) % n] for offset in range(block_size)] for start in range(n)]
    state = 31
    draws: list[float] = []
    for _ in range(samples):
        draw: list[Any] = []
        while len(draw) < n:
            state = _lcg(state)
            draw.extend(blocks[state % len(blocks)])
        draws.append(statistic_fn(draw[:n]))
    return draws


def _interval(values: list[float], *, confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    ordered = sorted(values)
    alpha = (1.0 - confidence) / 2.0
    lower = ordered[min(len(ordered) - 1, max(0, int(alpha * len(ordered))))]
    upper = ordered[min(len(ordered) - 1, max(0, int((1.0 - alpha) * len(ordered)) - 1))]
    return lower, upper


def _sharpe_like(values: list[float]) -> float:
    mean = sum(values) / len(values)
    std = _sample_std(values, mean)
    return mean / std if std else 0.0


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
    draws.sort()
    lower, upper = _quantile_interval(draws, confidence)
    mean = sum(values) / len(values)
    se = _sample_std(draws, sum(draws) / len(draws))
    return MeanUncertainty(n=len(values), mean=mean, standard_error=se, lower=lower, upper=upper)


def _quantile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    if not values:
        raise ValueError("need at least one value")
    ordered = sorted(values)
    alpha = (1.0 - confidence) / 2.0
    lower = ordered[min(len(ordered) - 1, max(0, int(alpha * len(ordered))))]
    upper = ordered[min(len(ordered) - 1, max(0, int((1.0 - alpha) * len(ordered)) - 1))]
    return lower, upper


def _lcg(state: int) -> int:
    return (1103515245 * state + 12345) % (2**31)
