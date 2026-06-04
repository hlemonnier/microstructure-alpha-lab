from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MeanUncertainty:
    n: int
    mean: float
    standard_error: float
    lower: float
    upper: float


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


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _lcg(state: int) -> int:
    return (1103515245 * state + 12345) % (2**31)
