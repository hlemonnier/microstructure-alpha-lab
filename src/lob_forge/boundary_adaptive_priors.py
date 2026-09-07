"""Causal same-day class-frequency estimates for decision-only research.

These estimates do not recalibrate a model's natural probabilities. They supply
the denominator of its class-balanced decision rule. Label observations enter
only at their actual release time; exponential age is measured from the original
decision, so a late outcome never masquerades as a recent market observation.
"""

from __future__ import annotations

import numpy as np

POLICIES = (
    "registered", "training", "labels_cumulative", "labels_300", "labels_900", "labels_3600",
    "forecast_300", "forecast_900", "forecast_3600", "hybrid_300", "hybrid_900",
)
PRIOR_STRENGTH = 60.0
PRIOR_FLOOR = 0.001


def _clock(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{name} must be a finite integer-millisecond vector")
    clock = raw.astype(np.int64)
    if (np.diff(clock) <= 0).any():
        raise ValueError(f"{name} must be strictly increasing")
    return clock


def _prior(values):
    prior = np.asarray(values, dtype=float)
    if prior.shape != (3,) or not np.isfinite(prior).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError("A positive normalized three-class prior is required")
    return prior


def _smoothed(counts):
    return PRIOR_FLOOR + (1 - 3 * PRIOR_FLOOR) * counts / counts.sum()


def label_priors(labels, origin_times, available_times, query_times, *, half_life_seconds=None):
    """Return available same-day empirical priors, with 60 uniform pseudocounts.

The pseudocounts originate at the earlier of the first observation and first
query. For an exponential estimate, each released observation has weight
2**(-(query_time-origin_time)/(1000*half_life_seconds)). The cumulative estimate
does not decay. A common floor prevents empty classes from dividing by zero.
"""
    y = np.asarray(labels)
    origins = _clock(origin_times, "Origin times")
    queries = _clock(query_times, "Query times")
    raw_release = np.asarray(available_times)
    if (
        y.shape != origins.shape or not np.isin(y, [-1, 0, 1]).all()
        or raw_release.shape != origins.shape or not np.isfinite(raw_release).all()
        or not np.equal(raw_release, np.floor(raw_release)).all()
    ):
        raise ValueError("Aligned three-class labels and integer release times are required")
    release = raw_release.astype(np.int64)
    if (release <= origins).any() or (np.diff(release) < 0).any():
        raise ValueError("Labels must be strictly delayed and released in nondecreasing order")
    if half_life_seconds is not None and (not np.isfinite(half_life_seconds) or half_life_seconds <= 0):
        raise ValueError("The half-life must be finite and positive")
    result = np.empty((len(queries), 3), dtype=float)
    if not len(queries):
        return result
    counts = np.full(3, PRIOR_STRENGTH / 3)
    previous = min(queries[0], origins[0]) if len(origins) else queries[0]
    cursor = 0
    for row, now in enumerate(queries):
        if half_life_seconds is not None:
            counts *= np.exp2(-(now - previous) / (1000 * half_life_seconds))
        while cursor < len(y) and release[cursor] <= now:
            weight = 1 if half_life_seconds is None else np.exp2(-(now - origins[cursor]) / (1000 * half_life_seconds))
            counts[int(y[cursor]) + 1] += weight
            cursor += 1
        # Extreme gaps can underflow all historical mass. A uniform fallback is
        # deterministic and depends only on the already available empty state.
        result[row] = _smoothed(counts) if counts.sum() > 0 else np.full(3, 1 / 3)
        previous = now
    return result


def forecast_priors(probabilities, query_times, initial_prior, *, half_life_seconds):
    """Causal marginal forecast average, including the current known forecast."""
    p = np.asarray(probabilities, dtype=float)
    queries = _clock(query_times, "Query times")
    initial = _prior(initial_prior)
    if (
        p.shape != (len(queries), 3) or not np.isfinite(p).all() or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1) or not np.isfinite(half_life_seconds) or half_life_seconds <= 0
    ):
        raise ValueError("Finite aligned natural probabilities and a positive half-life are required")
    counts = PRIOR_STRENGTH * initial
    result = np.empty_like(p)
    for row, now in enumerate(queries):
        if row:
            counts *= np.exp2(-(now - queries[row - 1]) / (1000 * half_life_seconds))
        counts += p[row]
        result[row] = _smoothed(counts)
    return result


def all_policies(probabilities, query_times, training_prior, registered_prior, label_estimates):
    """Compose the fixed development family without modifying probabilities."""
    n = len(query_times)
    if not n:
        raise ValueError("At least one forecast is required")
    result = {
        "registered": np.broadcast_to(_prior(registered_prior), (n, 3)).copy(),
        "training": np.broadcast_to(_prior(training_prior), (n, 3)).copy(),
    }
    for name in ("labels_cumulative", "labels_300", "labels_900", "labels_3600"):
        value = np.asarray(label_estimates[name], dtype=float)
        if value.shape != (n, 3) or not np.isfinite(value).all() or (value <= 0).any() or not np.allclose(value.sum(axis=1), 1):
            raise ValueError("Positive aligned available-label estimates are required")
        result[name] = value.copy()
    for half_life in (300, 900, 3600):
        result[f"forecast_{half_life}"] = forecast_priors(
            probabilities, query_times, result["labels_cumulative"][0], half_life_seconds=half_life,
        )
    for half_life in (300, 900):
        result[f"hybrid_{half_life}"] = (result[f"labels_{half_life}"] + result[f"forecast_{half_life}"]) / 2
    return {name: result[name] for name in POLICIES}
