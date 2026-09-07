"""Causal online probability calibration with explicitly delayed label feedback."""

from __future__ import annotations

import numpy as np


def delayed_calibration(
    probabilities,
    labels,
    decision_times,
    label_available_times,
    *,
    learning_rate,
    half_life_seconds=600.0,
    bias_limit=2.0,
):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)
    times, release = np.asarray(decision_times, dtype=np.int64), np.asarray(label_available_times, dtype=np.int64)
    if (
        p.shape != (len(y), 3)
        or times.shape != y.shape
        or release.shape != y.shape
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1)
        or not np.isin(y, [-1, 0, 1]).all()
        or (np.diff(times) <= 0).any()
        or (np.diff(release) < 0).any()
        or (release <= times).any()
    ):
        raise ValueError("Aligned probabilities, ordered timestamps, and strictly delayed labels required")
    if learning_rate < 0 or half_life_seconds <= 0 or bias_limit <= 0:
        raise ValueError("Nonnegative rate and positive half life and bias bound required")
    if learning_rate == 0:
        return p.copy(), {"updates": 0, "final_bias": [0.0, 0.0, 0.0]}
    logits = np.log(np.maximum(p, 1e-12))
    calibrated = np.empty_like(p)
    bias, cursor = np.zeros(3), 0
    for row in range(len(y)):
        elapsed = times[row] - times[row - 1] if row else 0
        bias *= np.exp2(-elapsed / (1000 * half_life_seconds))
        while cursor < row and release[cursor] <= times[row]:
            gradient = -calibrated[cursor].copy()
            gradient[y[cursor] + 1] += 1
            bias = np.clip(bias + learning_rate * gradient, -bias_limit, bias_limit)
            cursor += 1
        scores = logits[row] + bias
        probabilities_row = np.exp(scores - scores.max())
        calibrated[row] = probabilities_row / probabilities_row.sum()
    return calibrated, {"updates": cursor, "final_bias": bias.tolist()}
