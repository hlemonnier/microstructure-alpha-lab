"""Estimate decision class frequencies from information available at prediction time."""

from __future__ import annotations

import numpy as np


def decision_priors(policy, training_priors, previous_priors, labels, times, label_available_times):
    training, previous = np.asarray(training_priors, dtype=float), np.asarray(previous_priors, dtype=float)
    y, clock, release = (
        np.asarray(labels, dtype=int),
        np.asarray(times, dtype=np.int64),
        np.asarray(label_available_times, dtype=np.int64),
    )
    if (
        training.shape != (3,)
        or previous.shape != (3,)
        or (training <= 0).any()
        or (previous <= 0).any()
        or not np.isclose(training.sum(), 1)
        or not np.isclose(previous.sum(), 1)
        or y.shape != clock.shape
        or release.shape != clock.shape
        or not np.isin(y, [-1, 0, 1]).all()
        or (np.diff(clock) <= 0).any()
        or (np.diff(release) < 0).any()
        or (release <= clock).any()
    ):
        raise ValueError("Valid priors, aligned observations, and strictly delayed ordered labels required")
    if policy in ["training", "previous_noon", "blend_training_previous"]:
        value = (
            training if policy == "training" else previous if policy == "previous_noon" else (training + previous) / 2
        )
        return np.broadcast_to(value, (len(y), 3)).copy()
    if policy not in ["online_300", "online_1800", "online_7200"]:
        raise ValueError("Unknown registered prior policy")
    half_life = int(policy.split("_")[1])
    counts = previous * half_life / np.log(2)
    result = np.empty((len(y), 3))
    cursor = 0
    for row in range(len(y)):
        if row:
            counts *= np.exp2(-(clock[row] - clock[row - 1]) / (1000 * half_life))
        while cursor < row and release[cursor] <= clock[row]:
            counts[y[cursor] + 1] += 1
            cursor += 1
        result[row] = 0.01 + 0.97 * counts / counts.sum()
    return result
