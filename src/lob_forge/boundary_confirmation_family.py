"""Multiplicity accounting for two procedures fixed before outcome inspection."""

from __future__ import annotations

import numpy as np


def family_interval(values, *, block_length=1, seed=20260907):
    """97.5% marginal percentile interval for a two-procedure Bonferroni family."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) != 20 or not np.isfinite(x).all():
        raise ValueError("Exactly twenty finite paired date effects required")
    if block_length not in {1, 3}:
        raise ValueError("Registered dependence blocks are one or three dates")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), size=(10000, (len(x) + block_length - 1) // block_length))
    indices = ((starts[:, :, None] + np.arange(block_length)) % len(x)).reshape(10000, -1)[:, :len(x)]
    lower, upper = np.quantile(x[indices].mean(axis=1), [0.0125, 0.9875])
    return {
        "mean": float(x.mean()), "lower_97_5": float(lower), "upper_97_5": float(upper),
        "block_length": block_length, "distinct_dates": len(x), "resamples": 10000,
        "seed": seed, "registered_procedure_count": 2,
        "scope": "Bonferroni allocation across two frozen candidate procedures, subject to bootstrap coverage assumptions.",
    }
