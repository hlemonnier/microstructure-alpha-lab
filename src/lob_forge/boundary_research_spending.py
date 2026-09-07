"""A declared nominal error budget for continuing independent research rounds."""

from __future__ import annotations

import numpy as np


def research_round_interval(values, *, round_number, procedures, block_length=1):
    x = np.asarray(values, dtype=float)
    if round_number < 1 or procedures < 1 or int(round_number) != round_number or int(procedures) != procedures:
        raise ValueError("Positive integer round and registered procedure counts required")
    if x.ndim != 1 or len(x) < 20 or not np.isfinite(x).all() or not 1 <= block_length <= len(x):
        raise ValueError("At least twenty finite date effects and a valid dependence block required")
    round_alpha = 0.05 / 2**round_number
    tail = round_alpha / (2 * procedures)
    if tail * 20000 < 20:
        raise ValueError("Register more bootstrap resamples before using a tail with fewer than twenty expected draws")
    rng = np.random.default_rng(20260907)
    starts = rng.integers(0, len(x), size=(20000, (len(x) + block_length - 1) // block_length))
    indices = ((starts[:, :, None] + np.arange(block_length)) % len(x)).reshape(20000, -1)[:, :len(x)]
    lower, upper = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail])
    return {
        "mean": float(x.mean()), "lower": float(lower), "upper": float(upper),
        "round_number": int(round_number), "registered_procedures": int(procedures),
        "round_alpha": round_alpha, "marginal_interval_coverage": 1 - round_alpha / procedures,
        "bootstrap_resamples": 20000, "block_length": block_length, "seed": 20260907,
        "scope": "Nominal alpha spending: round j receives 0.05/2**j, split across its registered procedures. The union-bound allocation assumes each bootstrap interval has its stated coverage; it does not guarantee that approximation under arbitrary market dependence.",
    }
