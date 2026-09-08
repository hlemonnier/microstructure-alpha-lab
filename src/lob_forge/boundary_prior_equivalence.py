"""Exact equivalence of constant and per-query three-class decision priors."""

from __future__ import annotations

import numpy as np


def assert_decision_priors_equal(actual, expected, *, rows: int) -> None:
    """Allow only (3,) or (rows, 3), then compare every value exactly.

    Broadcasting a constant prior changes its storage, never its values. Do not
    normalize, average, cast, or use a numerical tolerance for this comparison.
    """
    if isinstance(rows, (bool, np.bool_)) or not isinstance(rows, (int, np.integer)) or rows < 1:
        raise ValueError("The query count must be a positive integer")
    expanded = []
    for prior in (actual, expected):
        values = np.asarray(prior)
        if values.shape not in ((3,), (rows, 3)):
            raise ValueError("A decision prior must have shape (3,) or (rows, 3)")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Decision priors must contain finite positive numeric values")
        expanded.append(np.broadcast_to(values, (rows, 3)))
    np.testing.assert_array_equal(expanded[0], expanded[1])
