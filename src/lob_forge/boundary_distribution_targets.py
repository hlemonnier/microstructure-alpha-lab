"""Nine exact move-size bins that aggregate to the frozen three-class target."""

from __future__ import annotations

from math import gcd

import numpy as np

from lob_forge.label_math import _decimal_fraction, exact_label_array

DISTRIBUTION_SEMANTICS = "exact_nine_move_bins_v1_threshold_multiples_1_2_4"
COARSE_CLASSES = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int8)


def exact_move_bins(entry_bid, entry_ask, future_bid, future_ask, *, min_tick):
    """Keep direction, exact zero and magnitudes <=T, <=2T, <=4T and >4T.

    T is twice the canonical mid-price threshold. Signed bins are symmetric:
    equality at any magnitude boundary belongs to the nearer-to-zero bin.
    Bins 0:3, 3:6 and 6:9 exactly recover down, neutral and up respectively.
    """
    tick = _decimal_fraction(min_tick)
    arrays = [np.asarray(v, dtype=float) for v in (entry_bid, entry_ask, future_bid, future_ask)]
    if tick <= 0 or any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("A positive finite tick and aligned quote vectors are required")
    if any(not np.isfinite(a).all() or (a <= 0).any() for a in arrays):
        raise ValueError("Finite positive quotes are required")
    if (arrays[1] < arrays[0]).any() or (arrays[3] < arrays[2]).any():
        raise ValueError("Entry and future quotes must not cross")
    if not len(arrays[0]):
        return np.empty(0, dtype=np.int8)
    values, inverse = np.unique(np.concatenate(arrays), return_inverse=True)
    fractions = [*map(_decimal_fraction, values), tick]
    scale = 1
    for value in fractions:
        scale = scale // gcd(scale, value.denominator) * value.denominator
    units = [v.numerator * (scale // v.denominator) for v in fractions]
    # The extra safety factor covers absolute differences and 4 * 2 * tick.
    dtype = np.int64 if max(units) <= np.iinfo(np.int64).max // 16 else object
    eb, ea, fb, fa = np.asarray(units[:-1], dtype=dtype)[inverse].reshape(4, -1)
    delta = fb + fa - eb - ea
    threshold = np.maximum(ea - eb, 2 * units[-1])
    magnitude = np.abs(delta)
    radius = (magnitude > 0).astype(np.int8)
    for multiplier in (1, 2, 4):
        radius += (magnitude > multiplier * threshold).astype(np.int8)
    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    result = (4 + direction * radius).astype(np.int8)
    np.testing.assert_array_equal(COARSE_CLASSES[result], exact_label_array(*arrays, min_tick=min_tick))
    return result


def coarse_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 9 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Normalized nine-bin probabilities are required")
    return p.reshape(-1, 3, 3).sum(axis=2)
