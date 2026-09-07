"""Strict price-movement labels in exact decimal quote arithmetic.

Quotes are interpreted as their shortest decimal representations. Comparing
twice the mid-price change avoids division and prevents binary subtraction from
turning a threshold-equal move into a directional outcome. No tolerance, tick
rounding, or assumption that the minimum tick is the observed price grid is used.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd


def _decimal_fraction(value) -> Fraction:
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Prices and minimum tick must have finite decimal values") from error


def _validate_mode_tick(mode, min_tick):
    if mode not in {"half_spread", "one_tick", "zero"}:
        raise ValueError("threshold must be one of: half_spread, one_tick, zero")
    tick = _decimal_fraction(min_tick)
    if tick < 0:
        raise ValueError("Minimum tick must be nonnegative")
    return tick


def price_movement_label(entry_bid, entry_ask, future_bid, future_ask, *, min_tick, mode="half_spread") -> int:
    tick = _validate_mode_tick(mode, min_tick)
    eb, ea, fb, fa = map(_decimal_fraction, [entry_bid, entry_ask, future_bid, future_ask])
    if min(eb, ea, fb, fa) <= 0 or ea < eb or fa < fb:
        raise ValueError("Positive, non-crossed entry and future quotes required")
    delta2 = fb + fa - eb - ea
    threshold2 = max(ea - eb, 2 * tick) if mode == "half_spread" else 2 * tick if mode == "one_tick" else 0
    return 1 if delta2 > threshold2 else -1 if delta2 < -threshold2 else 0


def exact_label_array(entry_bid, entry_ask, future_bid, future_ask, *, min_tick, mode="half_spread"):
    """Vectorized equivalent for aligned one-dimensional float quote arrays.

    Convert each distinct decimal quote once to a common exact integer scale.
    Use arbitrary-precision integers when an int64 intermediate could overflow.
    NumPy remains optional for the dependency-free canonical scalar builder.
    """
    import numpy as np

    tick = _validate_mode_tick(mode, min_tick)
    arrays = [np.asarray(value, dtype=float) for value in [entry_bid, entry_ask, future_bid, future_ask]]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Aligned one-dimensional quote arrays required")
    if any(not np.isfinite(a).all() or (a <= 0).any() for a in arrays):
        raise ValueError("Finite positive quote arrays required")
    if (arrays[1] < arrays[0]).any() or (arrays[3] < arrays[2]).any():
        raise ValueError("Non-crossed entry and future quotes required")
    if not len(arrays[0]):
        return np.empty(0, dtype=np.int8)
    values, inverse = np.unique(np.concatenate(arrays), return_inverse=True)
    fractions = [*map(_decimal_fraction, values), tick]
    scale = 1
    for fraction in fractions:
        scale = scale // gcd(scale, fraction.denominator) * fraction.denominator
    units = [fraction.numerator * (scale // fraction.denominator) for fraction in fractions]
    dtype = np.int64 if max(units) <= np.iinfo(np.int64).max // 4 else object
    eb, ea, fb, fa = np.asarray(units[:-1], dtype=dtype)[inverse].reshape(4, -1)
    delta2 = fb + fa - eb - ea
    threshold2 = np.maximum(ea - eb, 2 * units[-1]) if mode == "half_spread" else 2 * units[-1] if mode == "one_tick" else 0
    return np.where(delta2 > threshold2, 1, np.where(delta2 < -threshold2, -1, 0)).astype(np.int8)
