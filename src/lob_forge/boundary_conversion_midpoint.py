"""Explicit trade-side midpoint proxies for observed conversion transactions."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from lob_forge.boundary_quote_currency_features import _integer_clock, converted_spot_feature_frame


def side_adjusted_conversion_frame(trades, *, half_spacing):
    """Subtract a fixed half-spread hypothesis times the observed aggressor sign.

    `price` and `half_spacing` both have units of target quote per native quote.
    The caller fixes the hypothesis from earlier training data. This is a midpoint
    proxy, not an observed executable quote or a guaranteed unbiased estimate.
    The zero-half-spacing control retains the identical trade-side information.
    """
    if (not isinstance(half_spacing, Real) or isinstance(half_spacing, (bool, np.bool_))
        or not np.isfinite(half_spacing) or half_spacing < 0):
        raise ValueError("A fixed finite nonnegative numeric half-spacing is required")
    clock = _integer_clock(trades.transact_time.to_numpy(), strict=False)
    prices = trades.price.to_numpy(dtype=float)
    maker = trades.is_buyer_maker.to_numpy()
    if (prices.shape != clock.shape or maker.shape != clock.shape or not np.issubdtype(maker.dtype, np.bool_)
        or not np.isfinite(prices).all() or (prices <= half_spacing).any()):
        raise ValueError("Finite positive conversion trades and explicit boolean maker sides required")
    side = np.where(maker, -1., 1.)
    return pd.DataFrame({"publisher_time": clock, "price": prices - float(half_spacing) * side, "trade_side": side})


def side_aware_converted_spot_frame(canonical, trades, conversion_trades, *, half_spacing,
                                  information_delay_ms, conversion_delay_ms, maximum_conversion_age_ms):
    """Keep raw/adjusted variants matched on the exact observed FX side channel."""
    conversion = side_adjusted_conversion_frame(conversion_trades, half_spacing=half_spacing)
    features = converted_spot_feature_frame(canonical, trades, conversion,
        information_delay_ms=information_delay_ms, conversion_delay_ms=conversion_delay_ms,
        maximum_conversion_age_ms=maximum_conversion_age_ms)
    clock = _integer_clock(canonical.decision_time.to_numpy(), strict=True)
    if len(conversion):
        index = np.searchsorted(conversion.publisher_time.to_numpy() + conversion_delay_ms, clock, side="left") - 1
        side = conversion.trade_side.to_numpy()[np.maximum(index, 0)]
        features["conversion_last_side"] = np.where(features.conversion_available.to_numpy(dtype=bool), side, 0.)
    else:
        features["conversion_last_side"] = 0.
    return features
