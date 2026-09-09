"""Past-only quote conversion for alternative spot-market observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_spot_features import spot_feature_frame


def _integer_clock(values, *, strict):
    clock = np.asarray(values)
    if (clock.ndim != 1 or not np.issubdtype(clock.dtype, np.integer)
        or (clock < 0).any() or (clock >= 2**53).any()):
        raise ValueError("Ordered, nonnegative, exactly represented integer milliseconds required")
    clock = clock.astype(np.int64, copy=False)
    if (np.diff(clock) <= 0 if strict else np.diff(clock) < 0).any():
        raise ValueError("Ordered integer milliseconds required; unsigned subtraction must not hide regressions")
    return clock


def converted_spot_feature_frame(canonical, trades, conversion, *, information_delay_ms,
                                 conversion_delay_ms, maximum_conversion_age_ms):
    """Keep native returns/flow and express price-basis fields in target units.

    `conversion.price` is target quote units per one native quote unit. Its
    `publisher_time` receives the explicit extra conversion delay. A quote is
    usable only strictly after publication plus delay and within its age limit.
    Window VWAPs are valued at the currently observed conversion rate; they are
    not averages of historically converted execution prices. The source return
    fields remain returns in the native quote currency.
    """
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           for v in (information_delay_ms, conversion_delay_ms, maximum_conversion_age_ms)):
        raise ValueError("Information delays and the conversion age bound must be integer milliseconds")
    information_delay_ms, conversion_delay_ms, maximum_conversion_age_ms = map(
        int, (information_delay_ms, conversion_delay_ms, maximum_conversion_age_ms))
    if (information_delay_ms not in (100, 500) or conversion_delay_ms not in (0, 100, 500)
        or not conversion_delay_ms < maximum_conversion_age_ms < 2**53):
        raise ValueError("A registered conversion delay and larger finite age bound are required")
    clock = _integer_clock(canonical.decision_time.to_numpy(), strict=True)
    publisher = _integer_clock(conversion.publisher_time.to_numpy(), strict=False)
    rates = conversion.price.to_numpy(dtype=float)
    if rates.shape != publisher.shape or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Aligned finite positive target-quote per native-quote conversion prices required")
    if (publisher > 2**53 - 1 - conversion_delay_ms).any():
        raise ValueError("Conversion availability clocks must remain exactly represented")
    available, chosen, age = np.zeros(len(clock), dtype=bool), np.ones(len(clock)), np.zeros(len(clock), dtype=np.int64)
    if len(publisher):
        last = np.searchsorted(publisher + conversion_delay_ms, clock, side="left") - 1
        present, safe = last >= 0, np.maximum(0, last)
        age = np.where(present, clock - publisher[safe], 0)
        available = present & (age <= maximum_conversion_age_ms)
        chosen[available] = rates[safe[available]]
    known = canonical.loc[:, ["decision_time", "bid", "ask", "bid_qty", "ask_qty"]].copy()
    known["bid"] = known.bid.to_numpy(dtype=float) / chosen
    known["ask"] = known.ask.to_numpy(dtype=float) / chosen
    frame = spot_feature_frame(known, trades, information_delay_ms=information_delay_ms)
    valid_basis = available & frame.spot_available.to_numpy(dtype=bool)
    basis = frame.spot_last_basis_bps.to_numpy().copy()
    basis_columns = [c for c in frame if "basis" in c]
    frame.loc[~valid_basis, basis_columns] = 0.0
    # Missing conversion rows must not enter the persistent basis estimate.
    # Masking only its output would contaminate forecasts after coverage returns.
    observed = pd.Series(np.where(valid_basis, basis, np.nan))
    for horizon in (60000, 300000, 3600000):
        slow = observed.ewm(halflife=pd.Timedelta(milliseconds=horizon),
            times=pd.to_datetime(clock, unit="ms"), adjust=True).mean().fillna(0).to_numpy()
        frame[f"spot_basis_mean_{horizon}ms"] = np.where(valid_basis, slow, 0.0)
        frame[f"spot_basis_deviation_{horizon}ms"] = np.where(valid_basis, basis - slow, 0.0)
    frame["conversion_available"] = available.astype(float)
    frame["conversion_log_age_ms"] = np.where(available, np.log1p(np.maximum(0, age)), 0.0)
    frame["conversion_log_target_quote_per_native_quote"] = np.where(available, np.log(chosen), 0.0)
    frame["converted_basis_available"] = valid_basis.astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Converted source observations must remain finite")
    return frame
