"""Exact-row ablations of currency conversion and alternative spot information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_spot_features import SPOT_FEATURES

SYMBOLS = ("BTCUSDT", "ETHUSDT")
CONVERSION_COLUMNS = ("conversion_available", "conversion_log_age_ms", "conversion_log_target_quote_per_native_quote")
NATIVE_COLUMNS = (*SPOT_FEATURES, "converted_basis_available")
QUOTE_VARIANTS = {"fx100": (100, ()), "btc100": (100, ("BTCUSDT",)), "both100": (100, SYMBOLS),
                  "fx500": (500, ()), "btc500": (500, ("BTCUSDT",)), "both500": (500, SYMBOLS)}


def quote_variant_columns(original_columns, variant):
    if variant not in QUOTE_VARIANTS:
        raise ValueError("Use a declared conversion-only, BTC, or joint-source representation")
    delay, native_sources = QUOTE_VARIANTS[variant]
    columns = [*original_columns]
    for symbol in SYMBOLS:
        fields = (*CONVERSION_COLUMNS, *(NATIVE_COLUMNS if symbol in native_sources else ()))
        columns.extend(f"quote_{symbol}_{delay}__{c}" for c in fields)
    if len(columns) != len(set(columns)):
        raise ValueError("Original and alternative-source columns must have distinct names")
    return columns


def append_quote_sources(original, decision_times, sources):
    """Join separately normalized markets on exactly the original clocks.

    Both FX-only controls retain both conversion streams. Native BTC/ETH fields
    keep explicit market prefixes and their own base-asset quantity units, even
    when used to forecast the other target asset. No forward fill or row dropping
    is permitted here; original model preparation defines the common row set.
    """
    clock = np.asarray(decision_times)
    if (clock.ndim != 1 or len(clock) != len(original) or not np.issubdtype(clock.dtype, np.integer)
        or (clock < 0).any() or (clock >= 2**53).any() or (np.diff(clock.astype(np.int64)) <= 0).any()):
        raise ValueError("Exactly aligned, strictly increasing integer decision clocks required")
    expected = {(s, delay) for s in SYMBOLS for delay in (100, 500)}
    if set(sources) != expected:
        raise ValueError("Both observed markets and both declared delays are required")
    fields = [*CONVERSION_COLUMNS, *NATIVE_COLUMNS]
    pieces = [original.reset_index(drop=True)]
    for symbol, delay in sorted(expected):
        source = sources[symbol, delay]
        index = source.index.to_numpy()
        if (source.columns.duplicated().any() or set(source.columns) != set(fields)
            or not np.issubdtype(index.dtype, np.integer) or (index < 0).any() or (index >= 2**53).any()
            or (np.diff(index.astype(np.int64)) <= 0).any() or not np.isin(clock, index).all()):
            raise ValueError("Each source must cover every original row with its exact unique schema and clock")
        selected = source.loc[clock, fields].reset_index(drop=True)
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError("Explicitly masked source fields must be finite")
        pieces.append(selected.add_prefix(f"quote_{symbol}_{delay}__"))
    frame = pd.concat(pieces, axis=1)
    if frame.columns.duplicated().any():
        raise ValueError("Original and alternative-source schemas must remain unambiguous")
    pd.testing.assert_frame_equal(frame[list(original.columns)], original.reset_index(drop=True), check_exact=True)
    return frame
