"""Matched raw/adjusted FX representations with unchanged original model inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_quote_currency_features import _integer_clock
from lob_forge.boundary_quote_currency_inputs import CONVERSION_COLUMNS, NATIVE_COLUMNS, quote_variant_columns

PREFIX = "quote_BTCUSDT_100__"
FIELDS = (*CONVERSION_COLUMNS, *NATIVE_COLUMNS, "conversion_last_side")


def append_midpoint_conversion(original, decision_times, raw_side, midpoint_side):
    """Add two representations of the same strictly observed conversion trades."""
    clock = _integer_clock(decision_times, strict=True)
    if len(clock) != len(original) or original.columns.duplicated().any():
        raise ValueError("The original selected-row frame and its unique schema must remain intact")
    selected = []
    for frame in (raw_side, midpoint_side):
        source_clock = _integer_clock(frame.index.to_numpy(), strict=True)
        if (frame.columns.duplicated().any() or set(frame.columns) != set(FIELDS)
            or not np.isin(clock, source_clock).all()):
            raise ValueError("Both conversion representations must cover every original clock with the exact source schema")
        chosen = frame.loc[clock, list(FIELDS)].reset_index(drop=True)
        if not np.isfinite(chosen.to_numpy(dtype=float)).all():
            raise ValueError("Explicitly masked side-aware source observations must be finite")
        selected.append(chosen)
    raw, midpoint = selected
    unchanged = [c for c in FIELDS if ("basis" not in c or c == "converted_basis_available")
                 and c != "conversion_log_target_quote_per_native_quote"]
    pd.testing.assert_frame_equal(raw[unchanged], midpoint[unchanged], check_exact=True)
    original_fields = [c for c in FIELDS if c != "conversion_last_side"]
    expected = raw[original_fields].add_prefix(PREFIX)
    if not set(expected.columns).issubset(original.columns):
        raise ValueError("The exact original raw conversion and native-market fields must be present")
    pd.testing.assert_frame_equal(original[expected.columns].reset_index(drop=True), expected, check_exact=True)
    augmented = pd.concat([original.reset_index(drop=True), raw[["conversion_last_side"]].add_prefix(PREFIX),
        midpoint.add_prefix("midpoint_" + PREFIX)], axis=1)
    if augmented.columns.duplicated().any():
        raise ValueError("Original and side-aware conversion column names must be unambiguous")
    pd.testing.assert_frame_equal(augmented[original.columns], original.reset_index(drop=True), check_exact=True)
    return augmented


def midpoint_model_columns(metadata, *, representation, variant, conversion_mode):
    if representation not in ("observations", "combined") or variant not in ("fx100", "btc100", "both100"):
        raise ValueError("Use an original representation and a declared 100 ms source ablation")
    if conversion_mode not in ("raw_side", "midpoint_side"):
        raise ValueError("The raw and midpoint controls both include observed conversion side")
    original = metadata["original_columns"] if representation == "combined" else metadata["observation_columns"]
    columns = [*quote_variant_columns(original, variant), PREFIX + "conversion_last_side"]
    if conversion_mode == "midpoint_side":
        columns = ["midpoint_" + c if c.startswith(PREFIX) else c for c in columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Every model must select a unique explicit feature schema")
    return columns
