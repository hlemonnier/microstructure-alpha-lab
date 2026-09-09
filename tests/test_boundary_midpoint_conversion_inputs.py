import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_midpoint_conversion_inputs import (  # noqa: E402
    FIELDS, PREFIX, append_midpoint_conversion, midpoint_model_columns,
)
from lob_forge.boundary_quote_currency_inputs import CONVERSION_COLUMNS, NATIVE_COLUMNS, quote_variant_columns  # noqa: E402


def _sources():
    clock = np.array([1000, 2000, 3000], dtype=np.int64)
    metadata = {"original_columns": ["observation", "old_auxiliary"], "observation_columns": ["observation"]}
    columns = quote_variant_columns(metadata["original_columns"], "both100")
    original = pd.DataFrame({c: np.arange(3, dtype=float)+i for i, c in enumerate(columns)})
    raw = original[[PREFIX+c for c in (*CONVERSION_COLUMNS, *NATIVE_COLUMNS)]].copy()
    raw.columns = [c[len(PREFIX):] for c in raw.columns]
    raw["conversion_last_side"] = [1., -1., 1.]
    raw.index = clock
    midpoint = raw.copy()
    for c in midpoint:
        if ("basis" in c and c != "converted_basis_available") or c == "conversion_log_target_quote_per_native_quote":
            midpoint[c] += .1
    return original, clock, raw, midpoint, metadata


def test_exact_original_frame_and_shared_observation_values_survive_both_representations():
    original, clock, raw, midpoint, metadata = _sources()
    augmented = append_midpoint_conversion(original, clock, raw, midpoint)
    pd.testing.assert_frame_equal(augmented[original.columns], original, check_exact=True)
    for representation in ("observations", "combined"):
        for variant in ("fx100", "btc100", "both100"):
            raw_columns = midpoint_model_columns(metadata, representation=representation, variant=variant, conversion_mode="raw_side")
            mid_columns = midpoint_model_columns(metadata, representation=representation, variant=variant, conversion_mode="midpoint_side")
            assert len(raw_columns) == len(mid_columns) and len(augmented[raw_columns]) == 3
            assert PREFIX+"conversion_last_side" in raw_columns
            assert "midpoint_"+PREFIX+"conversion_last_side" in mid_columns
            assert ("old_auxiliary" in mid_columns) == (representation == "combined")
            if variant == "fx100":
                assert not any("__spot_" in c for c in raw_columns+mid_columns)


def test_mismatched_original_raw_values_or_native_flow_cannot_be_silently_replaced():
    original, clock, raw, midpoint, _ = _sources()
    original.loc[1, PREFIX+"spot_last_basis_bps"] += 1.
    with pytest.raises(AssertionError):
        append_midpoint_conversion(original, clock, raw, midpoint)
    original, clock, raw, midpoint, _ = _sources()
    midpoint.loc[2000, "spot_time_5000_log_quantity"] += 1.
    with pytest.raises(AssertionError):
        append_midpoint_conversion(original, clock, raw, midpoint)


def test_missing_rows_unknown_fields_and_unregistered_variants_fail_explicitly():
    original, clock, raw, midpoint, metadata = _sources()
    assert len(FIELDS) == 79
    with pytest.raises(ValueError, match="every original clock"):
        append_midpoint_conversion(original, clock, raw.drop(index=2000), midpoint)
    midpoint["label"] = 1
    with pytest.raises(ValueError, match="source schema"):
        append_midpoint_conversion(original, clock, raw, midpoint)
    with pytest.raises(ValueError):
        midpoint_model_columns(metadata, representation="combined", variant="both500", conversion_mode="midpoint_side")
