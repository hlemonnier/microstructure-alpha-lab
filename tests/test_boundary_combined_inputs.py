import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_combined_inputs import combine_observed_frames, original_combined_columns  # noqa: E402


def test_source_union_preserves_original_fields_indices_and_prefixes():
    original = pd.DataFrame({"observed": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    depth = original.assign(foreign_depth=[0.5, 0.6, 0.7])
    spot = original.assign(aux_trade=[-0.2, 0.2, 0.1])
    combined = combine_observed_frames(depth, spot)
    assert list(combined.columns) == ["observed", "foreign_depth", "aux_trade"]
    pd.testing.assert_frame_equal(combined[original_combined_columns(combined)], original, check_exact=True)
    pd.testing.assert_frame_equal(combine_observed_frames(depth.iloc[:2], spot.iloc[:2]), combined.iloc[:2], check_exact=True)
    assert "aux_trade" not in depth
    assert "foreign_depth" not in spot


def test_source_union_rejects_changed_old_fields_reordered_clocks_and_duplicate_additions():
    depth = pd.DataFrame({"observed": [1.0, 2.0], "foreign_depth": [0.1, 0.2]})
    spot = pd.DataFrame({"observed": [1.0, 2.0], "aux_trade": [0.3, 0.4]})
    with pytest.raises(AssertionError):
        combine_observed_frames(depth, spot.assign(observed=[1.0, 2.00000001]))
    with pytest.raises(AssertionError):
        combine_observed_frames(depth, spot.iloc[::-1])
    with pytest.raises(ValueError, match="disjoint"):
        combine_observed_frames(depth, spot.assign(foreign_depth=[0.3, 0.4]))
    with pytest.raises(ValueError, match="finite"):
        combine_observed_frames(depth, spot.assign(aux_trade=np.nan))
