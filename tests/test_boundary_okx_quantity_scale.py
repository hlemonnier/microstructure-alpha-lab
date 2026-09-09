import copy

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_okx_quantity_scale import quantity_scale_features  # noqa: E402


def _source():
    depth = np.ones((3, 1, 100, 4))
    depth[:, :, :, (1, 3)] *= 12
    return {"depth": depth, "order_counts": np.full((3, 1, 100, 2), 3, dtype=np.int64),
        "decision_times": np.array([1000, 2000, 3000], dtype=np.int64), "delays_ms": np.array([100]),
        "query_cutoffs": np.array([[900], [1900], [2900]]), "publisher_times": np.array([[899], [1899], [2899]]),
        "known_depths": np.full((3, 1, 2), 400, dtype=np.int16)}


def _features(source):
    return quantity_scale_features(source, levels=100, delay_ms=100)


def test_absolute_quantity_and_mean_quantity_have_correct_distinct_information():
    source = _source()
    original = _features(source)
    np.testing.assert_allclose(original.depth_100_log_source_quantity, np.log(2400))
    np.testing.assert_allclose(original.count_100_log_mean_source_quantity, np.log(4))
    changed = copy.deepcopy(source)
    changed["order_counts"] *= 2
    other = _features(changed)
    quantities = [c for c in original if c.startswith("depth_")]
    pd.testing.assert_frame_equal(original[quantities], other[quantities], check_exact=True)
    np.testing.assert_allclose(other.count_100_log_mean_source_quantity, np.log(2))


def test_quantity_unit_rescaling_has_an_explicit_logarithmic_effect():
    source = _source()
    original = _features(source)
    source["depth"][:, :, :, (1, 3)] *= 1024
    transformed = _features(source)
    np.testing.assert_allclose(transformed - original, np.log(1024), atol=1e-12, rtol=1e-12)


def test_unknown_or_late_scale_does_not_fill_missing_depth():
    source = _source()
    source["known_depths"][1] = 25
    result = _features(source)
    assert (result.iloc[1] == 0).all()
    source["publisher_times"][0] = 900
    with pytest.raises(ValueError):
        _features(source)
