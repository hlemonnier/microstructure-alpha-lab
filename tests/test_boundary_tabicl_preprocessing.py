import copy
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sklearn.preprocessing import FunctionTransformer, PowerTransformer  # noqa: E402

from lob_forge.boundary_tabicl_preprocessing import RowLocalPowerFallback, fallback_counts, install_row_local_power_fallback  # noqa: E402


def unstable_pipeline():
    power = PowerTransformer().fit(np.array([[0.], [.1], [.2], [.3]]))
    # A large fitted exponent makes an extreme later query overflow, while
    # training-range values and moderate extrapolation remain finite.
    power.lambdas_[:] = 400
    return SimpleNamespace(normalization_method="power", normalizer_=power,
        standard_scaler_=FunctionTransformer(), outlier_remover_=FunctionTransformer(),
        X_min_=np.array([[0.]]), X_max_=np.array([[.3]]), X_transformed_=np.zeros((4, 1)))


def test_overflow_fallback_does_not_clip_unrelated_earlier_query_rows():
    original = unstable_pipeline()
    safe = RowLocalPowerFallback(original)
    ordinary = np.array([[1.], [.2], [.1]])
    expected = original.normalizer_.transform(ordinary)
    np.testing.assert_array_equal(safe.transform(ordinary), expected)
    augmented = np.vstack([ordinary, [[100.]]])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            original.normalizer_.transform(augmented)
    result = safe.transform(augmented)
    np.testing.assert_array_equal(result[:3], expected)
    np.testing.assert_array_equal(result[-1:], original.normalizer_.transform(np.array([[.3]])))
    assert safe.row_fallbacks == 1
    # The former all-row clipping fallback alters this valid extrapolation.
    globally_clipped = original.normalizer_.transform(np.clip(augmented, original.X_min_, original.X_max_))
    assert globally_clipped[0, 0] != expected[0, 0]


def test_row_local_fallback_is_batch_partition_invariant_and_serializable(tmp_path):
    import pickle

    safe = RowLocalPowerFallback(unstable_pipeline())
    values = np.array([[1.], [100.], [.2], [90.], [.1]])
    whole = safe.transform(values)
    split = np.concatenate([safe.transform(values[:2]), safe.transform(values[2:])])
    single = np.concatenate([safe.transform(values[i:i + 1]) for i in range(len(values))])
    np.testing.assert_array_equal(whole, split)
    np.testing.assert_array_equal(whole, single)
    path = tmp_path / "fitted_local_adapter.pkl"
    path.write_bytes(pickle.dumps(safe))
    restored = pickle.loads(path.read_bytes())
    np.testing.assert_array_equal(whole, restored.transform(values))
    np.testing.assert_array_equal(safe.X_transformed_, copy.deepcopy(safe).X_transformed_)


def test_installation_preserves_other_preprocessors_and_resets_only_counters():
    other = object()
    model = SimpleNamespace(ensemble_generator_=SimpleNamespace(preprocessors_={"none": other, "power": unstable_pipeline()}))
    install_row_local_power_fallback(model)
    adapted = model.ensemble_generator_.preprocessors_["power"]
    install_row_local_power_fallback(model)
    assert model.ensemble_generator_.preprocessors_["power"] is adapted
    assert model.ensemble_generator_.preprocessors_["none"] is other
    adapted.transform(np.array([[100.]]))
    assert fallback_counts(model, reset=True) == {"power": 1}
    assert fallback_counts(model) == {"power": 0}
    with pytest.raises(ValueError, match="finite"):
        adapted.transform(np.array([[np.nan]]))
