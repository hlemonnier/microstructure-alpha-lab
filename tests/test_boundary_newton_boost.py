import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_newton_boost import NewtonBoostForecaster, fit_newton_boost  # noqa: E402


def _require_runtime():
    try:
        pytest.importorskip("xgboost")
    except Exception as exc:
        if type(exc).__name__ == "XGBoostError" and "could not be loaded" in str(exc):
            pytest.skip("Optional XGBoost native OpenMP runtime is unavailable")
        raise


def test_balanced_newton_tree_recovers_different_asset_priors_and_native_checkpoint(tmp_path):
    _require_runtime()
    x = {s: pd.DataFrame({"observed": np.zeros(600)}) for s in ("BTCUSDT", "ETHUSDT")}
    y = {"BTCUSDT": np.repeat([-1, 0, 1], [100, 200, 300]),
         "ETHUSDT": np.repeat([-1, 0, 1], [300, 200, 100])}
    model = fit_newton_boost(x, y, max_depth=3, class_balanced=True, rounds=30)
    model.save(tmp_path)
    restored = NewtonBoostForecaster.load(tmp_path)
    for i, symbol in enumerate(model.symbols):
        prediction = model.predict_proba(x[symbol].iloc[:10], symbol)
        np.testing.assert_allclose(prediction, np.tile(model.priors[i], (10, 1)), atol=2e-7)
        np.testing.assert_array_equal(prediction, restored.predict_proba(x[symbol].iloc[:10], symbol))


@pytest.mark.parametrize("balanced", [True, False])
def test_newton_class_mapping_and_training_only_transform_preserve_input(balanced):
    _require_runtime()
    y = np.repeat([-1, 0, 1], [200, 400, 600])
    frame = pd.DataFrame({"observed": y.astype(float), "constant": np.ones(len(y))})
    original = frame.copy(deep=True)
    x = {"BTCUSDT": frame, "ETHUSDT": frame.iloc[::-1].reset_index(drop=True)}
    labels = {"BTCUSDT": y, "ETHUSDT": y[::-1]}
    model = fit_newton_boost(x, labels, max_depth=3, class_balanced=balanced, rounds=60, min_child_weight=1)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    p = model.predict_proba(frame, "BTCUSDT")
    assert np.mean(p.argmax(axis=1) - 1 == y) > 0.99
    np.testing.assert_allclose(p.sum(axis=1), 1)
    low, high = np.quantile(pd.concat(list(x.values())).to_numpy(), [0.001, 0.999], axis=0)
    np.testing.assert_array_equal(model.lower, low)
    np.testing.assert_array_equal(model.upper, high)
    with pytest.raises(ValueError, match="schema"):
        model.predict_proba(frame.iloc[:, ::-1], "BTCUSDT")
    with pytest.raises(ValueError, match="Finite"):
        model.predict_proba(frame.assign(observed=np.inf), "BTCUSDT")
