import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_residual_boost import ResidualBoostForecaster, fit_residual_boost, residual_base_margin, select_residual_rounds  # noqa: E402


def _require_runtime():
    try:
        pytest.importorskip("xgboost")
    except Exception as exc:
        if type(exc).__name__ == "XGBoostError" and "could not be loaded" in str(exc):
            pytest.skip("Optional XGBoost native OpenMP runtime is unavailable")
        raise


def test_residual_weighted_logit_rebasing_preserves_natural_forecasts():
    probabilities = np.array([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]])
    priors = np.array([[0.2, 0.7, 0.1], [0.4, 0.5, 0.1]])
    for balanced in (True, False):
        margin = residual_base_margin(probabilities, priors, class_balanced=balanced)
        q = np.exp(margin - margin.max(axis=1, keepdims=True))
        q /= q.sum(axis=1, keepdims=True)
        natural = q * priors if balanced else q
        natural /= natural.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(natural, probabilities, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="strictly positive"):
        residual_base_margin([[0, 0.5, 0.5]], [0.2, 0.6, 0.2], class_balanced=True)


@pytest.mark.parametrize("balanced", [True, False])
def test_residual_correction_keeps_a_calibrated_constant_parent_and_zero_is_exact(tmp_path, balanced):
    _require_runtime()
    x = {s: pd.DataFrame({"constant": np.zeros(600)}) for s in ("BTCUSDT", "ETHUSDT")}
    y = {"BTCUSDT": np.repeat([-1, 0, 1], [100, 200, 300]), "ETHUSDT": np.repeat([-1, 0, 1], [300, 200, 100])}
    base = {s: np.tile([(y[s] == k).mean() for k in (-1, 0, 1)], (600, 1)) for s in x}
    model = fit_residual_boost(x, y, base, max_depth=2, class_balanced=balanced, rounds=30)
    model.selected_rounds = 16
    model.save(tmp_path)
    restored = ResidualBoostForecaster.load(tmp_path)
    for s in x:
        np.testing.assert_array_equal(model.predict_proba(x[s], s, base[s], rounds=0), base[s])
        np.testing.assert_allclose(model.predict_proba(x[s], s, base[s]), base[s], atol=3e-7)
        for count in (0, 16, 30):
            np.testing.assert_array_equal(model.predict_proba(x[s], s, base[s], rounds=count), restored.predict_proba(x[s], s, base[s], rounds=count))


def test_residual_tree_learns_missing_information_and_selection_includes_parent():
    _require_runtime()
    labels = np.repeat([-1, 0, 1], [300, 300, 300])
    x = {s: pd.DataFrame({"information": labels.astype(float), "constant": 1.0}) for s in ("BTCUSDT", "ETHUSDT")}
    y = {s: labels.copy() for s in x}
    base = {s: np.tile([0.1, 0.8, 0.1], (len(labels), 1)) for s in x}
    model = fit_residual_boost(x, y, base, max_depth=2, class_balanced=True, rounds=64, min_child_weight=1)
    prior = {s: np.array([1/3, 1/3, 1/3]) for s in x}
    chosen = select_residual_rounds(model, x, y, base, prior, candidates=(0, 16, 64))
    assert chosen["selected_rounds"] > 0
    for s in x:
        assert np.mean(model.predict_proba(x[s], s, base[s]).argmax(axis=1) - 1 == labels) > 0.99
    reversed_labels = {s: -labels for s in x}
    perfect = {s: np.eye(3)[reversed_labels[s] + 1] * 0.97 + 0.01 for s in x}
    zero = select_residual_rounds(model, x, reversed_labels, perfect, prior, candidates=(0, 16, 64))
    assert zero["selected_rounds"] == 0
    np.testing.assert_array_equal(model.predict_proba(x["BTCUSDT"], "BTCUSDT", perfect["BTCUSDT"]), perfect["BTCUSDT"])
    with pytest.raises(ValueError, match="schema"):
        model.predict_proba(x["BTCUSDT"].iloc[:, ::-1], "BTCUSDT", base["BTCUSDT"])
