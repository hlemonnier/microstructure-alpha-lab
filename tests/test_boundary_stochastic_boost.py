import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("catboost")

from lob_forge.boundary_stochastic_boost import (  # noqa: E402
    VARIANTS, StochasticBoostForecaster, chronological_training_pool, fit_stochastic_boost,
)


def fixture():
    rng = np.random.default_rng(25)
    features, labels, clocks = {}, {}, {}
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        y = np.resize([-1, 0, 0, 0, 0, 1] if asset == 0 else [-1, -1, 0, 0, 0, 1], 90)
        x = np.column_stack([y + rng.normal(0, .4, len(y)), rng.normal(size=(len(y), 2))])
        order = rng.permutation(len(y))
        features[symbol] = pd.DataFrame(x[order], columns=["signal", "noise1", "noise2"])
        labels[symbol] = y[order]
        clocks[symbol] = (1700000000000 + 4000 * np.arange(len(y), dtype=np.int64))[order]
    return features, labels, clocks


def test_chronological_packing_preserves_features_labels_and_global_cell_weights():
    features, labels, times = fixture()
    packed = chronological_training_pool(features, labels, times)
    expected = []
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        for row, clock in enumerate(times[symbol]):
            expected.append((int(clock), asset, row, symbol))
    expected.sort()
    for i, (clock, asset, row, symbol) in enumerate(expected):
        assert packed["times"][i] == clock and packed["assets"][i] == asset
        assert packed["labels"][i] == labels[symbol][row]
        raw = features[symbol].iloc[row].to_numpy()
        np.testing.assert_array_equal(packed["matrix"][i, :-1], np.clip(raw, packed["lower"], packed["upper"]).astype(np.float32))
        assert packed["matrix"][i, -1] == asset
        count = np.sum(labels[symbol] == labels[symbol][row])
        assert packed["weights"][i] == pytest.approx(180 / (6 * count), abs=1e-15)
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        np.testing.assert_array_equal(packed["priors"][asset], [(labels[symbol] == c).mean() for c in (-1, 0, 1)])
    assert packed["weights"].sum() == pytest.approx(180.)


def test_all_native_variants_prior_recovery_effective_settings_and_checkpoint(tmp_path):
    features, labels, times = fixture()
    queries = features["BTCUSDT"].iloc[:17].copy()
    for variant in VARIANTS:
        model = fit_stochastic_boost(features, labels, times, variant=variant, iterations=12)
        settings = model.effective_parameters
        assert settings["boosting_type"] == VARIANTS[variant]["boosting_type"]
        assert settings["depth"] == VARIANTS[variant]["depth"]
        assert settings["verified_serialized_configuration"] == {"has_time": True, "thread_count": 2, "task_type": "CPU"}
        assert model.booster.tree_count_ == 12
        if variant == "langevin6":
            assert settings["langevin"] is True
            assert settings["diffusion_temperature"] == 180
            assert settings["model_shrink_rate"] == pytest.approx(1 / 360, rel=1e-6)
        p = model.predict_proba(queries, "BTCUSDT")
        raw = model.booster.predict(model.matrix(queries, "BTCUSDT"), prediction_type="RawFormulaVal", thread_count=2)
        raw -= raw.max(1, keepdims=True)
        expected = np.exp(raw) * model.priors[0]
        expected /= expected.sum(1, keepdims=True)
        np.testing.assert_allclose(p, expected, rtol=0, atol=3e-16)
        changed = queries.copy()
        changed.iloc[6:] = 1e100
        np.testing.assert_array_equal(p[:6], model.predict_proba(changed, "BTCUSDT")[:6])
        np.testing.assert_array_equal(p[:6], model.predict_proba(queries.iloc[:6], "BTCUSDT"))
        individual = np.concatenate([model.predict_proba(queries.iloc[i:i + 1], "BTCUSDT") for i in range(len(queries))])
        np.testing.assert_array_equal(p, individual)
        model.save(tmp_path / variant)
        restored = StochasticBoostForecaster.load(tmp_path / variant)
        np.testing.assert_array_equal(p, restored.predict_proba(queries, "BTCUSDT"))
        np.testing.assert_array_equal(model.priors, restored.priors)
        with pytest.raises(ValueError, match="schema"):
            restored.predict_proba(queries.iloc[:, ::-1], "BTCUSDT")


def test_native_training_rejects_ambiguous_clocks_and_missing_class_support():
    features, labels, times = fixture()
    fractional = {**times, "BTCUSDT": times["BTCUSDT"].astype(float) + .5}
    with pytest.raises(ValueError, match="integer"):
        chronological_training_pool(features, labels, fractional)
    repeated = {**times, "BTCUSDT": np.repeat(times["BTCUSDT"][0], len(times["BTCUSDT"]))}
    with pytest.raises(ValueError, match="unique"):
        chronological_training_pool(features, labels, repeated)
    missing = {**labels, "BTCUSDT": np.zeros_like(labels["BTCUSDT"])}
    with pytest.raises(ValueError, match="all three classes"):
        chronological_training_pool(features, missing, times)
