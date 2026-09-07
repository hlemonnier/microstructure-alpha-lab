import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network  # noqa: E402


def test_training_weights_equalize_asset_and_class_mass():
    labels = np.array([-1, -1, 0, 1, -1, 0, 0, 0, 1, 1])
    assets = np.array([0] * 4 + [1] * 6)
    priors, weights = balanced_asset_weights(labels, assets)
    for asset in [0, 1]:
        assert np.isclose(priors[asset].sum(), 1)
        for label in [-1, 0, 1]:
            assert np.isclose(weights[(assets == asset) & (labels == label)].sum(), len(labels) / 6)


def test_member_forecasts_are_diverse_normalized_and_recoverable(tmp_path):
    from sklearn.preprocessing import QuantileTransformer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(100, 4)), columns=list("abcd"))
    normalizer = QuantileTransformer(n_quantiles=20, output_distribution="normal", random_state=42).fit(x.to_numpy())
    network = build_member_network(5, members=4, hidden_size=8)
    model = PooledForecaster(
        network,
        normalizer,
        np.ones(4, dtype=bool),
        list(x.columns),
        {0: np.array([0.2, 0.5, 0.3]), 1: np.array([0.3, 0.3, 0.4])},
        4,
        8,
    )
    logits = network(torch.from_numpy(model.matrix(x, 0))).detach().numpy()
    assert logits.shape == (100, 4, 3)
    assert not np.allclose(logits[:, 0], logits[:, 1])
    expected = model.predict_proba(x, 0)
    model.save(tmp_path)
    restored = PooledForecaster.load(tmp_path)
    np.testing.assert_array_equal(restored.predict_proba(x, 0), expected)
    np.testing.assert_allclose(expected.sum(axis=1), 1)
