import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from lob_forge.boundary_weighted_boost import asset_class_weights, recover_natural_posterior  # noqa: E402


def test_weighted_loss_gives_equal_asset_class_mass_and_unit_mean_weight():
    y = np.array([-1, 0, 0, 0, 1, 1, -1, -1, 0, 1])
    a = np.array([0] * 6 + [1] * 4)
    priors, weights = asset_class_weights(y, a, class_balanced=True)
    assert np.isclose(weights.mean(), 1)
    for asset in [0, 1]:
        np.testing.assert_allclose(priors[asset], [(y[a == asset] == k).mean() for k in [-1, 0, 1]])
        for label in [-1, 0, 1]:
            assert np.isclose(weights[(a == asset) & (y == label)].sum(), len(y) / 6)
    _, natural_weights = asset_class_weights(y, a, class_balanced=False)
    assert np.isclose(natural_weights[a == 0].sum(), len(y) / 2)
    assert len(np.unique(natural_weights[a == 0])) == 1
    with pytest.raises(ValueError, match="all three classes"):
        asset_class_weights(y[y != 1], a[y != 1], class_balanced=True)


def test_natural_probability_recovery_inverts_class_weighting_per_asset():
    p = np.array([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]])
    for prior in [np.array([0.1, 0.8, 0.1]), np.array([0.4, 0.2, 0.4])]:
        q = p / prior
        q /= q.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(recover_natural_posterior(q, prior, class_balanced=True), p)
        np.testing.assert_allclose(recover_natural_posterior(p, prior, class_balanced=False), p)
    with pytest.raises(ValueError, match="positive"):
        recover_natural_posterior(p, np.array([0, 0.5, 0.5]), class_balanced=True)
