import copy

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural  # noqa: E402
from lob_forge.boundary_embedding import (  # noqa: E402
    EmbeddingForecaster,
    QuantileBins,
    build_piecewise_embedding,
    fit_embedding_neural,
    mean_member_natural_probabilities,
)


def synthetic_partitions():
    rng = np.random.default_rng(821)
    tx, ty, vx, vy = {}, {}, {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        for x, y, count in ((tx, ty, 60), (vx, vy, 24)):
            labels = np.resize(np.array([-1, 0, 1]), count)
            values = rng.normal(size=(count, 3))
            values[:, 0] += labels + asset / 3
            values[:, 2] = 7
            x[symbol] = pd.DataFrame(values, columns=["signal", "noise", "constant"])
            y[symbol] = labels
    return tx, ty, vx, vy


def test_quantile_bins_interpolate_deduplicate_and_clip_without_refitting():
    bins = QuantileBins.fit(np.array([[0, 0], [1, 0], [2, 1], [3, 1], [4, 1]]), bins=4)
    assert [len(e) - 1 for e in bins.edges] == [4, 1]
    frozen = copy.deepcopy(bins.edges)
    index, fraction = bins.transform(np.array([[-100, -1], [0.5, 0.25], [2, 1], [100, 100]]))
    np.testing.assert_array_equal(index, [[0, 0], [0, 0], [2, 0], [3, 0]])
    np.testing.assert_array_equal(fraction, [[0, 0], [.5, .25], [0, 1], [1, 1]])
    assert fraction.dtype == np.float32
    for actual, expected in zip(bins.edges, frozen):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="constant"):
        QuantileBins.fit(np.ones((10, 2)))
    with pytest.raises(ValueError, match="Finite"):
        bins.transform([[np.nan, 1]])


def test_sparse_embedding_matches_dense_values_and_all_parameter_gradients():
    torch.manual_seed(323)
    sparse = build_piecewise_embedding([4, 1, 3], dimensions=4).double()
    dense = build_piecewise_embedding([4, 1, 3], dimensions=4).double()
    dense.load_state_dict(sparse.state_dict())
    indices = torch.tensor([[0, 0, 0], [1, 0, 1], [3, 0, 2], [2, 0, 1]])
    fractions = torch.tensor([[0, 0, 0], [.5, .25, .5], [1, 1, 1], [0, .8, .7]], dtype=torch.float64)
    basis = torch.zeros(4, 3, 4, dtype=torch.float64)
    for row in range(4):
        for feature in range(3):
            loc = indices[row, feature]
            basis[row, feature, :loc] = 1
            basis[row, feature, loc] = fractions[row, feature]
    actual = sparse(indices, fractions)
    expected = torch.einsum("bfk,fkd->bfd", basis, dense.weight) + dense.bias
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-14)
    multiplier = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape) / 20
    (actual.square() * multiplier).sum().backward()
    (expected.square() * multiplier).sum().backward()
    torch.testing.assert_close(sparse.weight.grad, dense.weight.grad, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(sparse.bias.grad, dense.bias.grad, atol=1e-14, rtol=1e-13)
    assert torch.count_nonzero(sparse.weight.grad[1, 1:]) == 0


def test_natural_member_mean_differs_from_recovering_the_mean_weighted_posterior():
    q = np.array([[[.85, .10, .05], [.05, .15, .80]], [[.1, .8, .1], [.4, .2, .4]]])
    pi = np.array([.15, .70, .15])
    expected = np.stack([row * pi / (row * pi).sum(axis=1, keepdims=True) for row in q]).mean(axis=1)
    actual = mean_member_natural_probabilities(q, pi)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    alternative = q.mean(axis=1) * pi
    alternative /= alternative.sum(axis=1, keepdims=True)
    assert not np.allclose(actual, alternative)
    with pytest.raises(ValueError, match="Normalized"):
        mean_member_natural_probabilities(q * 2, pi)
    with pytest.raises(ValueError, match="positive"):
        mean_member_natural_probabilities(q, [0, .5, .5])


def test_scalar_one_exactly_reproduces_original_training_history_and_forecasts():
    partitions = synthetic_partitions()
    original, original_history = fit_confirm_neural(*partitions)
    current, current_history = fit_embedding_neural(*partitions, architecture="scalar1")
    assert current_history["history"] == original_history["history"]
    assert current_history["best_epoch"] == original_history["best_epoch"]
    np.testing.assert_array_equal(current.active, [True, True, False])
    for name, state in original.network.state_dict().items():
        torch.testing.assert_close(state, current.network.state_dict()[name], rtol=0, atol=0)
    for asset, symbol in enumerate(SYMBOLS):
        np.testing.assert_array_equal(current.predict_proba(partitions[2][symbol], asset),
                                      original.predict_proba(partitions[2][symbol], asset))


@pytest.mark.parametrize("architecture", ["scalar8", "piecewise1", "piecewise8"])
def test_ensemble_fit_uses_training_knots_and_reloads_exactly(tmp_path, architecture):
    partitions = synthetic_partitions()
    # Extreme validation values cannot participate in knot or active-column fitting.
    for symbol in SYMBOLS:
        partitions[2][symbol].iloc[0] = [1e6, -1e6, 100]
    model, history = fit_embedding_neural(*partitions, architecture=architecture)
    assert len(history["history"]) == 12
    assert 1 <= history["best_epoch"] <= 12
    np.testing.assert_array_equal(model.active, [True, True, False])
    if architecture.startswith("piecewise"):
        training = pd.concat(list(partitions[0].values())).to_numpy()[:, :2]
        for column, edges in enumerate(model.normalizer.edges):
            assert edges[0] == training[:, column].min()
            assert edges[-1] == training[:, column].max()
    model.save(tmp_path)
    restored = EmbeddingForecaster.load(tmp_path)
    for asset, symbol in enumerate(SYMBOLS):
        p = model.predict_proba(partitions[2][symbol], asset)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(restored.predict_proba(partitions[2][symbol], asset), p)
    with pytest.raises(ValueError, match="schema"):
        model.predict_proba(partitions[2][SYMBOLS[0]].iloc[:, ::-1], 0)
    with pytest.raises(ValueError, match="integer"):
        model.predict_proba(partitions[2][SYMBOLS[0]], 0.5)
