import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_distribution_model import DistributionForecaster, distribution_loss, fit_distribution_neural  # noqa: E402
from lob_forge.boundary_distribution_targets import COARSE_CLASSES  # noqa: E402


def test_distribution_scores_have_zero_expected_gradient_at_weighted_posterior():
    natural = torch.tensor([.04,.06,.1,.1,.3,.1,.1,.12,.08], dtype=torch.float64)
    priors = torch.tensor([.25,.5,.25], dtype=torch.float64)
    weights = 1 / priors[torch.tensor(COARSE_CLASSES + 1, dtype=torch.long)]
    q = natural * weights
    q /= q.sum()
    for fine, ranked in [(0,0),(.25,0),(1,0),(1,1)]:
        logits = q.log().clone().requires_grad_(True)
        losses = distribution_loss(logits[None,:].expand(9,-1), torch.arange(9), fine_weight=fine, ranked_weight=ranked)
        (losses * natural * weights).sum().backward()
        np.testing.assert_allclose(logits.grad.numpy(), 0, rtol=0, atol=1e-15)


def test_distribution_fit_roundtrip_preserves_schema_normalizer_and_probabilities(tmp_path):
    rng = np.random.default_rng(45)
    bins = {s: np.tile(np.arange(9), 8) for s in ("BTCUSDT", "ETHUSDT")}
    x = {s: pd.DataFrame({"x": rng.normal(size=72), "constant": 2.0}) for s in bins}
    original = {s: f.copy(deep=True) for s,f in x.items()}
    y = {s: COARSE_CLASSES[b] for s,b in bins.items()}
    model, history = fit_distribution_neural(x, bins, x, y, fine_weight=.25, ranked_weight=1)
    assert 1 <= history["best_epoch"] <= 12
    np.testing.assert_array_equal(model.active, [True,False])
    model.save(tmp_path / "model")
    restored = DistributionForecaster.load(tmp_path / "model")
    for asset,symbol in enumerate(bins):
        p = model.predict_proba(x[symbol], asset)
        np.testing.assert_array_equal(p, restored.predict_proba(x[symbol], asset))
        np.testing.assert_allclose(p.sum(axis=1), 1)
        pd.testing.assert_frame_equal(x[symbol], original[symbol], check_exact=True)
        with pytest.raises(ValueError):
            restored.predict_proba(x[symbol].assign(target=bins[symbol]), asset)
