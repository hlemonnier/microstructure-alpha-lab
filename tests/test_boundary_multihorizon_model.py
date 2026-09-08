import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural  # noqa: E402
from lob_forge.boundary_multihorizon_model import (  # noqa: E402
    MultihorizonForecaster,
    build_multihorizon_network,
    fit_multihorizon_neural,
    multihorizon_loss,
    multihorizon_weights,
)


def partitions():
    rng = np.random.default_rng(33)
    tx, ty, valid, vx, vy = {}, {}, {}, {}, {}
    for symbol in SYMBOLS:
        tx[symbol] = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
        vx[symbol] = pd.DataFrame(rng.normal(size=(24, 3)), columns=list("abc"))
        ty[symbol] = np.column_stack([np.roll(np.resize(np.array([-1, 0, 1]), 60), head) for head in range(5)])
        vy[symbol] = np.resize(np.array([-1, 0, 1]), 24)
        valid[symbol] = np.ones((60, 5), dtype=bool)
        valid[symbol][:6, 4] = False
        ty[symbol][:6, 4] = -2
    return tx, ty, valid, vx, vy


def test_head_weights_equalize_valid_asset_class_mass_and_mask_missing_labels():
    _, targets, valid, _, _ = partitions()
    y, available = np.concatenate(list(targets.values())), np.concatenate(list(valid.values()))
    a = np.repeat([0, 1], 60)
    priors, weights = multihorizon_weights(y, available, a)
    assert len(priors) == 5
    np.testing.assert_array_equal(weights[~available], 0)
    for head in range(5):
        for asset in range(2):
            for label in (-1, 0, 1):
                mask = available[:, head] & (a == asset) & (y[:, head] == label)
                assert np.isclose(weights[mask, head].sum(), len(y) / 6)
    with pytest.raises(ValueError, match="Primary"):
        multihorizon_weights(y, np.zeros_like(available), a)
    with pytest.raises(ValueError, match="three classes"):
        bad = y.copy()
        bad[available] = 0
        multihorizon_weights(bad, available, a)


def test_auxiliary_loss_exact_formula_and_gradients_reach_shared_and_auxiliary_weights():
    torch.manual_seed(82)
    network = build_multihorizon_network(4, hidden_size=8)
    x = torch.randn(6, 4)
    targets = torch.arange(30).reshape(6, 5) % 3
    weights = torch.ones(6, 5)
    weights[0, 2] = 0
    logits = network.all_horizons(x)
    raw = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="none").reshape(6, 5)
    expected = (.75 * raw[:, 0] + .25 * (raw[:, 1:] * weights[:, 1:]).mean(dim=1)).mean()
    actual = multihorizon_loss(logits, targets, weights, auxiliary_weight=.25)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    for parameter in network.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    changed = targets.clone()
    changed[0, 2] = (changed[0, 2] + 1) % 3
    torch.testing.assert_close(multihorizon_loss(logits.detach(), changed, weights, auxiliary_weight=.25), actual.detach(), atol=0, rtol=0)


def test_zero_auxiliary_training_exactly_reproduces_original_weights_history_and_predictions():
    tx, ty, valid, vx, vy = partitions()
    original, oh = fit_confirm_neural(tx, {s: ty[s][:, 0] for s in SYMBOLS}, vx, vy)
    model, history = fit_multihorizon_neural(tx, ty, valid, vx, vy, auxiliary_weight=0)
    assert history["history"] == oh["history"]
    assert history["best_epoch"] == oh["best_epoch"]
    for name, value in original.network.state_dict().items():
        torch.testing.assert_close(model.network.base.state_dict()[name], value, atol=0, rtol=0)
    for asset, symbol in enumerate(SYMBOLS):
        np.testing.assert_array_equal(model.predict_proba(vx[symbol], asset), original.predict_proba(vx[symbol], asset))


@pytest.mark.parametrize("auxiliary_weight", [.25, .5])
def test_joint_supervision_fits_finitely_and_checkpoint_main_forecasts_recover(tmp_path, auxiliary_weight):
    data = partitions()
    model, history = fit_multihorizon_neural(*data, auxiliary_weight=auxiliary_weight)
    assert len(history["history"]) == 12
    model.save(tmp_path)
    restored = MultihorizonForecaster.load(tmp_path)
    for asset, symbol in enumerate(SYMBOLS):
        p = model.predict_proba(data[3][symbol], asset)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(p, restored.predict_proba(data[3][symbol], asset))
