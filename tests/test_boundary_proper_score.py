import copy

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural  # noqa: E402
from lob_forge.boundary_pooled import PooledForecaster  # noqa: E402
from lob_forge.boundary_proper_score import VARIANTS, fit_proper_score, predict_matrix, proper_score_loss  # noqa: E402


def independent_losses(q, target, variant):
    if variant == "log":
        return -np.log(q[np.arange(len(q)), target])
    if variant == "brier":
        return 1.5 * ((q - np.eye(3)[target]) ** 2).sum(1)
    beta = 2 if variant == "spherical2" else 4
    norm = (q ** beta).sum(1) ** (1 / beta)
    return 3 ** ((beta - 1) / beta) / (beta - 1) * (1 - (q[np.arange(len(q)), target] / norm) ** (beta - 1))


def test_losses_match_independent_probability_norm_formulas_and_log_control():
    u = torch.tensor([[.3, -.8, .4], [-1.2, .6, 1.1], [0., 0., 0.]], dtype=torch.float64)
    target = torch.tensor([0, 1, 2])
    q = np.exp(u.numpy() - u.numpy().max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for variant in VARIANTS:
        expected = independent_losses(q, target.numpy(), variant)
        np.testing.assert_allclose(proper_score_loss(u, target, variant).numpy(), expected, rtol=2e-14, atol=2e-15)
    assert torch.equal(proper_score_loss(u, target, "log"), torch.nn.functional.cross_entropy(u, target, reduction="none"))


def test_logit_gradients_match_closed_forms_finite_differences_and_uniform_scale():
    torch.set_num_threads(2)
    target = torch.tensor([0, 2])
    for variant in VARIANTS:
        u = torch.tensor([[.3, -.8, .4], [-1.2, .6, 1.1]], dtype=torch.float64, requires_grad=True)
        actual = torch.autograd.grad(proper_score_loss(u, target, variant).sum(), u)[0]
        q = torch.softmax(u.detach(), -1)
        one_hot = torch.nn.functional.one_hot(target, 3).to(torch.float64)
        if variant == "log":
            expected = q - one_hot
        elif variant == "brier":
            residual = q - one_hot
            expected = 3 * q * (residual - (q * residual).sum(1, keepdim=True))
        else:
            beta = 2 if variant == "spherical2" else 4
            alpha = (beta - 1) / beta
            s = torch.softmax(beta * u.detach(), -1)
            expected = 3 ** alpha * s[torch.arange(2), target, None] ** alpha * (s - one_hot)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)
        assert torch.autograd.gradcheck(lambda x: proper_score_loss(x, target, variant), (u,))
        uniform = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(proper_score_loss(uniform, torch.arange(3), variant).sum(), uniform)[0]
        torch.testing.assert_close(gradient, torch.full((3, 3), 1 / 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64), rtol=0, atol=3e-16)


def test_population_risk_has_same_unique_probability_target_and_weighted_prior():
    rng = np.random.default_rng(7)
    candidates = rng.dirichlet([.5, .5, .5], size=500)
    for eta in (np.array([.13, .31, .56]), np.array([.82, .06, .12]), np.full(3, 1 / 3)):
        for variant in VARIANTS:
            q = np.vstack([eta, candidates])
            risks = sum(eta[c] * independent_losses(q, np.full(len(q), c), variant) for c in range(3))
            assert np.all(risks[1:] > risks[0])
            if variant == "brier":
                np.testing.assert_allclose(risks - risks[0], 1.5 * ((q - eta) ** 2).sum(1), rtol=2e-12, atol=1e-15)
            elif variant.startswith("spherical"):
                beta = 2 if variant == "spherical2" else 4
                expected = 3 ** ((beta - 1) / beta) / (beta - 1) * (1 - (eta ** beta).sum() ** (1 / beta))
                assert risks[0] == pytest.approx(expected, abs=2e-15)
            logits = torch.tensor(np.tile(np.log(eta), (3, 1)), requires_grad=True)
            loss = (proper_score_loss(logits, torch.arange(3), variant) * torch.tensor(eta)).sum()
            gradient = torch.autograd.grad(loss, logits)[0].sum(0)
            torch.testing.assert_close(gradient, torch.zeros(3, dtype=torch.float64), rtol=0, atol=3e-16)
    natural, prior = np.array([.22, .61, .17]), np.array([.1, .7, .2])
    balanced = natural / prior
    balanced /= balanced.sum()
    recovered = balanced * prior
    recovered /= recovered.sum()
    # Two probability normalizations introduce ordinary float64 rounding.
    np.testing.assert_allclose(recovered, natural, rtol=0, atol=2 * np.finfo(np.float64).eps)
    # Raw GCE probabilities would instead elicit a sharpened vector. Verify the
    # registered spherical parameterization reports the inverse link, q=eta.
    for beta in (2, 4):
        escort = balanced ** beta / (balanced ** beta).sum()
        assert not np.allclose(escort, balanced)
        restored = escort ** (1 / beta)
        np.testing.assert_allclose(restored / restored.sum(), balanced, rtol=2e-15, atol=0)


def test_loss_shift_class_permutation_and_extreme_logit_stability():
    u = torch.tensor([[.3, -.8, .4], [-1.2, .6, 1.1]], dtype=torch.float64)
    target = torch.tensor([0, 2])
    perm = torch.tensor([2, 0, 1])
    inverse = torch.argsort(perm)
    for variant in VARIANTS:
        before = proper_score_loss(u, target, variant)
        torch.testing.assert_close(before, proper_score_loss(u + 73, target, variant), rtol=0, atol=2e-14)
        torch.testing.assert_close(before, proper_score_loss(u[:, perm], inverse[target], variant), rtol=0, atol=2e-15)
        extremes = torch.tensor([[1e4, -1e4, 0.], [-1e4, 0., 1e4]], dtype=torch.float64, requires_grad=True)
        values = proper_score_loss(extremes, torch.tensor([0, 0]), variant)
        gradient = torch.autograd.grad(values.sum(), extremes)[0]
        assert torch.isfinite(values).all() and torch.isfinite(gradient).all() and (values >= 0).all()
        assert values[0] == 0


def test_small_training_exact_log_control_and_all_loss_checkpoint_semantics(tmp_path):
    rng = np.random.default_rng(31)
    tx, ty, vx, vy = {}, {}, {}, {}
    for a, symbol in enumerate(SYMBOLS):
        ty[symbol] = np.resize([-1, 0, 0, 0, 1] if a == 0 else [-1, -1, 0, 0, 1], 90)
        tx[symbol] = pd.DataFrame(rng.normal(size=(90, 4)) + ty[symbol][:, None] * .2, columns=list("abcd"))
        vy[symbol] = np.resize([-1, 0, 1], 33)
        vx[symbol] = pd.DataFrame(rng.normal(size=(33, 4)), columns=list("abcd"))
    original, expected_history = fit_confirm_neural(tx, ty, vx, vy)
    initial = copy.deepcopy(original.network.state_dict())
    matrix = np.concatenate([original.matrix(tx[s], a) for a, s in enumerate(SYMBOLS)])
    labels = np.concatenate([ty[s] for s in SYMBOLS])
    assets = np.repeat([0, 1], 90)
    validation = {a: original.matrix(vx[s], a) for a, s in enumerate(SYMBOLS)}
    validation_y = {a: vy[s] for a, s in enumerate(SYMBOLS)}
    for variant in VARIANTS:
        model, history = fit_proper_score(original, matrix, labels, assets, validation, validation_y, variant=variant)
        if variant == "log":
            assert history["history"] == expected_history["history"]
            assert history["best_epoch"] == expected_history["best_epoch"]
            for name, value in initial.items():
                assert torch.equal(model.network.state_dict()[name], value)
        model.save(tmp_path / variant)
        restored = PooledForecaster.load(tmp_path / variant)
        for a, symbol in enumerate(SYMBOLS):
            p = predict_matrix(model, validation[a], a)
            np.testing.assert_array_equal(p, model.predict_proba(vx[symbol], a))
            np.testing.assert_array_equal(p, predict_matrix(restored, validation[a], a))
            if variant == "log":
                np.testing.assert_array_equal(p, original.predict_proba(vx[symbol], a))
            changed = validation[a].copy()
            changed[11:, :-1] = 1e4
            np.testing.assert_array_equal(p[:11], predict_matrix(model, changed, a)[:11])
            np.testing.assert_allclose(p[:11], predict_matrix(model, validation[a][:11], a), rtol=0, atol=2e-7)
            np.testing.assert_allclose(p[:7], predict_matrix(model, validation[a][:7], a, batch_size=1), rtol=0, atol=2e-7)
    for name, value in initial.items():
        assert torch.equal(original.network.state_dict()[name], value)


def test_loss_rejects_invalid_probability_contract():
    for variant in ("unregistered", "log", "brier", "spherical2", "spherical4"):
        with pytest.raises(ValueError):
            proper_score_loss(torch.zeros((3, 3)), torch.tensor([0., 1., 2.]), variant)
    with pytest.raises(ValueError):
        proper_score_loss(torch.tensor([[float("nan"), 0., 1.]]), torch.tensor([0]), "brier")
    with pytest.raises(ValueError):
        proper_score_loss(torch.zeros((3, 2)), torch.tensor([0, 1, 2]), "spherical2")
