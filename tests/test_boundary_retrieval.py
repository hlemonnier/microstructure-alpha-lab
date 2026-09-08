import copy
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from lob_forge.boundary_retrieval import (  # noqa: E402
    ARCHITECTURES, HistoricalBankSampler, RetrievalForecaster, build_retrieval_network,
    fit_retrieval, kernel_log_prob, retrieval_training_loss, streamed_log_prob,
)


def historical_fixture():
    labels, assets, dates = [], [], []
    for day in range(4):
        for asset in range(2):
            for label in (-1, 0, 1):
                count = 4 + day + asset + (label + 1) * (day + 1)
                labels.extend([label] * count)
                assets.extend([asset] * count)
                dates.extend([day] * count)
    return tuple(np.array(v) for v in (labels, assets, dates))


def test_stratified_weights_recover_class_mass_and_exclude_entire_query_day():
    labels, assets, dates = historical_fixture()
    bank = HistoricalBankSampler(labels, assets, dates, keys_per_cell=3)
    chosen = bank.sample(np.random.default_rng(21))
    assert len(chosen) == len(np.unique(chosen)) == 72
    qa, qd = np.tile([0, 1], 4), np.repeat(np.arange(4), 2)
    weights = np.exp(bank.log_weights(qa, qd, chosen))
    for row, (asset, day) in enumerate(zip(qa, qd)):
        assert np.all(weights[row, (dates[chosen] == day) | (assets[chosen] != asset)] == 0)
        for label in (-1, 0, 1):
            # Float32 log/exp and the nine-term sum each round; allow eight
            # machine epsilons while checking the analytical mixture below.
            assert weights[row, labels[chosen] == label].sum() == pytest.approx(1., abs=8 * np.finfo(np.float32).eps)
        for key, weight in zip(chosen, weights[row]):
            if weight:
                numerator = np.sum((dates == dates[key]) & (assets == asset) & (labels == labels[key]))
                denominator = 3 * np.sum((dates != day) & (assets == asset) & (labels == labels[key]))
                assert weight == pytest.approx(numerator / denominator, abs=2e-8)
    flat = kernel_log_prob(torch.zeros((8, 2)), torch.zeros((72, 2)), torch.tensor(labels[chosen] + 1),
        torch.from_numpy(bank.log_weights(qa, qd, chosen))).exp()
    torch.testing.assert_close(flat, torch.full((8, 3), 1 / 3), rtol=0, atol=1e-7)
    for asset in range(2):
        for label in (-1, 0, 1):
            assert bank.query_weights[(assets == asset) & (labels == label)].sum() == pytest.approx(len(labels) / 6, rel=2e-7)
    with pytest.raises(ValueError, match="support"):
        HistoricalBankSampler(labels, assets, dates, keys_per_cell=1000)
    with pytest.raises(ValueError, match="unique"):
        bank.log_weights(qa, qd, np.repeat(chosen[0], 72))


def test_kernel_value_and_gradient_match_explicit_pairwise_calculation():
    torch.set_num_threads(2)
    query = torch.tensor([[.2, -.3], [1., .4]], dtype=torch.float64, requires_grad=True)
    keys = torch.tensor([[-.6, .3], [.1, .2], [.8, -.1], [-.7, .9], [.5, .8], [1., -1.]], dtype=torch.float64, requires_grad=True)
    classes = torch.tensor([0, 1, 2, 0, 1, 2])
    w = torch.tensor([[.3, .7, .4, 0., .2, .9], [.6, .4, 0., .7, .2, .3]], dtype=torch.float64)
    target = torch.tensor([1, 2])
    actual = kernel_log_prob(query, keys, classes, w.log())
    pairwise = torch.sqrt(((query[:, None] - keys[None]) ** 2).sum(-1) + 1e-12)
    scores = torch.stack([(w * torch.exp(-pairwise))[:, classes == c].sum(1) for c in range(3)], 1)
    expected = (scores / scores.sum(1, keepdim=True)).log()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    first = torch.autograd.grad(-actual[torch.arange(2), target].mean(), (query, keys), retain_graph=True)
    second = torch.autograd.grad(-expected[torch.arange(2), target].mean(), (query, keys))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)
    assert torch.autograd.gradcheck(lambda q, k: kernel_log_prob(q, k, classes, w.log()), (query, keys))


def test_full_bank_streaming_and_class_duplication_invariance():
    rng = np.random.default_rng(7)
    query = torch.tensor(rng.normal(size=(11, 4)), dtype=torch.float64)
    keys = torch.tensor(rng.normal(size=(19, 4)), dtype=torch.float64)
    classes = torch.tensor([0] * 4 + [1] * 6 + [2] * 9)
    distances = np.linalg.norm(query.numpy()[:, None] - keys.numpy()[None], axis=-1)
    scores = np.column_stack([np.exp(-np.sqrt(distances[:, classes.numpy() == c] ** 2 + 1e-12)).mean(1) for c in range(3)])
    expected = scores / scores.sum(1, keepdims=True)
    actual = streamed_log_prob(query, keys, classes, key_chunk_rows=5).exp()
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-14, atol=1e-14)
    repeated = torch.repeat_interleave(torch.arange(len(keys)), torch.tensor([1, 3, 2])[classes])
    copied = streamed_log_prob(query, keys[repeated], classes[repeated], key_chunk_rows=7).exp()
    torch.testing.assert_close(actual, copied, rtol=1e-14, atol=1e-14)
    order = torch.tensor(rng.permutation(len(keys)))
    reordered = streamed_log_prob(query, keys[order], classes[order], key_chunk_rows=2).exp()
    torch.testing.assert_close(actual, reordered, rtol=1e-14, atol=1e-14)


def test_forecast_prefix_state_and_checkpoint_invariants_for_every_architecture(tmp_path):
    torch.set_num_threads(2)
    labels, assets, dates = historical_fixture()
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(len(labels), 4))
    normalizer = QuantileTransformer(n_quantiles=32, output_distribution="normal", random_state=3).fit(raw)
    matrix = np.column_stack([normalizer.transform(raw), 2 * assets - 1]).astype(np.float32)
    sampler = HistoricalBankSampler(labels, assets, dates, keys_per_cell=3)
    queries = pd.DataFrame(rng.normal(size=(35, 4)), columns=list("abcd"))
    for kind in ARCHITECTURES:
        model = build_retrieval_network(5, kind, width=8)
        if kind != "fixed_metric":
            loss = retrieval_training_loss(model, matrix, labels, sampler, np.arange(32), sampler.sample(rng))
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        learner = RetrievalForecaster(model, normalizer, np.ones(4, dtype=bool), list("abcd"), sampler.priors, {})
        learner.refresh_bank(matrix, labels, assets)
        before = copy.deepcopy(model.state_dict())
        original = learner.predict_proba(queries, 0, batch_size=16, key_chunk_rows=29)
        changed = queries.copy()
        changed.iloc[13:] = 1e100
        future = learner.predict_proba(changed, 0, batch_size=16, key_chunk_rows=29)
        short = learner.predict_proba(queries.iloc[:13], 0, batch_size=3, key_chunk_rows=17)
        np.testing.assert_allclose(original[:13], future[:13], rtol=0, atol=3e-7)
        np.testing.assert_allclose(original[:13], short, rtol=0, atol=3e-7)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
        learner.save(tmp_path / kind)
        restored = RetrievalForecaster.load(tmp_path / kind)
        np.testing.assert_array_equal(original, restored.predict_proba(queries, 0, batch_size=16, key_chunk_rows=29))
        with pytest.raises(ValueError, match="schema"):
            restored.predict_proba(queries[list("dcba")], 0)


def test_learned_metric_finds_signal_beyond_fixed_distance_on_a_new_date():
    torch.set_num_threads(2)
    rng = np.random.default_rng(106)
    labels = np.tile(np.repeat([-1, 0, 1], 32), 8)
    assets = np.repeat(np.tile([0, 1], 4), 96)
    dates = np.repeat(np.arange(4), 192)
    x = rng.normal(size=(len(labels), 9)).astype(np.float32)
    x[:, 0] = .25 * labels + rng.normal(0, .015, len(labels))
    heldout_y = np.tile([-1, 0, 1], 100)
    heldout = rng.normal(size=(len(heldout_y), 9)).astype(np.float32)
    heldout[:, 0] = .25 * heldout_y + rng.normal(0, .015, len(heldout_y))
    sampler = HistoricalBankSampler(labels, assets, dates, keys_per_cell=8)
    fixed = RetrievalForecaster(build_retrieval_network(9, "fixed_metric"), None, np.ones(8, dtype=bool), [], sampler.priors, {})
    fixed.refresh_bank(x, labels, assets)
    baseline = np.mean(np.argmax(fixed.predict_matrix(heldout, 0), 1) - 1 == heldout_y)
    model = build_retrieval_network(9, "linear_nca", width=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.01)
    for _ in range(90):
        optimizer.zero_grad()
        loss = retrieval_training_loss(model, x, labels, sampler, rng.choice(len(labels), 128, replace=False), sampler.sample(rng))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
    learned = RetrievalForecaster(model, None, np.ones(8, dtype=bool), [], sampler.priors, {})
    learned.refresh_bank(x, labels, assets)
    accuracy = np.mean(np.argmax(learned.predict_matrix(heldout, 0), 1) - 1 == heldout_y)
    assert accuracy > baseline + .3
    assert accuracy > .9


def test_fitting_uses_validation_checkpoint_and_full_original_bank(tmp_path):
    torch.set_num_threads(2)
    labels, assets, dates = historical_fixture()
    rng = np.random.default_rng(28)
    raw = rng.normal(size=(len(labels), 2))
    normalizer = QuantileTransformer(n_quantiles=16, output_distribution="normal", random_state=3).fit(raw)
    matrix = np.column_stack([normalizer.transform(raw), 2 * assets - 1]).astype(np.float32)
    sampler = HistoricalBankSampler(labels, assets, dates, keys_per_cell=2)
    original = SimpleNamespace(normalizer=normalizer, active=np.ones(2, dtype=bool), columns=["x", "y"], priors=sampler.priors)
    validation = {a: matrix[assets == a][:12] for a in (0, 1)}
    vy = {a: np.tile([-1, 0, 1], 4) for a in (0, 1)}
    learner, history = fit_retrieval(original, matrix, labels, assets, dates, validation, vy,
        architecture="linear_nca", epochs=2, batch_size=48, keys_per_cell=2)
    keys = [(np.mean([v["balanced_accuracy"] for v in r["validation"].values()]),
        -np.mean([v["log_loss"] for v in r["validation"].values()])) for r in history["history"]]
    assert history["best_epoch"] == keys.index(max(keys)) + 1
    assert history["optimizer_steps"] == 2 * int(np.ceil(len(labels) / 48))
    for a in (0, 1):
        np.testing.assert_array_equal(learner.banks[a]["classes"], labels[assets == a] + 1)
        np.testing.assert_array_equal(learner.priors[a], original.priors[a])
    before = {a: learner.predict_matrix(validation[a], a) for a in (0, 1)}
    learner.save(tmp_path / "fitted")
    restored = RetrievalForecaster.load(tmp_path / "fitted")
    for a in (0, 1):
        np.testing.assert_array_equal(before[a], restored.predict_matrix(validation[a], a))
