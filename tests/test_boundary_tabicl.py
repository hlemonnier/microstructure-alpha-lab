import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_tabicl import SYMBOLS, TabiclForecaster, case_control_posterior, select_balanced_context  # noqa: E402


def test_context_sampling_is_unique_balanced_deterministic_and_pool_only():
    labels = {SYMBOLS[0]: np.repeat([-1, 0, 1], [10, 70, 20]), SYMBOLS[1]: np.repeat([-1, 0, 1], [30, 40, 10])}
    selected, population, sampled = select_balanced_context(labels, requested_rows=48)
    again, _, _ = select_balanced_context(dict(reversed(list(labels.items()))), requested_rows=48)
    for symbol in SYMBOLS:
        np.testing.assert_array_equal(selected[symbol], again[symbol])
        assert len(selected[symbol]) == 24 and len(np.unique(selected[symbol])) == 24
        assert (np.diff(selected[symbol]) > 0).all()
        np.testing.assert_array_equal([(labels[symbol][selected[symbol]] == c).sum() for c in (-1, 0, 1)], [8] * 3)
        np.testing.assert_array_equal(sampled[symbol], np.full(3, 1 / 3))
        np.testing.assert_array_equal(population[symbol], [(labels[symbol] == c).mean() for c in (-1, 0, 1)])
    small, _, _ = select_balanced_context(labels, requested_rows=600)
    assert all(len(v) == 30 for v in small.values())  # Smallest cell is ten; no replacement.


def test_context_sampling_rejects_missing_classes_and_invalid_budgets():
    labels = {s: np.array([-1, 0, 1]) for s in SYMBOLS}
    for budget in (0, 7, True, 6.0):
        with pytest.raises(ValueError, match="budget"):
            select_balanced_context(labels, requested_rows=budget)
    labels[SYMBOLS[0]] = np.array([0, 1])
    with pytest.raises(ValueError, match="all three classes"):
        select_balanced_context(labels)
    labels[SYMBOLS[0]] = np.array([-2, 0, 1])
    with pytest.raises(ValueError, match="three-class"):
        select_balanced_context(labels)


def test_case_control_correction_inverts_analytic_selection_for_each_asset():
    posterior = np.array([[.02, .8, .18], [.5, .3, .2], [.1, .6, .3]])
    for population in (np.array([.1, .8, .1]), np.array([.4, .3, .3])):
        for sampled in (np.full(3, 1 / 3), np.array([.2, .3, .5])):
            selected_posterior = posterior * sampled / population
            selected_posterior /= selected_posterior.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(case_control_posterior(selected_posterior, sampled, population), posterior)
            # In an uninformative context the prior must be restored exactly.
            np.testing.assert_allclose(case_control_posterior(sampled[None], sampled, population)[0], population)


def test_case_control_correction_rejects_invalid_probability_contracts():
    prior = np.full(3, 1 / 3)
    for bad in (np.array([[1, 1, 1]]), np.array([[0, np.nan, 1]]), np.array([[1.1, -.1, 0]])):
        with pytest.raises(ValueError, match="probabilities"):
            case_control_posterior(bad, prior, prior)
    for bad in (np.array([0, .5, .5]), np.array([1, 1, 1]), np.array([.5, .5])):
        with pytest.raises(ValueError, match="priors"):
            case_control_posterior(prior[None], prior, bad)


class DummyEstimator:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, matrix):
        logits = np.column_stack([matrix[:, 0], matrix[:, -1], -matrix[:, 0]])
        q = np.exp(logits - logits.max(axis=1, keepdims=True))
        return q / q.sum(axis=1, keepdims=True)


def test_wrapper_preserves_rows_asset_identity_batches_and_natural_posterior():
    population = {SYMBOLS[0]: np.array([.1, .8, .1]), SYMBOLS[1]: np.array([.2, .6, .2])}
    sampled = {s: np.full(3, 1 / 3) for s in SYMBOLS}
    model = TabiclForecaster(DummyEstimator(), ["observed"], population, sampled, "unused", "unused")
    frame = pd.DataFrame({"observed": np.arange(11) / 10})
    for asset, symbol in enumerate(SYMBOLS):
        matrix = model.matrix(frame, symbol)
        np.testing.assert_array_equal(matrix[:, -1], asset)
        expected = case_control_posterior(model.estimator.predict_proba(matrix), sampled[symbol], population[symbol])
        np.testing.assert_array_equal(model.predict_proba(frame, symbol, batch_size=3), expected)
        np.testing.assert_array_equal(model.predict_proba(frame.iloc[:2], symbol), expected[:2])
        assert model.predict_proba(frame.iloc[:0], symbol).shape == (0, 3)


def test_wrapper_rejects_changed_schema_class_order_and_query_batch():
    priors = {s: np.full(3, 1 / 3) for s in SYMBOLS}
    estimator = DummyEstimator()
    model = TabiclForecaster(estimator, ["observed"], priors, priors, "unused", "unused")
    frame = pd.DataFrame({"observed": [1.]})
    with pytest.raises(ValueError, match="schema"):
        model.predict_proba(frame.rename(columns={"observed": "future"}), SYMBOLS[0])
    for batch in (0, True, 1.5):
        with pytest.raises(ValueError, match="batch"):
            model.predict_proba(frame, SYMBOLS[0], batch_size=batch)
    with pytest.raises(ValueError, match="Finite"):
        model.predict_proba(pd.DataFrame({"observed": [np.nan]}), SYMBOLS[0])
    estimator.classes_ = np.array([1, 0, -1])
    with pytest.raises(ValueError, match="ordering"):
        model.predict_proba(frame, SYMBOLS[0])
