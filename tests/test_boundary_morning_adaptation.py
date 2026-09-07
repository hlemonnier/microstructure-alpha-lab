import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_morning_adaptation import morning_masks, prior_rebasing_offsets  # noqa: E402


def softmax(values):
    p = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def test_rebased_weighted_logits_preserve_natural_posterior_for_each_asset():
    logits = np.array([[0.7, -0.4, 1.2], [-2.0, 1.5, 0.4]])
    old = np.array([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
    recent = np.array([[0.05, 0.88, 0.07], [0.4, 0.2, 0.4]])
    offsets = prior_rebasing_offsets(old, recent)
    expected = softmax(logits + np.log(old))
    actual = softmax(logits + offsets) * recent
    actual /= actual.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    assert not np.allclose(softmax(logits) * recent / (softmax(logits) * recent).sum(axis=1, keepdims=True), expected)


def test_class_weighted_loss_has_zero_expected_gradient_at_correct_natural_posterior():
    old = np.array([[0.3, 0.4, 0.3]])
    recent = np.array([[0.07, 0.85, 0.08]])
    natural = np.array([[0.2, 0.7, 0.1]])
    raw_logits = np.log(natural) - np.log(old)
    weighted_q = softmax(raw_logits + prior_rebasing_offsets(old, recent))
    weighted_target_mass = natural / recent
    expected_gradient = weighted_target_mass.sum(axis=1, keepdims=True) * weighted_q - weighted_target_mass
    np.testing.assert_allclose(expected_gradient, 0, atol=1e-14)


def test_rebasing_rejects_missing_or_nonpositive_asset_priors():
    for old, new in [([[0.2, 0.6, 0.2]], [[0, 0.8, 0.2]]), ([[0.2, 0.6, 0.2]], [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]]), ([[0.2, 0.6, 0.2]], [[0.2, float("nan"), 0.2]])]:
        with pytest.raises(ValueError):
            prior_rebasing_offsets(old, new)


def test_morning_partitions_use_actual_label_release_and_have_a_purge():
    cutoff = (11 * 60 + 30) * 60000
    times = np.array([120000, cutoff - 6 * 3600000, cutoff - 10000, cutoff - 5000, cutoff + 10000, cutoff + 20000, (11 * 60 + 57) * 60000])
    release = times + 5100
    training, validation = morning_masks(times, release, 0, window="six_hours")
    np.testing.assert_array_equal(training, [False, True, True, False, False, False, False])
    np.testing.assert_array_equal(validation, [False, False, False, False, True, True, False])
    full, _ = morning_masks(times, release, 0, window="full_morning")
    assert full[0]
    assert not np.any(training & validation)
    assert np.max(release[training]) <= cutoff
    assert np.max(release[validation]) <= (11 * 60 + 57) * 60000 + 10000


def test_morning_finetuning_preserves_normalizer_and_checkpoint_probability_contract(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    torch = pytest.importorskip("torch")
    from sklearn.preprocessing import QuantileTransformer

    from lob_forge.binance_vision import sha256_file
    from lob_forge.boundary_morning_adaptation import fit_morning_neural
    from lob_forge.boundary_pooled import PooledForecaster, build_member_network

    torch.set_num_threads(2)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(48, 2)), columns=["observed_a", "observed_b"])
    normalizer = QuantileTransformer(n_quantiles=16, output_distribution="normal").fit(x.to_numpy())
    original = PooledForecaster(build_member_network(3, members=1, hidden_size=8), normalizer, np.ones(2, dtype=bool), list(x.columns), {0: np.ones(3) / 3, 1: np.ones(3) / 3}, 1, 8)
    checkpoint = tmp_path / "source"
    original.save(checkpoint)
    source_hash = sha256_file(checkpoint / "model.pt")
    training = {s: x for s in ("BTCUSDT", "ETHUSDT")}
    labels = {s: np.array([-1] * 6 + [0] * 36 + [1] * 6) for s in training}
    validation = {s: x.iloc[:12] for s in training}
    validation_y = {s: np.resize([-1, 0, 1], 12) for s in training}
    decision_priors = {s: np.array([0.125, 0.75, 0.125]) for s in training}
    model, metadata = fit_morning_neural(checkpoint, training, labels, validation, validation_y, decision_priors, learning_rate=0.0003)
    assert metadata["best_epoch"] in range(5)
    assert len(metadata["history"]) == 5
    assert metadata["history"][0]["mean_training_loss"] is None
    assert all(np.isfinite(r["mean_training_loss"]) for r in metadata["history"][1:])
    assert sha256_file(checkpoint / "model.pt") == source_hash
    np.testing.assert_array_equal(model.matrix(x, 0), original.matrix(x, 0))
    np.testing.assert_array_equal(model.priors[0], original.priors[0])
    np.testing.assert_allclose(metadata["recent_training_priors"], [[0.125, 0.75, 0.125]] * 2)
    model.save(tmp_path / "adapted")
    restored = PooledForecaster.load(tmp_path / "adapted")
    p = model.predict_proba(x, 0)
    np.testing.assert_array_equal(restored.predict_proba(x, 0), p)
    np.testing.assert_allclose(p.sum(axis=1), 1)
