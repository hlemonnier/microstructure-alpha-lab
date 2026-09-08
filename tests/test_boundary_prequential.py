import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from lob_forge.boundary_morning_adaptation import prior_rebasing_offsets  # noqa: E402
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network  # noqa: E402
from lob_forge.boundary_prequential import SYMBOLS, PrequentialState, integer_clock, released_window, run_prequential  # noqa: E402


def original_model():
    torch.manual_seed(7)
    raw = np.random.default_rng(7).normal(size=(100, 2))
    normalizer = QuantileTransformer(n_quantiles=20, output_distribution="normal").fit(raw)
    return PooledForecaster(build_member_network(3, hidden_size=5), normalizer, np.ones(2, dtype=bool), ["x", "y"],
        {0: np.array([.2, .6, .2]), 1: np.array([.3, .4, .3])}, 1, 5)


def synthetic_stream():
    rng = np.random.default_rng(4)
    x = {s: pd.DataFrame(rng.normal(size=(120, 2)), columns=["x", "y"]) for s in SYMBOLS}
    y = {s: np.tile(np.array([-1, 0, 1]), 40) for s in SYMBOLS}
    t = {s: np.arange(120, dtype=np.int64) * 1000 for s in SYMBOLS}
    release = {s: t[s] + 5100 for s in SYMBOLS}
    return x, y, t, release


def replay_bank(model, x, y):
    return (np.concatenate([model.matrix(x[s].iloc[:30], a) for a, s in enumerate(SYMBOLS)]),
            np.concatenate([y[s][:30] for s in SYMBOLS]), np.repeat([0, 1], 30))


def test_window_uses_actual_release_observation_delay_and_original_age():
    origins = np.arange(10, dtype=np.int64) * 1000
    release = origins + 5100
    np.testing.assert_array_equal(released_window(origins, release, 9200, window_seconds=9), [False, True, True, True, True, False, False, False, False, False])
    assert not released_window(origins, release, 9199, window_seconds=9)[4]
    # A very delayed label remains old even if it has just become available.
    late = release.copy()
    late[:] = 20100
    assert not released_window(origins, late, 20200, window_seconds=5).any()
    with pytest.raises(ValueError, match="strictly delayed"):
        released_window(origins, origins, 10000)
    with pytest.raises(ValueError, match="Nonoverflowing"):
        integer_clock(np.array([2**63], dtype=np.uint64), strict=True)
    with pytest.raises(ValueError, match="Nonoverflowing"):
        integer_clock(np.array([1.5]), strict=True)


def test_accumulated_recent_and_replay_gradient_matches_full_cohort_mean():
    model = original_model()
    first = PrequentialState(model, "full_replay")
    second = PrequentialState(model, "full_replay")
    x, y, _, _ = synthetic_stream()
    matrix, labels, assets = replay_bank(model, x, y)
    labels = labels.copy()
    labels[[0, 3, 6, 30]] = 0
    priors, weights = balanced_asset_weights(labels, assets)
    offsets = prior_rebasing_offsets(np.array([model.priors[a] for a in (0, 1)]), np.array([priors[a] for a in (0, 1)]))
    first.optimizer.zero_grad()
    first.accumulate_loss_gradient(matrix, labels, assets, weights, offsets, coefficient=.5, batch_size=13)
    first.accumulate_loss_gradient(matrix, np.tile([-1, 0, 1], 20), assets, np.ones(60), np.zeros((2, 3)), coefficient=.5, batch_size=13)
    second.optimizer.zero_grad()
    tx, ta = torch.from_numpy(matrix), torch.tensor(assets)
    logits = second.logits(tx, ta)
    recent = (torch.nn.functional.cross_entropy(logits + torch.tensor(offsets, dtype=torch.float32)[ta], torch.tensor(labels + 1), reduction="none") * torch.from_numpy(weights)).mean()
    historical = torch.nn.functional.cross_entropy(logits, torch.tensor(np.tile([0, 1, 2], 20)))
    (.5 * recent + .5 * historical).backward()
    for left, right in zip(first.parameters, second.parameters):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)


def test_delayed_publication_uses_previous_model_and_future_labels_cannot_change_prefix():
    model = original_model()
    x, y, t, release = synthetic_stream()
    replay = replay_bank(model, x, y)
    anchors = np.array([60000, 80000, 100000])
    p, state, history = run_prequential(model, "full_recent", x, y, t, release, x, t, replay, anchors, window_seconds=30)
    changed_labels = {s: y[s].copy() for s in SYMBOLS}
    for symbol in SYMBOLS:
        changed_labels[symbol][release[symbol] > 79900] *= -1
    other, _, _ = run_prequential(model, "full_recent", x, changed_labels, t, release, x, t, replay, anchors, window_seconds=30)
    for asset, symbol in enumerate(SYMBOLS):
        # First update starts at 60s but cannot affect forecasts before 70s.
        np.testing.assert_allclose(p[symbol][:70], model.predict_proba(x[symbol].iloc[:70], asset), rtol=0, atol=2e-6)
        # Modified labels first enter the 100s update, published at 110s.
        np.testing.assert_array_equal(p[symbol][:110], other[symbol][:110])
    assert state.updates == 3 and history["optimizer_steps"] == 3
    for row in history["updates"]:
        assert all(v + 100 <= row["anchor_ms"] for v in row["last_used_release_ms"].values())
        assert row["published_ms"] == row["anchor_ms"] + 10000


def test_future_features_and_stream_truncation_preserve_earlier_predictions():
    model = original_model()
    x, y, t, release = synthetic_stream()
    replay = replay_bank(model, x, y)
    anchors = np.array([60000, 80000, 100000])
    p, _, _ = run_prequential(model, "full_replay", x, y, t, release, x, t, replay, anchors, window_seconds=30)
    changed = {s: x[s].copy() for s in SYMBOLS}
    for symbol in SYMBOLS:
        changed[symbol].iloc[100:] = 10000
    other, _, _ = run_prequential(model, "full_replay", changed, y, t, release, changed, t, replay, anchors, window_seconds=30)
    short_x, short_y = {s: x[s].iloc[:99] for s in SYMBOLS}, {s: y[s][:99] for s in SYMBOLS}
    short_t, short_release = {s: t[s][:99] for s in SYMBOLS}, {s: release[s][:99] for s in SYMBOLS}
    truncated, _, _ = run_prequential(model, "full_replay", short_x, short_y, short_t, short_release, short_x, short_t, replay, anchors[:2], window_seconds=30)
    for symbol in SYMBOLS:
        np.testing.assert_array_equal(p[symbol][:100], other[symbol][:100])
        np.testing.assert_allclose(p[symbol][:99], truncated[symbol], rtol=0, atol=2e-6)


def test_bias_control_freezes_original_weights_and_frozen_path_is_exact():
    model = original_model()
    x, y, t, release = synthetic_stream()
    replay = replay_bank(model, x, y)
    anchors = np.array([60000, 80000, 100000])
    frozen, unchanged, _ = run_prequential(model, "frozen", x, y, t, release, x, t, replay, anchors)
    _, bias, _ = run_prequential(model, "bias_recent", x, y, t, release, x, t, replay, anchors, window_seconds=30)
    for asset, symbol in enumerate(SYMBOLS):
        np.testing.assert_array_equal(frozen[symbol], model.predict_proba(x[symbol], asset))
    assert unchanged.updates == 0 and bias.updates == 3 and torch.any(bias.bias != 0)
    for original, actual in zip(model.network.parameters(), bias.base.network.parameters()):
        torch.testing.assert_close(original, actual, rtol=0, atol=0)


def test_checkpoint_restores_predictions_optimizer_and_exact_next_update(tmp_path):
    model = original_model()
    x, y, _, _ = synthetic_stream()
    replay = replay_bank(model, x, y)
    for variant in ("bias_recent", "full_replay"):
        state = PrequentialState(model, variant)
        state.update(*replay, replay)
        folder = tmp_path / variant
        state.save(folder)
        restored = PrequentialState.load(folder)
        assert state.updates == restored.updates == 1
        np.testing.assert_array_equal(state.predict_matrix(replay[0][:30], 0), restored.predict_matrix(replay[0][:30], 0))
        state.update(*replay, replay)
        restored.update(*replay, replay)
        np.testing.assert_array_equal(state.predict_matrix(replay[0][:30], 0), restored.predict_matrix(replay[0][:30], 0))
        torch.testing.assert_close(state.bias, restored.bias, rtol=0, atol=0)


def test_missing_recent_class_skips_update_and_invalid_replay_is_rejected():
    model = original_model()
    x, y, _, _ = synthetic_stream()
    rx, ry, ra = replay_bank(model, x, y)
    state = PrequentialState(model, "full_replay")
    result = state.update(rx, np.zeros(len(ry), dtype=int), ra, (rx, ry, ra))
    assert not result["optimizer_step"] and state.updates == 0
    with pytest.raises(ValueError, match="equal nonempty"):
        state.update(rx, ry, ra, (rx[:-1], ry[:-1], ra[:-1]))
