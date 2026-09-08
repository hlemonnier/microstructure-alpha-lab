import copy

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from lob_forge.boundary_learned_memory import (  # noqa: E402
    ARCHITECTURES, LearnedMemoryForecaster, SequenceInputs, TrainingGrid, accumulate_temporal_block,
    build_residual_network, corrected_posterior, fit_learned_memory, map_state, masked_segments, predict_deltas,
)
from lob_forge.boundary_pooled import PooledForecaster, build_member_network  # noqa: E402


def active_network(kind, dimensions=3, width=5):
    torch.set_num_threads(2)
    model = build_residual_network(dimensions, kind, width=width)
    with torch.no_grad():
        model.head.weight.normal_(std=.2)
    return model


def test_natural_posterior_correction_and_row_local_exact_zero_path():
    z = np.array([[.2, -.4, 1.], [-1., .5, .3]])
    prior = np.array([.2, .5, .3])
    original = np.exp(z) * prior
    original /= original.sum(axis=1, keepdims=True)
    delta = np.array([[0., 0., 0.], [.4, -.7, .1]])
    actual = corrected_posterior(z, delta, prior, original)
    expected = np.exp(z + delta) * prior
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(actual[0], original[0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    changed = delta.copy()
    changed[1] = 100
    np.testing.assert_array_equal(corrected_posterior(z, changed, prior, original)[0], actual[0])
    with pytest.raises(ValueError, match="positive"):
        corrected_posterior(z, delta, [0., .5, .5], original)


def test_forward_chunking_future_prefix_and_gap_resets_for_every_architecture():
    x = np.random.default_rng(4).normal(size=(48, 3)).astype(np.float32)
    times = np.arange(len(x), dtype=np.int64) * 1000
    for kind in ARCHITECTURES:
        model = active_network(kind)
        whole, _ = predict_deltas(model, x, times)
        chunked, _ = predict_deltas(model, x, times, chunk_rows=7)
        np.testing.assert_allclose(chunked, whole, rtol=0, atol=2e-7)
        changed = x.copy()
        changed[20:] = 1e6
        later, _ = predict_deltas(model, changed, times)
        shorter, _ = predict_deltas(model, x[:20], times[:20])
        np.testing.assert_allclose(later[:20], whole[:20], rtol=0, atol=2e-7)
        np.testing.assert_allclose(shorter, whole[:20], rtol=0, atol=2e-7)
        gaps = times.copy()
        gaps[20:] += 1000
        gapped, record = predict_deltas(model, x, gaps)
        fresh, _ = predict_deltas(model, x[20:], gaps[20:])
        assert record["state_resets"] == 2
        np.testing.assert_allclose(gapped[20:], fresh, rtol=0, atol=2e-7)


def test_full_block_gradient_matches_independent_streams_and_missing_rows_are_ignored():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 12, 3)).astype(np.float32)
    base = rng.normal(size=(2, 12, 3)).astype(np.float32)
    labels = rng.integers(-1, 2, size=(2, 12))
    weights = rng.uniform(.2, 2., size=(2, 12)).astype(np.float32)
    valid = np.ones((2, 12), dtype=bool)
    valid[0, 3] = False
    valid[1, 8] = False
    keep = valid & (np.arange(12)[None] % 2 == 0)
    labels[~valid] = -2
    assert [(a, b) for a, b, _ in masked_segments(valid, truncation_seconds=4)] == [(0, 3), (3, 4), (4, 8), (8, 9), (9, 12)]
    for kind in ARCHITECTURES:
        batched = active_network(kind)
        independent = copy.deepcopy(batched)
        poisoned = copy.deepcopy(batched)
        carry, record = accumulate_temporal_block(batched, x, base, labels, weights, valid, keep, None, truncation_seconds=4)
        states = [independent.initial_state(1) for _ in range(2)]
        losses = []
        # Explicit fixed boundaries and independent per-stream forwards provide
        # a separate reference from the batched gather/scatter implementation.
        for left, right in [(0, 3), (3, 4), (4, 8), (8, 9), (9, 12)]:
            for stream in range(2):
                if not valid[stream, left]:
                    states[stream] = independent.initial_state(1)
                    continue
                delta, next_state = independent(torch.from_numpy(x[stream:stream + 1, left:right]), states[stream])
                states[stream] = map_state(next_state, lambda v: v.detach())
                take = keep[stream, left:right]
                if take.any():
                    logits = torch.from_numpy(base[stream, left:right][take]) + delta[0][take]
                    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels[stream, left:right][take] + 1), reduction="none")
                    losses.append((loss * torch.from_numpy(weights[stream, left:right][take])).sum())
        objective = sum(losses) / keep.sum()
        objective.backward()
        assert record["supervised_rows"] == int(keep.sum())
        assert record["weighted_mean_loss"] == pytest.approx(float(objective.detach()), abs=1e-6)
        for a, b in zip(batched.parameters(), independent.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
        if carry is not None:
            components = carry if isinstance(carry, tuple) else (carry,)
            assert all(not state.requires_grad and state.grad_fn is None for state in components)
        contaminated = x.copy()
        contaminated[~valid] = np.nan
        accumulate_temporal_block(poisoned, contaminated, base, labels, weights, valid, keep, None, truncation_seconds=4)
        for a, b in zip(batched.parameters(), poisoned.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_recurrence_learns_a_lagged_synthetic_signal_on_new_streams():
    torch.set_num_threads(2)
    rng = np.random.default_rng(14)
    x = rng.integers(-1, 2, size=(64, 24, 1)).astype(np.float32)
    labels = np.roll(x[:, :, 0], 1, axis=1).astype(int)
    valid = np.ones(labels.shape, dtype=bool)
    keep = valid.copy()
    keep[:, 0] = False
    model = build_residual_network(1, "gru_residual", width=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.01)
    for _ in range(50):
        optimizer.zero_grad()
        accumulate_temporal_block(model, x, np.zeros((*labels.shape, 3), dtype=np.float32), labels,
            np.ones(labels.shape, dtype=np.float32), valid, keep, None, truncation_seconds=32)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    heldout = rng.integers(-1, 2, size=(64, 24, 1)).astype(np.float32)
    target = np.roll(heldout[:, :, 0], 1, axis=1).astype(int)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(heldout))
        loss = torch.nn.functional.cross_entropy(logits[:, 1:].reshape(-1, 3), torch.tensor(target[:, 1:].reshape(-1) + 1))
    assert float(loss) < .8 * np.log(3)


def test_original_base_is_frozen_and_checkpoint_reproduces_sequence_forecasts(tmp_path):
    torch.manual_seed(9)
    raw = np.random.default_rng(9).normal(size=(64, 2))
    normalizer = QuantileTransformer(n_quantiles=16, output_distribution="normal", random_state=9).fit(raw)
    original = PooledForecaster(build_member_network(3, members=1, hidden_size=64), normalizer,
        np.array([True, True]), ["x", "y"], {0: np.array([.2, .5, .3]), 1: np.array([.3, .5, .2])}, 1, 64)
    frame = pd.DataFrame(raw, columns=["x", "y"])
    times = np.arange(len(frame), dtype=np.int64) * 1000
    keep = np.arange(len(frame)) >= 20
    for kind in ARCHITECTURES:
        model = LearnedMemoryForecaster(copy.deepcopy(original), build_residual_network(3, kind, width=5), kind)
        assert all(not parameter.requires_grad for parameter in model.original.network.parameters())
        zero, _ = model.predict_proba(frame, times, 0, keep)
        np.testing.assert_array_equal(zero, original.predict_proba(frame.loc[keep], 0))
        with torch.no_grad():
            model.adapter.head.weight.normal_(std=.2)
        before, _ = model.predict_proba(frame, times, 0, keep)
        model.save(tmp_path / kind)
        restored = LearnedMemoryForecaster.load(tmp_path / kind)
        after, _ = restored.predict_proba(frame, times, 0, keep)
        np.testing.assert_array_equal(after, before)


def test_full_fitter_counts_historical_rows_and_preserves_original_parameters(tmp_path):
    from lob_forge.boundary_confirm_model import SYMBOLS

    torch.set_num_threads(2)
    raw = np.random.default_rng(23).normal(size=(24, 2))
    normalizer = QuantileTransformer(n_quantiles=12, output_distribution="normal", random_state=23).fit(raw)
    original = PooledForecaster(build_member_network(3, members=1, hidden_size=64), normalizer,
        np.ones(2, dtype=bool), ["x", "y"], {0: np.full(3, 1 / 3), 1: np.full(3, 1 / 3)}, 1, 64)
    unchanged = copy.deepcopy(original.network.state_dict())
    shape = 8, 86400
    available = np.zeros(shape, dtype=bool)
    available[:, 120:144] = True
    selected = available & (np.arange(86400)[None] % 4 == 0)
    matrix = np.zeros((*shape, 3), dtype=np.float32)
    matrix[:, 120:144] = np.random.default_rng(24).normal(size=(8, 24, 3))
    labels = np.full(shape, -2, dtype=np.int8)
    labels[:, 120:144] = np.tile(np.repeat([-1, 0, 1], 4), 2)
    grid = TrainingGrid(matrix, np.zeros((*shape, 3), dtype=np.float32), labels,
        selected.astype(np.float32), available, selected, np.repeat([0, 1], 4))
    validation = SequenceInputs(matrix[0, 120:144].copy(), np.arange(24, dtype=np.int64) * 1000,
        np.ones(24, dtype=bool), np.zeros((24, 3), dtype=np.float32), np.full((24, 3), 1 / 3))
    validation.save(tmp_path / "validation.npz")
    restored = SequenceInputs.load(tmp_path / "validation.npz")
    np.testing.assert_array_equal(validation.matrix, restored.matrix)
    model, history = fit_learned_memory(original, grid, dict.fromkeys(SYMBOLS, restored),
        {s: labels[0, 120:144].copy() for s in SYMBOLS}, architecture="gru_residual", device="cpu")
    assert len(history["history"]) == 6
    assert history["optimizer_steps"] == 6
    assert all(r["supervised_rows"] == 48 for r in history["history"])
    assert 1 <= history["best_epoch"] <= 6
    assert torch.count_nonzero(model.adapter.head.weight).item() > 0
    for name, value in original.network.state_dict().items():
        torch.testing.assert_close(value, unchanged[name], rtol=0, atol=0)
    assert all(not p.requires_grad for p in original.network.parameters())
    wrong = copy.deepcopy(grid)
    wrong.supervised[0, 121] = True
    with pytest.raises(AssertionError):
        wrong.validate(3, original.priors)
