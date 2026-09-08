import copy
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from lob_forge.boundary_learned_memory import SequenceInputs, TrainingGrid, matrix_logits  # noqa: E402
from lob_forge.boundary_pooled import PooledForecaster, build_member_network  # noqa: E402
from lob_forge.boundary_temporal_attention import (  # noqa: E402
    ARCHITECTURES, CONFIG, TemporalAttentionForecaster, accumulate_attention_block, build_attention_network,
    contiguous_segments, fit_temporal_attention, predict_deltas, sequence_probabilities,
)


def active_network(kind, *, dimensions=3, width=8, heads=2, window=8):
    torch.set_num_threads(2)
    model = build_attention_network(dimensions, kind, width=width, heads=heads, history_observations=window)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(31)
        model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=generator) * .2)
    return model


def reference_window(model, matrix, segments, position):
    """Independent per-query definition, without batched temporal gathers."""
    allowed = [j for j in range(max(0, position - model.history_observations + 1), position + 1)
        if segments[j] == segments[position] and segments[position] >= 0]
    assert allowed
    encoded = model.stem(matrix[allowed])
    values = model.value(encoded).reshape(len(allowed), model.heads, -1)
    current = encoded[-1]
    aggregate = []
    for head in range(model.heads):
        if model.architecture == "instant":
            aggregate.append(values[-1, head])
            continue
        if model.architecture == "uniform":
            weights = torch.ones(len(allowed), dtype=matrix.dtype) / len(allowed)
        else:
            ages = torch.tensor([position - j for j in allowed])
            scores = model.age_bias[head, ages]
            if model.architecture == "content":
                query = model.query(current).reshape(model.heads, -1)[head]
                keys = model.key(encoded).reshape(len(allowed), model.heads, -1)[:, head]
                scores = scores + torch.stack([(query * key).sum() for key in keys]) / len(query) ** .5
            weights = torch.softmax(scores, dim=0)
        aggregate.append(torch.stack([weight * value for weight, value in zip(weights, values[:, head])]).sum(dim=0))
    return model.head(model.body(torch.cat([current, torch.cat(aggregate)])))


def test_segment_ids_and_independent_window_weighted_gradients():
    rng = np.random.default_rng(11)
    available = np.ones((2, 16), dtype=bool)
    available[0, [0, 5]] = False
    available[1, [6, 7, 13]] = False
    segments = contiguous_segments(available)
    np.testing.assert_array_equal(segments[0], [-1, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(contiguous_segments(np.zeros((1, 4), bool)), [[-1] * 4])
    x = rng.normal(size=(2, 16, 3))
    positions = np.array([4, 6, 8, 10, 12, 14])
    keep = available[:, positions].copy()
    keep[0, 3] = False
    labels = rng.integers(-1, 2, size=keep.shape)
    base = rng.normal(size=(*keep.shape, 3))
    weights = rng.uniform(.2, 3.5, size=keep.shape)
    for kind in ARCHITECTURES:
        actual = active_network(kind, window=5).double()
        reference = copy.deepcopy(actual)
        record = accumulate_attention_block(actual, x, segments, positions, base, labels, weights, keep)
        losses = []
        for stream in range(2):
            for q, position in enumerate(positions):
                if keep[stream, q]:
                    delta = reference_window(reference, torch.from_numpy(x[stream]), segments[stream], int(position))
                    logits = torch.from_numpy(base[stream, q]) + delta
                    target = torch.tensor([labels[stream, q] + 1], dtype=torch.long)
                    losses.append(torch.nn.functional.cross_entropy(logits[None], target) * weights[stream, q])
        loss = torch.stack(losses).sum() / int(keep.sum())
        loss.backward()
        assert record["supervised_rows"] == int(keep.sum())
        assert record["weighted_mean_loss"] == pytest.approx(float(loss.detach()), abs=2e-13)
        for (name, left), (_, right) in zip(actual.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=2e-12, msg=name)


def test_prefix_partition_gap_and_cross_stream_isolation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(47, 3)).astype(np.float32)
    times = np.arange(len(x), dtype=np.int64) * 1000
    for kind in ARCHITECTURES:
        model = active_network(kind)
        whole, _ = predict_deltas(model, x, times)
        for chunk in (1, 7, 13):
            chunked, _ = predict_deltas(model, x, times, chunk_rows=chunk)
            np.testing.assert_allclose(chunked, whole, rtol=0, atol=3e-7)
        later = x.copy()
        later[20:] = 1e4
        perturbed, _ = predict_deltas(model, later, times)
        short, _ = predict_deltas(model, x[:20], times[:20])
        np.testing.assert_allclose(perturbed[:20], whole[:20], rtol=0, atol=3e-7)
        np.testing.assert_allclose(short, whole[:20], rtol=0, atol=3e-7)
        keep = np.arange(len(x)) % 3 == 0
        sparse, _ = predict_deltas(model, x, times, selected=keep, chunk_rows=11)
        np.testing.assert_allclose(sparse, whole[keep], rtol=0, atol=3e-7)
        gaps = times.copy()
        gaps[20:] += 2000
        gapped, record = predict_deltas(model, x, gaps)
        fresh, _ = predict_deltas(model, x[20:], gaps[20:])
        assert record["coverage_segments"] == 2
        np.testing.assert_allclose(gapped[20:], fresh, rtol=0, atol=3e-7)
        batched = torch.from_numpy(np.stack([x, x]))
        segments = torch.zeros((2, len(x)), dtype=torch.long)
        positions = torch.arange(len(x))
        with torch.no_grad():
            before = model(batched, segments, positions)
            batched[1] = 1e4
            after = model(batched, segments, positions)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=3e-7)


def test_uniform_control_relative_age_orientation_and_missing_queries():
    x = torch.from_numpy(np.random.default_rng(3).normal(size=(1, 7, 3)).astype(np.float32))
    ids = torch.zeros((1, 7), dtype=torch.long)
    positions = torch.tensor([1, 4, 6])
    uniform = active_network("uniform", window=4)
    age = active_network("age", window=4)
    torch.testing.assert_close(uniform(x, ids, positions), age(x, ids, positions), rtol=0, atol=0)
    with torch.no_grad():
        age.age_bias[:, 0] = 20
    _, weights = age(x, ids, positions, return_attention=True)
    assert torch.all(weights[..., -1] > .999999)
    assert torch.equal(weights[:, 0, :, :2], torch.zeros_like(weights[:, 0, :, :2]))
    for kind in ARCHITECTURES:
        model = active_network(kind, window=4)
        missing = torch.full_like(ids, -1)
        output = model(x, missing, positions)
        assert torch.isfinite(output).all() and torch.equal(output, torch.zeros_like(output))
        output.sum().backward()
        assert all(p.grad is None or (torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) == 0) for p in model.parameters())


def test_excluded_values_cannot_poison_observed_gradients():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(2, 7, 3)).astype(np.float32)
    available = np.array([[True, True, False, True, True, False, True], [False] * 7])
    segments = contiguous_segments(available)
    changed = x.copy()
    changed[~available] = 1e25
    labels = rng.integers(-1, 2, size=(2, 7))
    for kind in ARCHITECTURES:
        original = active_network(kind, window=4)
        poisoned = copy.deepcopy(original)
        records = []
        for model, matrix in ((original, x), (poisoned, changed)):
            records.append(accumulate_attention_block(model, matrix, segments, np.arange(7), np.zeros((2, 7, 3), np.float32),
                labels, np.ones((2, 7), np.float32), available))
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert records[0] == records[1]
        for left, right in zip(original.parameters(), poisoned.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def original_model(frame, *, priors=None):
    normalizer = QuantileTransformer(n_quantiles=min(32, len(frame)), output_distribution="normal", random_state=4).fit(frame.to_numpy())
    model = PooledForecaster(build_member_network(frame.shape[1] + 1, members=1, hidden_size=64), normalizer,
        np.ones(frame.shape[1], dtype=bool), list(frame.columns), priors or {0: np.array([.2, .5, .3]), 1: np.array([.3, .4, .3])}, 1, 64)
    model.network.eval()
    return model


def test_exact_zero_correction_frozen_base_and_saved_model(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(9).normal(size=(40, 2)), columns=["a", "b"])
    times = np.arange(len(frame), dtype=np.int64) * 1000
    selected = np.arange(len(frame)) % 2 == 0
    for kind in ARCHITECTURES:
        base = original_model(frame)
        parameters = copy.deepcopy(base.network.state_dict())
        adapter = build_attention_network(3, kind, width=8, heads=2, history_observations=8)
        model = TemporalAttentionForecaster(base, adapter, kind)
        assert all(not p.requires_grad for p in base.network.parameters())
        for asset in (0, 1):
            expected = base.predict_proba(frame.loc[selected], asset)
            actual, _ = model.predict_proba(frame, times, asset, selected)
            np.testing.assert_array_equal(actual, expected)
        with torch.no_grad():
            adapter.head.weight.fill_(.1)
            adapter.head.bias.copy_(torch.tensor([-.3, .1, .2]))
        model.save(tmp_path / kind)
        restored = TemporalAttentionForecaster.load(tmp_path / kind)
        actual, _ = model.predict_proba(frame, times, 1, selected)
        replay, _ = restored.predict_proba(frame, times, 1, selected)
        np.testing.assert_array_equal(replay, actual)
        for name, value in parameters.items():
            assert torch.equal(value, base.network.state_dict()[name])


def test_learning_on_a_small_query_dependent_temporal_task():
    rng = np.random.default_rng(18)

    def sample(count):
        x = np.zeros((count, 5, 7), dtype=np.float32)
        y = np.empty((count, 1), dtype=np.int64)
        for row in range(count):
            keys = rng.permutation(4)
            values = rng.integers(0, 3, size=4)
            wanted = int(rng.integers(0, 4))
            x[row, np.arange(4), keys] = 1
            x[row, np.arange(4), 4 + values] = 1
            x[row, -1, wanted] = 1
            y[row, 0] = values[np.flatnonzero(keys == wanted)[0]] - 1
        return x, y

    train, labels = sample(384)
    test, targets = sample(256)
    model = build_attention_network(7, "content", width=32, heads=4, history_observations=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    for _ in range(350):
        rows = rng.choice(len(train), size=32, replace=False)
        optimizer.zero_grad()
        accumulate_attention_block(model, train[rows], np.zeros((32, 5), np.int64), np.array([4]),
            np.zeros((32, 1, 3), np.float32), labels[rows], np.ones((32, 1), np.float32), np.ones((32, 1), bool))
        optimizer.step()
    with torch.no_grad():
        forecast = model(torch.from_numpy(test), torch.zeros((len(test), 5), dtype=torch.long), torch.tensor([4]))
    accuracy = float((forecast.argmax(dim=-1).numpy() - 1 == targets).mean())
    assert accuracy > .9


def test_complete_fit_counts_targets_once_and_preserves_original_parameters(tmp_path):
    frame = pd.DataFrame({"a": (np.arange(48) // 4) % 3 - 1., "b": np.linspace(-1., 1., 48)})
    prior = np.full(3, 1 / 3)
    base = original_model(frame, priors={0: prior.copy(), 1: prior.copy()})
    before = copy.deepcopy(base.network.state_dict())
    shape = (8, 86400)
    grid = TrainingGrid(np.zeros((*shape, 3), np.float32), np.zeros((*shape, 3), np.float32),
        np.full(shape, -2, np.int8), np.zeros(shape, np.float32), np.zeros(shape, bool), np.zeros(shape, bool), np.repeat([0, 1], 4))
    labels = ((np.arange(48) // 4) % 3 - 1).astype(np.int64)
    validation, validation_labels = {}, {}
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        matrix = base.matrix(frame, asset)
        logits = matrix_logits(base, matrix)
        for stream in range(4 * asset, 4 * asset + 4):
            grid.matrix[stream, :48] = matrix
            grid.base_logits[stream, :48] = logits
            grid.labels[stream, :48] = labels
            grid.weights[stream, :48:4] = 1
            grid.available[stream, :48] = True
            grid.supervised[stream, :48:4] = True
        selected = np.ones(48, bool)
        validation[symbol] = SequenceInputs(matrix, np.arange(48, dtype=np.int64) * 1000, selected, logits, base.predict_proba(frame, asset))
        validation_labels[symbol] = labels.copy()
    with patch.dict(CONFIG, {"epochs": 2}):
        model, record = fit_temporal_attention(base, grid, validation, validation_labels, architecture="content", device="cpu")
    assert record["optimizer_steps"] == 2 and record["original_parameters_frozen"]
    assert all(r["supervised_rows"] == 96 and r["optimizer_steps"] == 1 for r in record["history"])
    for name, value in before.items():
        assert torch.equal(value, base.network.state_dict()[name])
    model.save(tmp_path / "fitted")
    restored = TemporalAttentionForecaster.load(tmp_path / "fitted")
    p, _ = sequence_probabilities(model.adapter, validation["BTCUSDT"], prior)
    q, _ = sequence_probabilities(restored.adapter, validation["BTCUSDT"], prior)
    np.testing.assert_array_equal(p, q)


def test_invalid_windows_and_supervision_are_rejected():
    with pytest.raises(ValueError):
        build_attention_network(3, "unknown")
    with pytest.raises(ValueError):
        build_attention_network(3, "content", width=7, heads=2)
    with pytest.raises(ValueError):
        contiguous_segments(np.array([[1, 0]]))
    model = active_network("content")
    x, ids = torch.zeros((1, 3, 3)), torch.zeros((1, 3), dtype=torch.long)
    for positions in (torch.tensor([2, 1]), torch.tensor([3]), torch.tensor([1., 2.])):
        with pytest.raises(ValueError):
            model(x, ids, positions)
    with pytest.raises(ValueError, match="new increasing id"):
        model(x, torch.tensor([[0, -1, 0]]), torch.tensor([2]))
    with pytest.raises(ValueError, match="Nonfinite"):
        model(torch.full_like(x, 1e30), ids, torch.tensor([2]))
    frame = pd.DataFrame([[1., 2.], [3., 4.]], columns=["a", "b"])
    with pytest.raises(ValueError, match="float32"):
        TemporalAttentionForecaster(original_model(frame), active_network("content").double(), "content")
    with pytest.raises(ValueError):
        accumulate_attention_block(model, x.numpy(), np.full((1, 3), -1), np.array([2]),
            np.zeros((1, 1, 3)), np.zeros((1, 1), dtype=int), np.ones((1, 1)), np.ones((1, 1), bool))
