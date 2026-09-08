from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.preprocessing import StandardScaler  # noqa: E402

from lob_forge.boundary_fading_memory import BLOCK_WIDTH, LEAKS, MECHANISMS, FadingMemoryMap, MemoryState, append_memory, observe_selected  # noqa: E402


def generator():
    normalizer = StandardScaler().fit(np.array([[-2., 0.], [0., 1.], [2., 2.]]))
    checkpoint = SimpleNamespace(members=1, hidden_size=64, active=np.array([True, True]),
                                 normalizer=normalizer, columns=["observed_a", "observed_b"])
    return FadingMemoryMap.from_checkpoint(checkpoint)


def observations(n=40):
    return pd.DataFrame(np.random.default_rng(2).normal(size=(n, 2)), columns=["observed_a", "observed_b"])


def test_memory_generator_is_deterministic_bounded_and_has_registered_operator_norm():
    first, second = generator(), generator()
    np.testing.assert_array_equal(first.projection, second.projection)
    np.testing.assert_array_equal(first.recurrent, second.recurrent)
    np.testing.assert_allclose(first.recurrent.T @ first.recurrent, .81 * np.eye(BLOCK_WIDTH), rtol=0, atol=1e-14)
    x = observations()
    times = np.arange(len(x), dtype=np.int64) * 1000
    values, state, record = first.observe(x, times, 0)
    assert record["state_resets"] == 1 and state.previous_time == times[-1]
    for mechanism in MECHANISMS:
        assert values[mechanism].shape == (len(x), 256)
        assert np.isfinite(values[mechanism]).all() and (np.abs(values[mechanism]) <= 1).all()


def test_chunking_future_perturbations_and_gap_resets_preserve_causality():
    model, x = generator(), observations()
    times = np.arange(len(x), dtype=np.int64) * 1000
    whole, _, _ = model.observe(x, times, 0)
    first, carry, _ = model.observe(x.iloc[:17], times[:17], 0)
    second, _, _ = model.observe(x.iloc[17:], times[17:], 0, state=carry)
    changed = x.copy()
    changed.iloc[17:] = 1e8
    future, _, _ = model.observe(changed, times, 0)
    for mechanism in MECHANISMS:
        np.testing.assert_allclose(np.concatenate([first[mechanism], second[mechanism]]), whole[mechanism], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(future[mechanism][:17], whole[mechanism][:17])
    gap_times = times.copy()
    gap_times[17:] += 1000
    gapped, _, record = model.observe(x, gap_times, 0)
    fresh, _, _ = model.observe(x.iloc[17:], gap_times[17:], 0)
    assert record["state_resets"] == 2
    for mechanism in MECHANISMS:
        np.testing.assert_allclose(gapped[mechanism][17:], fresh[mechanism], rtol=0, atol=1e-7)


def test_reservoir_contraction_bound_and_zero_recurrence_matches_ema():
    model, x = generator(), observations(1)
    state_a = MemoryState(np.zeros((2, 128)), np.full((2, 128), -.3), 0)
    state_b = MemoryState(np.zeros((2, 128)), np.full((2, 128), .4), 0)
    _, after_a, _ = model.observe(x, np.array([1000]), 1, state=state_a)
    _, after_b, _ = model.observe(x, np.array([1000]), 1, state=state_b)
    difference = np.linalg.norm(after_a.reservoir - after_b.reservoir, axis=1)
    bound = (1 - LEAKS[:, 0] + .9 * LEAKS[:, 0]) * np.linalg.norm(state_a.reservoir - state_b.reservoir, axis=1)
    assert (difference <= bound + 1e-12).all()
    model.recurrent[:] = 0
    longer = observations()
    values, _, _ = model.observe(longer, np.arange(len(longer), dtype=np.int64) * 1000, 1)
    np.testing.assert_array_equal(values["reservoir"], values["ema"])


def test_frozen_normalizer_schema_persistence_and_original_feature_preservation(tmp_path):
    model, x = generator(), observations()
    before_mean = model.normalizer.mean_.copy()
    times = np.arange(len(x), dtype=np.int64) * 1000
    values, _, _ = model.observe(x, times, 0)
    np.testing.assert_array_equal(before_mean, model.normalizer.mean_)
    path = tmp_path / "generator.joblib"
    model.save(path)
    restored = FadingMemoryMap.load(path)
    reloaded, _, _ = restored.observe(x, times, 0)
    for mechanism in MECHANISMS:
        np.testing.assert_array_equal(values[mechanism], reloaded[mechanism])
    appended = append_memory(x, values["reservoir"])
    pd.testing.assert_frame_equal(appended[x.columns], x)
    with pytest.raises(ValueError, match="schema"):
        model.observe(x.rename(columns={"observed_a": "future_target"}), times, 0)
    with pytest.raises(ValueError, match="advance"):
        model.observe(x.iloc[:1], np.array([1000]), 0, state=MemoryState(np.zeros((2, 128)), np.zeros((2, 128)), 1000))


def test_selected_rows_retain_intervening_observations_and_chunk_state():
    model, x = generator(), observations()
    times = np.arange(len(x), dtype=np.int64) * 1000
    keep = np.arange(len(x)) % 4 == 0
    whole, _, _ = model.observe(x, times, 0)
    selected, record = observe_selected(model, x, times, 0, keep, chunk_rows=7)
    assert record["observed_rows"] == 40 and record["retained_rows"] == 10 and record["state_resets"] == 1
    for mechanism in MECHANISMS:
        np.testing.assert_allclose(selected[mechanism], whole[mechanism][keep], rtol=0, atol=1e-7)
    skipped, _, _ = model.observe(x.loc[keep], times[keep], 0)
    assert not np.allclose(selected["reservoir"], skipped["reservoir"])
    with pytest.raises(ValueError, match="Boolean"):
        observe_selected(model, x, times, 0, keep.astype(int))
