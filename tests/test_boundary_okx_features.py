import copy

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_okx_features import counted_depth_features  # noqa: E402


def _inputs(rows=40):
    clock = 100000 + np.arange(rows, dtype=np.int64) * 1000
    delays = np.array([100, 500], dtype=np.int64)
    cutoff = clock[:, None] - delays[None, :]
    depth = np.zeros((rows, 2, 100, 4))
    counts = np.zeros((rows, 2, 100, 2), dtype=np.int64)
    for t in range(rows):
        depth[t, :, :, 0] = 100.01 + np.arange(100) * 0.01 + t * 0.002
        depth[t, :, :, 2] = 100.00 - np.arange(100) * 0.01 + t * 0.002
        depth[t, :, :, 1] = 20 + np.arange(100) % 7
        depth[t, :, :, 3] = 25 + np.arange(100) % 5
        counts[t, :, :, 0] = 1 + np.arange(100) % 5
        counts[t, :, :, 1] = 2 + np.arange(100) % 4
    canonical = pd.DataFrame({"decision_time": clock, "bid": 99.99, "ask": 100.02,
        "bid_qty": 1000., "ask_qty": 2000.})
    observations = {"depth": depth, "order_counts": counts, "decision_times": clock, "delays_ms": delays,
        "query_cutoffs": cutoff, "publisher_times": cutoff - 1, "segment_ids": np.ones((rows, 2), dtype=np.int64),
        "known_depths": np.full((rows, 2, 2), 400, dtype=np.int16)}
    return canonical, observations


def _features(canonical, observations, **kwargs):
    return counted_depth_features(canonical, observations, delay_ms=100, levels=kwargs.pop("levels", 25),
        include_counts=kwargs.pop("include_counts", True), **kwargs)


def test_quantity_control_is_exact_when_only_order_counts_change():
    canonical, obs = _inputs()
    changed = copy.deepcopy(obs)
    changed["order_counts"][:, :, :, 0] *= 2
    original, alternative = _features(canonical, obs), _features(canonical, changed)
    quantities = [c for c in original if not c.startswith("count_")]
    pd.testing.assert_frame_equal(original[quantities], alternative[quantities], check_exact=True)
    assert not np.array_equal(original.count_25_imbalance, alternative.count_25_imbalance)
    control = _features(canonical, changed, include_counts=False)
    pd.testing.assert_frame_equal(control, original[quantities], check_exact=True)


def test_count_imbalance_and_mean_order_size_have_known_independent_values():
    canonical, obs = _inputs()
    obs["depth"][:, :, :, 1] = obs["depth"][:, :, :, 3] = 12
    obs["order_counts"][:, :, :, 0] = 3
    obs["order_counts"][:, :, :, 1] = 6
    result = _features(canonical, obs)
    assert (result.depth_25_quantity_imbalance == 0).all()
    np.testing.assert_allclose(result.count_25_imbalance, 1 / 3)
    np.testing.assert_allclose(result.count_25_mean_order_imbalance, -1 / 3)
    np.testing.assert_allclose(result.count_25_log_orders, np.log1p(25 * 9))


def test_unknown_contract_multiplier_and_local_quantity_units_do_not_change_features():
    canonical, obs = _inputs()
    original = _features(canonical, obs, levels=100)
    changed = copy.deepcopy(obs)
    changed["depth"][:, :, :, [1, 3]] *= 1024
    other = canonical.copy()
    other["bid_qty"] *= 1e12
    other["ask_qty"] *= 1e-12
    transformed = _features(other, changed, levels=100)
    np.testing.assert_allclose(original, transformed, atol=1e-10, rtol=1e-10)


def test_future_changes_cannot_change_an_earlier_feature_prefix():
    canonical, obs = _inputs()
    full = _features(canonical, obs, levels=100)
    prefix = {k: v if k == "delays_ms" else v[:23].copy() for k, v in obs.items()}
    result = _features(canonical.iloc[:23], prefix, levels=100)
    pd.testing.assert_frame_equal(full.iloc[:23], result, check_exact=True)
    changed = copy.deepcopy(obs)
    changed["order_counts"][23:, :, :, 0] *= 9
    changed["depth"][23:, :, :, [1, 3]] *= 100
    pd.testing.assert_frame_equal(full.iloc[:23], _features(canonical, changed, levels=100).iloc[:23], check_exact=True)


def test_missing_rows_and_recovery_segments_break_every_crossing_history_window():
    canonical, obs = _inputs()
    for key in ("depth", "order_counts", "known_depths"):
        obs[key][20] = 0
    obs["publisher_times"][20] = obs["segment_ids"][20] = -1
    result = _features(canonical, obs)
    assert (result.iloc[20] == 0).all()
    assert result.venue_history_available_5s.iloc[25] == 0
    assert result.venue_history_available_5s.iloc[26] == 1
    obs["segment_ids"][22:] = 2
    result = _features(canonical, obs)
    assert result.venue_history_available_5s.iloc[26] == 0
    assert result.venue_history_available_5s.iloc[27] == 1
    assert result.count_25_sampled_count_pressure_5s.iloc[26] == 0


def test_insufficient_depth_keeps_rows_and_masks_only_the_affected_representation():
    canonical, obs = _inputs()
    obs["known_depths"][:, :, :] = 25
    for key in ("depth", "order_counts"):
        obs[key][:, :, 25:] = 0
    shallow = _features(canonical, obs, levels=25)
    deep = _features(canonical, obs, levels=100)
    assert len(shallow) == len(deep) == len(canonical)
    assert (shallow.venue_available == 1).all()
    assert (deep.to_numpy() == 0).all()


def test_causal_boundary_and_integer_counts_are_validated_before_feature_use():
    canonical, obs = _inputs()
    variants = []
    equal = copy.deepcopy(obs)
    equal["publisher_times"][10, 0] = equal["query_cutoffs"][10, 0]
    variants.append(equal)
    stale = copy.deepcopy(obs)
    stale["publisher_times"][10, 0] = stale["query_cutoffs"][10, 0] - 1001
    variants.append(stale)
    fractional = copy.deepcopy(obs)
    fractional["order_counts"] = fractional["order_counts"].astype(float)
    variants.append(fractional)
    for value in variants:
        with pytest.raises(ValueError):
            _features(canonical, value)
