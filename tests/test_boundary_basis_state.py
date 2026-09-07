import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from lob_forge.boundary_basis_state import basis_state_features, prior_basis_moments  # noqa: E402


def test_prior_basis_moments_match_explicit_past_weights_with_missing_records_and_reset():
    clock = np.array([0, 1000, 2000, 3000, 4000, 6000, 7000], dtype=np.int64)
    z = np.array([5.0, 6.0, 1000.0, 9.0, 7.0, 20.0, 21.0])
    valid = np.array([True, True, False, True, True, True, True])
    actual = prior_basis_moments(z, clock, valid, half_life_seconds=10)
    for i in (1, 3, 4, 6):
        start = 5 if i == 6 else 0
        indices = np.arange(start, i)[valid[start:i]]
        w = np.exp(-np.log(2) * (clock[i] - clock[indices]) / 10000)
        mean = np.average(z[indices], weights=w)
        std = np.sqrt(np.average((z[indices] - mean) ** 2, weights=w))
        delta = z[i] - mean
        np.testing.assert_allclose(actual[i], [mean, delta, delta / max(std, 0.001), np.log1p(std), 1], atol=1e-10)
    np.testing.assert_array_equal(actual[[0, 2, 5]], np.zeros((3, 5)))
    changed = z.copy()
    changed[2] = -1000000
    np.testing.assert_array_equal(prior_basis_moments(changed, clock, valid, half_life_seconds=10), actual)


def test_basis_moments_are_prefix_causal_and_innovations_ignore_constant_premium():
    clock = np.arange(12, dtype=np.int64) * 1000
    z = np.array([0, 0, 7, 8, 5, 9, 6, 7, 12, 10, 9, 8], dtype=float)
    available = np.arange(12) >= 2
    full = prior_basis_moments(z, clock, available, half_life_seconds=60)
    for end in range(1, len(z) + 1):
        np.testing.assert_array_equal(prior_basis_moments(z[:end], clock[:end], available[:end], half_life_seconds=60), full[:end])
    shifted = prior_basis_moments(z + 1000000, clock, available, half_life_seconds=60)
    np.testing.assert_array_equal(shifted[:, 1:], full[:, 1:])
    np.testing.assert_allclose(shifted[3:, 0] - 1000000, full[3:, 0], atol=1e-9)


def test_observed_basis_features_ignore_labels_and_future_values_and_share_log_units():
    n = 20
    frame = pd.DataFrame({"foreign_venue_mid_basis_bps": np.linspace(1, 3, n),
                          "foreign_venue_available": np.ones(n), "aux_last_basis_bps": np.linspace(0.1, 0.9, n),
                          "aux_available": np.ones(n), "label": np.ones(n)})
    clock = np.arange(n, dtype=np.int64) * 1000
    result = basis_state_features(frame, clock)
    assert len(result.columns) == 32
    pd.testing.assert_frame_equal(basis_state_features(frame.iloc[:10], clock[:10]), result.iloc[:10], check_exact=True)
    pd.testing.assert_frame_equal(basis_state_features(frame.assign(label=-1), clock), result, check_exact=True)
    first_log = 10000 * np.log1p(frame.foreign_venue_mid_basis_bps.iloc[0] / 10000)
    assert result.basis_state_bybit_mean_bps_60s.iloc[1] == first_log
    assert result.basis_state_bybit_spot_mean_bps_60s.iloc[1] == first_log - 0.1
    with pytest.raises(ValueError, match="positive"):
        basis_state_features(frame.assign(foreign_venue_mid_basis_bps=-10000), clock)
