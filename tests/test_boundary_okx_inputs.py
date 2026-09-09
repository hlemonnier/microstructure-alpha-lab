import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_okx_inputs import PEER_COUNTS, PEER_QUANTITY, append_counted_features, counted_columns  # noqa: E402


def _frames():
    times = np.array([1000, 2000, 3000], dtype=np.int64)
    original = pd.DataFrame({"original_observation": [1., 2., 3.]})
    own = pd.DataFrame({name: np.arange(3, dtype=float) for name in (*PEER_QUANTITY, *PEER_COUNTS)}, index=times)
    peer = own + 10
    return original, times, own, peer


def test_counted_join_retains_original_rows_and_removes_only_count_columns():
    original, times, own, peer = _frames()
    full = append_counted_features(original, times, own, peer, include_counts=True)
    control = append_counted_features(original, times, own, peer, include_counts=False)
    quantities = [c for c in full if not c.startswith(("okx_count_", "peer_okx_count_"))]
    pd.testing.assert_frame_equal(full[quantities], control, check_exact=True)
    pd.testing.assert_frame_equal(full[original.columns], original, check_exact=True)
    np.testing.assert_array_equal(full.peer_okx_count_25_imbalance, peer.count_25_imbalance)


def test_joins_fail_on_missing_peer_clock_or_duplicate_observation_identity():
    original, times, own, peer = _frames()
    with pytest.raises(KeyError):
        append_counted_features(original, times, own, peer.iloc[:2], include_counts=True)
    peer.index = [1000, 1000, 3000]
    with pytest.raises(ValueError):
        append_counted_features(original, times, own, peer, include_counts=True)
    with pytest.raises(ValueError):
        counted_columns(["venue_available", "label"], True)
