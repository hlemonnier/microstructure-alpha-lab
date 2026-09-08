import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_tardis_features import tardis_depth_features  # noqa: E402


def fixture():
    n = 40
    clock = np.arange(n) * 1000 + 100000
    raw = np.zeros((n, 2, 25, 4))
    for level in range(25):
        raw[:, :, level] = [101 + level, 2 + level, 99 - level, 3 + level]
    observed = {"decision_times": clock, "delays_ms": np.array([100, 500]), "depth": raw,
        "publisher_times_ms": clock[:, None] - np.array([102, 502]),
        "capture_times_ns": (clock[:, None] - np.array([101, 501])) * 1_000_000,
        "available": np.ones((n, 2), dtype=bool)}
    canonical = pd.DataFrame({"decision_time": clock, "bid": 99.0, "ask": 101.0, "bid_qty": 2.0, "ask_qty": 3.0, "label": 1})
    return canonical, observed


def test_native_depth_features_match_bbo_control_and_never_read_outcomes():
    frame, observations = fixture()
    bbo = tardis_depth_features(frame, observations, delay_ms=100, levels=1)
    deep = tardis_depth_features(frame, observations, delay_ms=100, levels=25)
    pd.testing.assert_frame_equal(bbo, deep[bbo.columns], check_exact=True)
    changed = {k: v.copy() for k, v in observations.items()}
    changed["depth"][:, :, 1:, 3] *= 9
    pd.testing.assert_frame_equal(bbo, tardis_depth_features(frame, changed, delay_ms=100, levels=1), check_exact=True)
    assert not deep.equals(tardis_depth_features(frame, changed, delay_ms=100, levels=25))
    prefix = {k: v.copy() if k == "delays_ms" else v[:20].copy() for k, v in observations.items()}
    pd.testing.assert_frame_equal(deep.iloc[:20], tardis_depth_features(frame.iloc[:20], prefix, delay_ms=100, levels=25), check_exact=True)
    frame.label = -1
    pd.testing.assert_frame_equal(deep, tardis_depth_features(frame, observations, delay_ms=100, levels=25), check_exact=True)
    assert (len(bbo.columns), len(deep.columns)) == (27, 88)


def test_native_depth_requires_both_strict_clocks_and_masks_unavailable_metadata():
    frame, observations = fixture()
    observations["capture_times_ns"][0, 0] = (frame.decision_time.iloc[0] - 100) * 1_000_000
    with pytest.raises(ValueError, match="strict capture"):
        tardis_depth_features(frame, observations, delay_ms=100, levels=1)
    observations["available"][0] = False
    observations["publisher_times_ms"][0] = frame.decision_time.iloc[0] + 5000
    observations["capture_times_ns"][0] = (frame.decision_time.iloc[0] + 6000) * 1_000_000
    result = tardis_depth_features(frame, observations, delay_ms=100, levels=25)
    np.testing.assert_array_equal(result.iloc[0], 0)
    assert result.venue_history_available_1s.iloc[1] == 0
    observations["publisher_times_ms"][1, 1] = frame.decision_time.iloc[1] - 500
    with pytest.raises(ValueError, match="strict capture"):
        tardis_depth_features(frame, observations, delay_ms=500, levels=25)
