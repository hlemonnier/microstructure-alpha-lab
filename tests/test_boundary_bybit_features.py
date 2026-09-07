import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_bybit_features import bybit_depth_features  # noqa: E402


def observations():
    n = 40
    clock = np.arange(n) * 1000 + 100000
    raw = np.zeros((n, 2, 25, 4))
    for row in range(n):
        for level in range(25):
            raw[row, :, level] = [101 + level + row / 100, 2 + level, 99 - level + row / 100, 3 + row / 10]
    source = {"decision_times": clock, "delays_ms": np.array([100, 500]), "depth": raw,
        "publisher_times": clock[:, None] - np.array([101, 501]), "available": np.ones((n, 2), dtype=bool)}
    frame = pd.DataFrame({"decision_time": clock, "bid": 99.0, "ask": 101.0, "bid_qty": 2.0, "ask_qty": 3.0, "label": 1})
    return frame, source


def test_foreign_depth_features_separate_level_information_and_preserve_prefixes():
    frame, source = observations()
    shallow = bybit_depth_features(frame, source, delay_ms=100, levels=1)
    deep = bybit_depth_features(frame, source, delay_ms=100, levels=25)
    pd.testing.assert_frame_equal(deep[shallow.columns], shallow, check_exact=True)
    altered = {k: v.copy() for k, v in source.items()}
    altered["depth"][:, :, 1:, 3] *= 5
    pd.testing.assert_frame_equal(shallow, bybit_depth_features(frame, altered, delay_ms=100, levels=1), check_exact=True)
    assert not np.allclose(deep.depth_25_imbalance, bybit_depth_features(frame, altered, delay_ms=100, levels=25).depth_25_imbalance)
    prefix = {k: (v.copy() if k == "delays_ms" else v[:20].copy()) for k, v in source.items()}
    pd.testing.assert_frame_equal(deep.iloc[:20], bybit_depth_features(frame.iloc[:20], prefix, delay_ms=100, levels=25), check_exact=True)
    frame["label"] = -1
    pd.testing.assert_frame_equal(deep, bybit_depth_features(frame, source, delay_ms=100, levels=25), check_exact=True)
    assert shallow.venue_top_imbalance.iloc[0] == pytest.approx(0.2)
    assert deep.depth_5_imbalance.iloc[0] == pytest.approx((15 - 20) / 35)


def test_foreign_depth_features_respect_units_unavailability_and_strict_delay():
    frame, source = observations()
    baseline = bybit_depth_features(frame, source, delay_ms=500, levels=25)
    scaled = {k: v.copy() for k, v in source.items()}
    scaled["depth"][..., [0, 2]] *= 4
    scaled["depth"][..., [1, 3]] *= 3
    changed = frame.copy()
    changed[["bid", "ask"]] *= 4
    changed[["bid_qty", "ask_qty"]] *= 3
    pd.testing.assert_frame_equal(baseline, bybit_depth_features(changed, scaled, delay_ms=500, levels=25), atol=1e-12, rtol=1e-12)
    source["available"][0] = False
    source["depth"][0] = 0
    source["publisher_times"][0] = -1
    result = bybit_depth_features(frame, source, delay_ms=500, levels=25)
    np.testing.assert_array_equal(result.iloc[0], 0)
    assert result.venue_history_available_1s.iloc[1] == 0
    source["publisher_times"][2, 1] = frame.decision_time.iloc[2] - 500
    with pytest.raises(ValueError):
        bybit_depth_features(frame, source, delay_ms=500, levels=25)
