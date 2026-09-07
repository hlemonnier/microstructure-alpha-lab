import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_bybit_flow_features import bybit_flow_features  # noqa: E402


def flow_observations():
    clock = np.arange(6, dtype=np.int64) * 1000 + 100000
    row = np.array([3, 3, 0, 0] * 3 + [0, 6, 0.2, 0.6, 10, 20, 5], dtype=float)
    return {"decision_times": clock, "delays_ms": np.array([100, 500]), "windows_ms": np.array([1000, 5000, 15000]),
            "flow_totals": np.broadcast_to(row, (6, 2, 3, 19)).copy(), "window_available": np.ones((6, 2, 3), dtype=bool),
            "source_publisher_times": clock[:, None] - np.array([101, 501]), "top_depth": np.full((6, 2), 20.0)}


def test_native_flow_features_preserve_round_trip_information_and_matched_control():
    source = flow_observations()
    clock = source["decision_times"]
    shallow = bybit_flow_features(clock, source, delay_ms=100, include_deep=False)
    deep = bybit_flow_features(clock, source, delay_ms=100, include_deep=True)
    assert len(shallow.columns) == 24 and len(deep.columns) == 93
    pd.testing.assert_frame_equal(deep[shallow.columns], shallow, check_exact=True)
    assert deep.native_bbo_pressure_1s.iloc[0] == 0
    assert deep.native_bbo_gross_1s.iloc[0] == pytest.approx(np.log1p(6 / 20))
    assert deep.native_deep_w5_recycling_1s.iloc[0] == 1
    assert deep.native_deep_unresolved_fraction_1s.iloc[0] == 0.2
    changed = {k: v.copy() for k, v in source.items()}
    changed["flow_totals"][..., 0:12] *= 8
    pd.testing.assert_frame_equal(shallow, bybit_flow_features(clock, changed, delay_ms=100, include_deep=False), check_exact=True)


def test_native_flow_features_respect_quantity_units_prefixes_and_information_boundaries():
    source = flow_observations()
    clock = source["decision_times"]
    baseline = bybit_flow_features(clock, source, delay_ms=500, include_deep=True)
    changed = {k: v.copy() for k, v in source.items()}
    changed["flow_totals"][..., :14] *= 11
    changed["top_depth"] *= 11
    pd.testing.assert_frame_equal(baseline, bybit_flow_features(clock, changed, delay_ms=500, include_deep=True), atol=1e-14, rtol=1e-14)
    prefix = {k: (v.copy() if k in ("delays_ms", "windows_ms") else v[:3].copy()) for k, v in source.items()}
    pd.testing.assert_frame_equal(baseline.iloc[:3], bybit_flow_features(clock[:3], prefix, delay_ms=500, include_deep=True), check_exact=True)
    source["window_available"][0] = False
    source["top_depth"][0] = 0
    source["source_publisher_times"][0] = -1
    assert not bybit_flow_features(clock, source, delay_ms=500, include_deep=True).iloc[0].any()
    source["source_publisher_times"][1, 1] = clock[1] - 500
    with pytest.raises(ValueError, match="strict"):
        bybit_flow_features(clock, source, delay_ms=500, include_deep=True)


def test_native_flow_side_reflection_reverses_pressure_and_preserves_activity():
    source = flow_observations()
    source["flow_totals"][..., 0] = 8
    source["flow_totals"][..., 12] = 2
    reflected = {k: v.copy() for k, v in source.items()}
    for i in range(3):
        reflected["flow_totals"][..., 4 * i:4 * i + 4] = source["flow_totals"][..., [4 * i + 2, 4 * i + 3, 4 * i, 4 * i + 1]]
    reflected["flow_totals"][..., [12, 14]] *= -1
    a = bybit_flow_features(source["decision_times"], source, delay_ms=100, include_deep=True)
    b = bybit_flow_features(source["decision_times"], reflected, delay_ms=100, include_deep=True)
    np.testing.assert_array_equal(a.native_deep_w1_pressure_1s, -b.native_deep_w1_pressure_1s)
    np.testing.assert_array_equal(a.native_deep_w1_recycling_1s, b.native_deep_w1_recycling_1s)
    np.testing.assert_array_equal(a.native_bbo_efficiency_1s, -b.native_bbo_efficiency_1s)
    np.testing.assert_array_equal(a.native_mid_path_length_1s, b.native_mid_path_length_1s)
