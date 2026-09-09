import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_conversion_midpoint import (  # noqa: E402
    side_adjusted_conversion_frame, side_aware_converted_spot_frame,
)
from lob_forge.boundary_quote_currency_features import converted_spot_feature_frame  # noqa: E402


def _data():
    canonical = pd.DataFrame({"decision_time": np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        "bid": [999., 999., 999., 999.], "ask": [1001., 1001., 1001., 1001.], "bid_qty": 1., "ask_qty": 1.})
    native = pd.DataFrame({"transact_time": np.array([1, 1001, 2001, 3001], dtype=np.int64),
        "agg_trade_id": np.arange(4), "price": [1000., 1000., 1000., 1000.], "quantity": 1.,
        "is_buyer_maker": [False, True, False, True]})
    fx = native[["transact_time", "price", "is_buyer_maker"]].copy()
    fx["price"] = [.9995, .9994, .9995, .9994]
    return canonical, native, fx


def _features(canonical, native, fx, half):
    return side_aware_converted_spot_frame(canonical, native, fx, half_spacing=half,
        information_delay_ms=100, conversion_delay_ms=100, maximum_conversion_age_ms=1500)


def test_taker_sign_correction_removes_a_known_one_spacing_bounce_without_changing_clocks():
    _, _, fx = _data()
    adjusted = side_adjusted_conversion_frame(fx, half_spacing=.00005)
    # Decimal reference versus one binary subtraction: permit exactly one ULP.
    np.testing.assert_allclose(adjusted.price, [.99945]*4, rtol=0, atol=np.spacing(.99945))
    np.testing.assert_array_equal(adjusted.trade_side, [1., -1., 1., -1.])
    np.testing.assert_array_equal(adjusted.publisher_time, fx.transact_time)
    raw = side_adjusted_conversion_frame(fx, half_spacing=0.)
    np.testing.assert_array_equal(raw.price, fx.price)
    np.testing.assert_array_equal(raw.trade_side, adjusted.trade_side)


def test_raw_plus_side_control_replays_original_features_and_adjustment_keeps_native_flow():
    canonical, native, fx = _data()
    raw, adjusted = _features(canonical, native, fx, 0.), _features(canonical, native, fx, .00005)
    plain_conversion = fx[["transact_time", "price"]].rename(columns={"transact_time": "publisher_time"})
    original = converted_spot_feature_frame(canonical, native, plain_conversion,
        information_delay_ms=100, conversion_delay_ms=100, maximum_conversion_age_ms=1500)
    pd.testing.assert_frame_equal(raw[original.columns], original, check_exact=True)
    native_fields = [c for c in original if "basis" not in c and c != "conversion_log_target_quote_per_native_quote"]
    pd.testing.assert_frame_equal(adjusted[native_fields], raw[native_fields], check_exact=True)
    np.testing.assert_array_equal(raw.conversion_last_side, adjusted.conversion_last_side)
    expected = 10000*np.log(np.array([.99945]*4)/fx.price.to_numpy())
    np.testing.assert_allclose(adjusted.spot_last_basis_bps-raw.spot_last_basis_bps, expected, rtol=0, atol=2e-12)


def test_conversion_side_uses_strict_delay_and_same_staleness_mask_as_rate():
    canonical, native, fx = _data()
    canonical["decision_time"] = np.array([101, 102, 1602, 2000], dtype=np.int64)
    one = fx.iloc[:1]
    frame = _features(canonical, native, one, .00005)
    np.testing.assert_array_equal(frame.conversion_available, [0., 1., 0., 0.])
    np.testing.assert_array_equal(frame.conversion_last_side, [0., 1., 0., 0.])


def test_future_source_changes_cannot_change_earlier_features_and_units_scale_together():
    canonical, native, fx = _data()
    expected = _features(canonical, native, fx, .00005)
    changed = fx.copy()
    changed.loc[3, "price"] = 2.
    changed.loc[3, "is_buyer_maker"] = False
    actual = _features(canonical, native, changed, .00005)
    pd.testing.assert_frame_equal(expected.iloc[:3], actual.iloc[:3], check_exact=True)
    scaled = fx.copy()
    scaled["price"] *= 100.
    left = side_adjusted_conversion_frame(scaled, half_spacing=.005)
    right = side_adjusted_conversion_frame(fx, half_spacing=.00005)
    np.testing.assert_allclose(left.price, right.price*100., rtol=0, atol=2e-14)
    fx["label"] = [1, 0, -1, 0]
    pd.testing.assert_frame_equal(expected, _features(canonical, native, fx, .00005), check_exact=True)


def test_invalid_price_side_clock_and_half_spacing_fail_explicitly():
    _, _, fx = _data()
    for half in (-.1, float("nan"), True, "0.00005"):
        with pytest.raises(ValueError):
            side_adjusted_conversion_frame(fx, half_spacing=half)
    with pytest.raises(ValueError):
        side_adjusted_conversion_frame(fx, half_spacing=1.)
    wrong = fx.copy()
    wrong["is_buyer_maker"] = [0, 1, 0, 1]
    with pytest.raises(ValueError):
        side_adjusted_conversion_frame(wrong, half_spacing=.00005)
    wrong = fx.copy()
    wrong["transact_time"] = np.array([1, 2000, 1000, 3000], dtype=np.uint64)
    with pytest.raises(ValueError):
        side_adjusted_conversion_frame(wrong, half_spacing=.00005)
