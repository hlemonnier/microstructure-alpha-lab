import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_quote_currency_features import converted_spot_feature_frame  # noqa: E402
from lob_forge.boundary_spot_features import SPOT_FEATURES, spot_feature_frame  # noqa: E402


def _data():
    known = pd.DataFrame({"decision_time": [1000, 1001, 3000, 5000], "bid": 99., "ask": 101., "bid_qty": 4., "ask_qty": 6.})
    trades = pd.DataFrame({"transact_time": [800, 2400, 4500], "agg_trade_id": [1, 2, 3],
        "price": [50., 60., 55.], "quantity": [2., 3., 4.], "is_buyer_maker": [False, True, False]})
    conversion = pd.DataFrame({"publisher_time": [800, 900, 4500], "price": [2., 4., 2.]})
    return known, trades, conversion


def _features(known, trades, conversion, maximum_age=1000):
    return converted_spot_feature_frame(known, trades, conversion, information_delay_ms=100,
        conversion_delay_ms=100, maximum_conversion_age_ms=maximum_age)


def test_conversion_direction_strict_cutoff_and_future_prefix():
    known, trades, conversion = _data()
    result = _features(known, trades, conversion)
    assert result.spot_last_basis_bps.iloc[0] == 0
    assert result.spot_last_basis_bps.iloc[1] == pytest.approx(np.log(2)*10000)
    assert result.spot_last_basis_bps.iloc[-1] == pytest.approx(np.log(1.1)*10000)
    changed = conversion.copy()
    changed.loc[2, "price"] = 1e9
    pd.testing.assert_frame_equal(result.iloc[:3], _features(known, trades, changed).iloc[:3], check_exact=True)
    pd.testing.assert_frame_equal(result.iloc[:2], _features(known.iloc[:2], trades.iloc[:1], conversion.iloc[:2]), check_exact=True)


def test_stale_conversion_keeps_native_information_without_contaminating_later_basis():
    known, trades, conversion = _data()
    result = _features(known, trades, conversion)
    assert result.conversion_available.iloc[2] == 0
    assert result.spot_available.iloc[2] == 1
    assert result.spot_time_1000_log_quantity.iloc[2] == pytest.approx(np.log1p(3))
    assert not result.filter(like="basis").iloc[2].any()
    changed = known.copy()
    changed.loc[2, ["bid", "ask"]] = [999999., 1000001.]
    pd.testing.assert_series_equal(result.iloc[-1], _features(changed, trades, conversion).iloc[-1], check_exact=True)
    absent = _features(known, trades, conversion.iloc[:0])
    assert not absent.filter(like="basis").to_numpy().any()
    np.testing.assert_array_equal(absent.spot_time_1000_log_quantity, result.spot_time_1000_log_quantity)


def test_native_quote_unit_change_preserves_converted_basis_returns_and_flow():
    known, trades, conversion = _data()
    result = _features(known, trades, conversion)
    rescaled_trades, rescaled_conversion = trades.copy(), conversion.copy()
    rescaled_trades["price"] *= 10
    rescaled_conversion["price"] /= 10
    rescaled = _features(known, rescaled_trades, rescaled_conversion)
    np.testing.assert_allclose(result[list(SPOT_FEATURES)], rescaled[list(SPOT_FEATURES)], atol=1e-9, rtol=0)
    present = result.conversion_available.to_numpy(dtype=bool)
    np.testing.assert_allclose(rescaled.conversion_log_target_quote_per_native_quote[present],
        result.conversion_log_target_quote_per_native_quote[present] - np.log(10), atol=1e-14, rtol=0)


def test_identity_conversion_reproduces_original_features_and_ignores_labels():
    known, trades, conversion = _data()
    conversion["price"] = 1.
    expected = spot_feature_frame(known, trades, information_delay_ms=100)
    actual = _features(known, trades, conversion, maximum_age=60000)
    pd.testing.assert_frame_equal(expected, actual[["decision_time", *SPOT_FEATURES]], check_exact=True)
    known["label"], known["future_mid"] = [1, 0, -1, 1], 1e9
    pd.testing.assert_frame_equal(actual, _features(known, trades, conversion, maximum_age=60000), check_exact=True)
    conversion["publisher_time"] = conversion.publisher_time.astype(np.float32)
    with pytest.raises(ValueError, match="integer milliseconds"):
        _features(known, trades, conversion)


def test_unsigned_regressions_and_noninteger_delays_are_rejected():
    known, trades, conversion = _data()
    conversion["publisher_time"] = np.array([900, 800, 4500], dtype=np.uint64)
    with pytest.raises(ValueError, match="unsigned subtraction"):
        _features(known, trades, conversion)
    known, trades, conversion = _data()
    with pytest.raises(ValueError, match="integer milliseconds"):
        converted_spot_feature_frame(known, trades, conversion, information_delay_ms=100,
            conversion_delay_ms=100., maximum_conversion_age_ms=60000)
