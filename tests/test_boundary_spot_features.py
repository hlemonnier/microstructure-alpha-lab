import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_spot_features import SPOT_FEATURES, spot_feature_frame  # noqa: E402


def test_spot_information_boundary_and_window_arithmetic():
    known = pd.DataFrame({"decision_time": [1000,1100,1101,1500,2100], "bid": 99.0, "ask": 101.0, "bid_qty": 4.0, "ask_qty": 6.0})
    trades = pd.DataFrame({"transact_time": [800,900,1400], "agg_trade_id": [1,2,3], "price": [99.0,100.0,101.0], "quantity": [1.0,2.0,3.0], "is_buyer_maker": [True,False,False]})
    result = spot_feature_frame(known, trades, information_delay_ms=100)
    assert result.spot_time_1000_log_count.iloc[0] == np.log1p(1)
    assert result.spot_time_1000_volume_imbalance.iloc[1] == pytest.approx(1/3)
    assert result.spot_last_basis_bps.iloc[3] == 0  # t=1400 trade +100ms is not available at t=1500.
    assert result.spot_last_basis_bps.iloc[4] == pytest.approx(np.log(1.01)*10000)
    assert result.spot_events_8_side_persistence.iloc[-1] == 0
    assert result.spot_events_8_volume_concentration.iloc[-1] == pytest.approx(14/36)
    pd.testing.assert_frame_equal(result.iloc[:4], spot_feature_frame(known.iloc[:4], trades.iloc[:2], information_delay_ms=100), check_exact=True)
    delayed = spot_feature_frame(known, trades, information_delay_ms=500)
    np.testing.assert_array_equal(delayed.spot_available, [0,0,0,1,1])


def test_spot_features_use_no_future_targets_and_support_absent_trade_history():
    known = pd.DataFrame({"decision_time": [1000,2000,3000], "bid": 99.0, "ask": 101.0, "bid_qty": 4.0, "ask_qty": 6.0, "label": [-1,0,1], "future_mid": [1e6,0,1]})
    trades = pd.DataFrame({"transact_time": [800,2900], "agg_trade_id": [1,2], "price": [100.0,101.0], "quantity": [2.0,3.0], "is_buyer_maker": [False,True]})
    original = known.copy(deep=True)
    result = spot_feature_frame(known, trades, information_delay_ms=100)
    known["label"], known["future_mid"] = 1, -1
    pd.testing.assert_frame_equal(result, spot_feature_frame(known, trades, information_delay_ms=100), check_exact=True)
    pd.testing.assert_frame_equal(original.drop(columns=["label", "future_mid"]), known.drop(columns=["label", "future_mid"]))
    absent = spot_feature_frame(known, trades.iloc[:0], information_delay_ms=500)
    assert list(absent.columns) == ["decision_time", *SPOT_FEATURES]
    np.testing.assert_array_equal(absent[list(SPOT_FEATURES)].to_numpy(), 0)
    with pytest.raises(ValueError):
        spot_feature_frame(known, trades, information_delay_ms=0)
