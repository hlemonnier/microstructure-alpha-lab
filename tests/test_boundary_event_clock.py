import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_event_clock import EVENT_CLOCK_FEATURES, event_clock_frame  # noqa: E402


def quotes():
    return pd.DataFrame({
        "update_id": [1, 2, 3, 4, 5, 6], "event_time": [100, 800, 900, 1000, 1100, 2100],
        "best_bid_price": [100, 100, 100, 101, 101, 101], "best_ask_price": [102, 102, 102, 103, 103, 103],
        "best_bid_qty": [10, 15, 12, 100, 101, 110], "best_ask_qty": [10, 8, 10, 100, 102, 110],
    })


def trades():
    return pd.DataFrame({
        "agg_trade_id": [1, 2, 3, 4, 5, 6],
        "transact_time": [200, 800, 900, 1000, 1200, 2000],
        "price": [100, 102, 101, 102, 101, 103], "quantity": [1, 3, 2, 4, 8, 5],
        "is_buyer_maker": [True, False, True, False, False, False],
    })


def test_trade_clock_features_observe_delay_and_have_correct_units():
    x = event_clock_frame(quotes(), trades(), [1000])
    # Only trades at 200 and 800 are available: the 900+100 equality is excluded.
    assert x.loc[0, "clock_trade_8_side_mean"] == 0
    assert x.loc[0, "clock_trade_8_volume_imbalance"] == 0.5
    assert x.loc[0, "clock_trade_8_side_persistence"] == -1
    assert x.loc[0, "clock_trade_8_volume_concentration"] == 10 / 16
    assert x.loc[0, "clock_trade_8_largest_volume_share"] == 3 / 4
    assert x.loc[0, "clock_last_trade_offset_spreads"] == 0.5
    assert x.loc[0, "clock_trade_8_vwap_offset_spreads"] == 0.25
    np.testing.assert_allclose(x.loc[0, "clock_trade_8_signed_volume_depth"], 2 / 22)
    np.testing.assert_allclose(x.loc[0, "clock_trade_8_price_return_bps"], 2 / 101 * 10000)


def test_queue_changes_exclude_price_resets_and_future_boundary_events():
    x = event_clock_frame(quotes(), trades(), [1000, 1200])
    # At 1000 the known depth is 12+10, and only 800/900 changes belong to 200ms.
    np.testing.assert_allclose(x.loc[0, "clock_bid_increase_depth_200ms"], 5 / 22)
    np.testing.assert_allclose(x.loc[0, "clock_bid_decrease_depth_200ms"], 3 / 22)
    np.testing.assert_allclose(x.loc[0, "clock_ask_increase_depth_200ms"], 2 / 22)
    np.testing.assert_allclose(x.loc[0, "clock_ask_decrease_depth_200ms"], 2 / 22)
    # The size reset from 12 to 100 at a changed best price is not replenishment.
    np.testing.assert_allclose(x.loc[1, "clock_bid_increase_depth_200ms"], 1 / 203)
    np.testing.assert_allclose(x.loc[1, "clock_ask_increase_depth_200ms"], 2 / 203)


def test_appending_or_changing_future_events_cannot_change_observed_prefix():
    q, t = quotes(), trades()
    full = event_clock_frame(q, t, [1000, 1200, 2200])
    prefix = event_clock_frame(q[q.event_time < 1200], t[t.transact_time + 100 < 1200], [1000, 1200])
    np.testing.assert_allclose(full.iloc[:2].to_numpy(), prefix.to_numpy(), rtol=0, atol=0)
    changed_q, changed_t = q.copy(), t.copy()
    changed_q.loc[changed_q.event_time >= 1000, "best_bid_qty"] *= 20
    changed_t.loc[changed_t.transact_time + 100 >= 1000, "quantity"] *= 50
    other = event_clock_frame(changed_q, changed_t, [1000, 1200, 2200])
    np.testing.assert_array_equal(full.iloc[0].to_numpy(), other.iloc[0].to_numpy())


def test_empty_and_zero_volume_trade_histories_stay_finite():
    q, t = quotes(), trades()
    empty = event_clock_frame(q, t.iloc[:0], [500, 1000])
    assert not empty["clock_trade_available"].any()
    t["quantity"] = 0
    zero = event_clock_frame(q, t, [150, 500, 1000])
    assert len(EVENT_CLOCK_FEATURES) == len(set(EVENT_CLOCK_FEATURES))
    assert len(zero.columns) == len(EVENT_CLOCK_FEATURES) + 1
    assert np.isfinite(zero.to_numpy()).all()
    assert not zero["clock_trade_8_volume_concentration"].any()
    assert not zero["clock_trade_8_vwap_offset_spreads"].any()


def test_invalid_event_order_and_crossed_quotes_are_rejected():
    q, t = quotes(), trades()
    for broken in (q.iloc[::-1], q.assign(best_ask_price=99)):
        with pytest.raises(ValueError):
            event_clock_frame(broken, t, [1000])
    with pytest.raises(ValueError):
        event_clock_frame(q, t.iloc[::-1], [1000])
    with pytest.raises(ValueError):
        event_clock_frame(q, t, [100])
