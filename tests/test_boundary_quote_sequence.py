import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_quote_sequence import (  # noqa: E402
    SEQUENCE_CHANNELS,
    QuoteSequenceSource,
    select_sequence_rows,
)


def quotes():
    return pd.DataFrame({"event_time": np.array([100, 100, 200, 250, 300, 400, 500, 600, 700, 800]),
        "update_id": np.arange(10), "best_bid_price": [100, 100, 101, 101, 101, 100, 100, 100, 100, 101],
        "best_ask_price": [102, 102, 103, 103, 102, 102, 102, 102, 102, 103],
        "best_bid_qty": [2, 5, 4, 6, 3, 7, 5, 8, 2, 9],
        "best_ask_qty": [4, 3, 5, 4, 2, 3, 1, 5, 2, 6]})


def test_quote_sequences_use_strict_events_and_preserve_prefixes():
    frame = quotes()
    decisions = np.array([101, 200, 251, 801])
    result = QuoteSequenceSource.from_frame(frame).observe(decisions)
    assert result.shape == (4, 2, 64, 14)
    # The quote at the decision timestamp is excluded, including all tied IDs.
    np.testing.assert_array_equal(result[0, :, :, :8], result[1, :, :, :8])
    np.testing.assert_array_equal(result[0, 0, :-2], 0)
    np.testing.assert_array_equal(result[:3], QuoteSequenceSource.from_frame(frame.iloc[:4]).observe(decisions[:3]))
    changed = frame.copy()
    changed.loc[4:, "best_bid_qty"] *= 100
    np.testing.assert_array_equal(result[:3], QuoteSequenceSource.from_frame(changed).observe(decisions[:3]))
    assert result[0, 0, -1, SEQUENCE_CHANNELS.index("log_endpoint_age_ms")] == pytest.approx(np.log(2))


def test_quote_block_flows_retain_every_event_and_respect_units():
    frame = quotes()
    result = QuoteSequenceSource.from_frame(frame).observe(np.array([801]))
    ofi = []
    for i in range(2, 10):  # The newest coarse block contains precisely events 2..9.
        now, prev = frame.iloc[i], frame.iloc[i - 1]
        ofi.append((now.best_bid_price >= prev.best_bid_price) * now.best_bid_qty
                   - (now.best_bid_price <= prev.best_bid_price) * prev.best_bid_qty
                   - (now.best_ask_price <= prev.best_ask_price) * now.best_ask_qty
                   + (now.best_ask_price >= prev.best_ask_price) * prev.best_ask_qty)
    depth = 15
    index = SEQUENCE_CHANNELS.index("ofi_depth")
    assert result[0, 1, -1, index] == pytest.approx(np.sign(sum(ofi)) * np.log1p(abs(sum(ofi)) / depth))
    assert result[0, 1, -1, SEQUENCE_CHANNELS.index("absolute_ofi_depth")] == pytest.approx(np.log1p(sum(abs(v) for v in ofi) / depth))
    transformed = frame.copy()
    for name in ("best_bid_price", "best_ask_price"):
        transformed[name] = 10 * transformed[name] + 1000
    for name in ("best_bid_qty", "best_ask_qty"):
        transformed[name] *= 3
    transformed["event_time"] += 10000
    np.testing.assert_array_equal(result, QuoteSequenceSource.from_frame(transformed).observe(np.array([10801])))


def test_sequence_alignment_and_invalid_observations_fail_closed():
    np.testing.assert_array_equal(select_sequence_rows([10, 20, 30], [10, 30]), [0, 2])
    for saved, requested in [([10, 20], [21]), ([10, 20], [15]), ([20, 10], [10]), ([10, 20], [])]:
        with pytest.raises(ValueError):
            select_sequence_rows(saved, requested)
    source = QuoteSequenceSource.from_frame(quotes())
    for times in (np.array([100]), np.array([101.5]), np.array([300, 200])):
        with pytest.raises(ValueError):
            source.observe(times)
    frame = quotes()
    frame.loc[0, ["best_bid_qty", "best_ask_qty"]] = 0
    with pytest.raises(ValueError):
        QuoteSequenceSource.from_frame(frame)
