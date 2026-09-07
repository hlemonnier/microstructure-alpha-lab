import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_adaptive_priors import label_priors  # noqa: E402
from lob_forge.boundary_long_context import (  # noqa: E402
    HALF_LIVES,
    observation_context,
    released_outcome_context,
)


def test_activity_context_uses_only_observations_and_preserves_prefixes():
    n = 40
    frame = pd.DataFrame({"decision_time": np.arange(n) * 1000, "bid": 100.0, "ask": 100.1,
        "bid_qty": 2.0, "ask_qty": 3.0, "mid_return_1": np.sin(np.arange(n)) / 1000,
        "quote_updates_in_bucket": np.arange(n) + 1, "trade_count": np.arange(n) % 3,
        "trade_notional": np.arange(n) * 10, "event_price_change_count": np.arange(n) % 2,
        "quote_ofi_normalized": np.cos(np.arange(n)), "top_imbalance": -.2,
        "label": np.ones(n)})
    original = frame.copy(deep=True)
    result = observation_context(frame)
    pd.testing.assert_frame_equal(result.iloc[:20], observation_context(frame.iloc[:20]), check_exact=True)
    frame["label"] = -1
    pd.testing.assert_frame_equal(result, observation_context(frame), check_exact=True)
    pd.testing.assert_frame_equal(original.drop(columns="label"), frame.drop(columns="label"))
    assert result.shape == (n, 46)
    assert np.isfinite(result.to_numpy()).all()
    assert result.context_log_depth_notional_300.iloc[-1] == pytest.approx(np.log1p(500.3))
    with pytest.raises(ValueError):
        observation_context(frame.drop(index=10))


def test_released_context_matches_explicit_decayed_labels_and_strict_release():
    origins = np.array([0, 1000, 2000, 3000])
    release = np.array([5100, 6100, 15100, 15100])
    labels = np.array([-1, 0, 1, -1])
    queries = np.array([5000, 5100, 5101, 6101, 15100, 15101, 80000000])
    result = released_outcome_context(labels, origins, release, queries)
    for half in HALF_LIVES:
        columns = [f"context_label_{k}_{half}" for k in ("down", "neutral", "up")]
        expected = label_priors(labels, origins, release, queries - 1, half_life_seconds=half)
        # Moving the query 1ms earlier enforces the strict event boundary. A
        # common decay factor cancels from the normalized counts.
        np.testing.assert_allclose(result[columns], expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(result[columns].sum(axis=1), 1)
    assert result.context_label_down_300.iloc[0] == result.context_label_down_300.iloc[1]
    assert result.context_label_down_300.iloc[2] > result.context_label_down_300.iloc[1]
    prefix = released_outcome_context(labels[:2], origins[:2], release[:2], queries[:5])
    pd.testing.assert_frame_equal(result.iloc[:5], prefix, check_exact=True)


def test_released_context_rejects_invalid_history_and_handles_empty_history():
    result = released_outcome_context([], [], [], [1000, 2000])
    for half in HALF_LIVES:
        np.testing.assert_allclose(result[f"context_label_neutral_{half}"], [1 / 3, 1 / 3], rtol=0, atol=1e-16)
    for labels, origins, release, query in [([1], [1000], [1000], [2000]), ([2], [1000], [2000], [3000]),
                                           ([1], [0], [5000], [86400001])]:
        with pytest.raises(ValueError):
            released_outcome_context(labels, origins, release, query)


def test_context_loader_keeps_base_rows_and_does_not_read_unreleased_outcomes(tmp_path):
    import json
    from unittest.mock import patch

    from lob_forge.boundary_long_context_inputs import context_columns, load_context_inputs

    n = 200
    clock = np.arange(n) * 1000
    raw = pd.DataFrame({"decision_time": clock, "bid": 100.0, "ask": 100.1,
        "bid_qty": 2.0, "ask_qty": 3.0, "mid_return_1": 0.0,
        "quote_updates_in_bucket": 2, "trade_count": 1, "trade_notional": 100.0,
        "event_price_change_count": 0, "quote_ofi_normalized": 0.1, "top_imbalance": -.2,
        "label": np.arange(n) % 3 - 1, "future_event_time": clock + 5100})
    keys = [(s, "2023-06-03") for s in ("BTCUSDT", "ETHUSDT")]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"sessions": [{"symbol": s, "session_date": d, "features_path": s + ".parquet"} for s, d in keys]}))

    def base_loader(*args):
        return ({k: pd.DataFrame({"frozen_feature": np.arange(40)}) for k in keys},
                {k: raw.label.to_numpy()[120:160].copy() for k in keys}, {k: clock[120:160].copy() for k in keys})

    with patch("lob_forge.boundary_long_context_inputs.load_clock_inputs", side_effect=base_loader), patch("pandas.read_parquet", side_effect=lambda p: raw.copy()):
        x, y, times = load_context_inputs(tmp_path, path)
        raw.loc[155:, "label"] = 1  # No changed outcome is available by t=159s.
        changed_x, _, _ = load_context_inputs(tmp_path, path)
    for k in keys:
        np.testing.assert_array_equal(x[k].frozen_feature, np.arange(40))
        np.testing.assert_array_equal(times[k], clock[120:160])
        assert len(y[k]) == 40
        pd.testing.assert_frame_equal(x[k], changed_x[k], check_exact=True)
        assert len(context_columns(x[k], "observations")) == 53
        assert len(context_columns(x[k], "released")) == 68
