import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from lob_forge.binance_vision import sha256_file  # noqa: E402
from lob_forge.boundary_event_inputs import load_event_inputs, utc_ms  # noqa: E402
from lob_forge.boundary_events import EVENT_FEATURES  # noqa: E402
from lob_forge.boundary_forecasts import OBSERVED_FIELDS  # noqa: E402


def test_event_inputs_reset_at_gap_and_buffer_source_end_without_label_selection(tmp_path):
    day = "2023-05-20"
    times = utc_ms(day, "12:00:00") + np.r_[np.arange(200), np.arange(400, 700)] * 1000
    frame = pd.DataFrame({name: np.zeros(len(times)) for name in (*OBSERVED_FIELDS, *EVENT_FEATURES)})
    frame["decision_time"] = times
    frame["bid"], frame["ask"], frame["mid"] = 100.0, 100.1, 100.05
    frame["bid_qty"], frame["ask_qty"], frame["quote_age_ms"] = 1.0, 2.0, 10
    frame["label"] = np.resize([-1.0, 0.0, 1.0], len(frame))
    frame.loc[frame.index[-5:], "label"] = np.nan
    sessions = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        path = tmp_path / f"{symbol}.parquet"
        frame.to_parquet(path, index=False)
        sessions.append(
            {"symbol": symbol, "session_date": day, "features_path": path.name, "sha256": sha256_file(path)}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sessions": sessions}))
    features, labels, clocks = load_event_inputs(tmp_path, manifest)
    expected = np.r_[times[120:200], times[320:439]]
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        np.testing.assert_array_equal(clocks[symbol, day], expected)
        assert len(features[symbol, day]) == len(expected)
        assert np.isfinite(labels[symbol, day]).all()
        assert (features[symbol, day]["peer_available"] == 1).all()
    # An unresolved target inside the admitted coverage is an error, never silently dropped.
    frame.loc[150, "label"] = np.nan
    frame.to_parquet(tmp_path / "BTCUSDT.parquet", index=False)
    with pytest.raises(ValueError, match="lacks a future label"):
        load_event_inputs(tmp_path, manifest, verify_hashes=False)
