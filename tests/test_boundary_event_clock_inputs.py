import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from lob_forge.binance_vision import sha256_file  # noqa: E402
from lob_forge.boundary_event_clock import EVENT_CLOCK_FEATURES, PEER_CLOCK_FEATURES  # noqa: E402
from lob_forge.boundary_event_clock_inputs import load_clock_inputs  # noqa: E402
from lob_forge.boundary_event_inputs import load_event_inputs, utc_ms  # noqa: E402
from lob_forge.boundary_events import EVENT_FEATURES  # noqa: E402
from lob_forge.boundary_forecasts import OBSERVED_FIELDS  # noqa: E402


def test_clock_loader_preserves_rows_labels_and_original_features(tmp_path):
    day = "2023-05-20"
    times = utc_ms(day) + np.arange(400) * 1000
    frame = pd.DataFrame({name: np.zeros(len(times)) for name in (*OBSERVED_FIELDS, *EVENT_FEATURES)})
    frame["decision_time"] = times
    frame["bid"], frame["ask"], frame["mid"] = 100.0, 100.1, 100.05
    frame["bid_qty"], frame["ask_qty"], frame["quote_age_ms"] = 1.0, 2.0, 10
    frame["label"] = np.resize([-1.0, 0.0, 1.0], len(frame))
    sessions = []
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        additions = pd.DataFrame({name: np.arange(len(frame)) + 1000 * asset for name in EVENT_CLOCK_FEATURES})
        path = tmp_path / f"{symbol}.parquet"
        pd.concat([frame, additions], axis=1).to_parquet(path, index=False)
        sessions.append({"symbol": symbol, "session_date": day, "features_path": path.name, "sha256": sha256_file(path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sessions": sessions}))
    base, base_y, base_t = load_event_inputs(tmp_path, manifest)
    extended, labels, clocks = load_clock_inputs(tmp_path, manifest)
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        key = symbol, day
        np.testing.assert_array_equal(clocks[key], base_t[key])
        np.testing.assert_array_equal(labels[key], base_y[key])
        pd.testing.assert_frame_equal(extended[key][base[key].columns], base[key])
        locations = (clocks[key] - utc_ms(day)) // 1000
        for name in EVENT_CLOCK_FEATURES:
            np.testing.assert_array_equal(extended[key][name], locations + 1000 * asset)
        for name in PEER_CLOCK_FEATURES:
            np.testing.assert_array_equal(extended[key]["peer_" + name], locations + 1000 * (1 - asset))
        assert "label" not in extended[key].columns
