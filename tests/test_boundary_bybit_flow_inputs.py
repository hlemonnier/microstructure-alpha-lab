import hashlib
import json
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_bybit_flow import FLOW_FIELDS  # noqa: E402
from lob_forge.boundary_bybit_flow_inputs import FLOW_VARIANTS, load_flow_inputs, original_flow_columns, select_flow_variant  # noqa: E402


def test_native_flow_loader_preserves_targets_rows_and_simultaneous_peer_features(tmp_path):
    day = "2023-06-03"
    clock = np.arange(8, dtype=np.int64) * 1000 + 100000
    keys = [(s, day) for s in ("BTCUSDT", "ETHUSDT")]
    frames = {k: pd.DataFrame({"original": np.arange(4), "foreign_observation": 2.0, "aux_observation": 3.0}) for k in keys}
    labels = {k: np.array([-1, 0, 1, -1]) for k in keys}
    times = {k: clock[2:6].copy() for k in keys}
    originals = {k: v.copy() for k, v in frames.items()}
    records, base = [], []
    for index, (symbol, _) in enumerate(keys):
        row = np.array([3, 2, 1, 1] * 3 + [index + 1, 6, 0.2, 0.6, 10, 20, 5], dtype=float)
        source = {"decision_times": clock, "delays_ms": np.array([100, 500]), "windows_ms": np.array([1000, 5000, 15000]),
            "flow_totals": np.broadcast_to(row, (8, 2, 3, 19)).copy(), "window_available": np.ones((8, 2, 3), dtype=bool),
            "source_publisher_times": clock[:, None] - np.array([101, 501]), "top_depth": np.full((8, 2), 20.0)}
        path = tmp_path / f"{symbol}.npz"
        np.savez_compressed(path, **source)
        records.append({"symbol": symbol, "session_date": day, "original_features_sha256": "original",
                        "observation_path": path.name, "observation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fields": list(FLOW_FIELDS)})
        base.append({"symbol": symbol, "session_date": day, "sha256": "original"})
    manifest, flow_manifest = tmp_path / "base.json", tmp_path / "flow.json"
    manifest.write_text(json.dumps({"sessions": base}))
    flow_manifest.write_text(json.dumps({"sessions": records}))
    with patch("lob_forge.boundary_bybit_flow_inputs.load_combined_inputs", return_value=(frames, labels, times)):
        x, y, t = load_flow_inputs(tmp_path, manifest, None, None, flow_manifest)
    for key in keys:
        other = keys[1] if key == keys[0] else keys[0]
        pd.testing.assert_frame_equal(x[key][original_flow_columns(x[key])], originals[key], check_exact=True)
        np.testing.assert_array_equal(y[key], labels[key])
        np.testing.assert_array_equal(t[key], times[key])
        schemas = {}
        for variant, (_, deep) in FLOW_VARIANTS.items():
            selected = select_flow_variant(x[key], variant)
            peer = select_flow_variant(x[other], variant)
            assert len(selected.columns) == 3 + (103 if deep else 29)
            np.testing.assert_array_equal(selected.peer_flow_native_bbo_pressure_1s, peer.flow_native_bbo_pressure_1s)
            schemas[variant] = list(selected.columns)
        assert schemas["bbo100"] == schemas["bbo500"]
        assert schemas["deep100"] == schemas["deep500"]
    records[0]["fields"] = list(reversed(FLOW_FIELDS))
    flow_manifest.write_text(json.dumps({"sessions": records}))
    with patch("lob_forge.boundary_bybit_flow_inputs.load_combined_inputs", return_value=(originals, labels, times)):
        with pytest.raises(ValueError, match="field order"):
            load_flow_inputs(tmp_path, manifest, None, None, flow_manifest)
