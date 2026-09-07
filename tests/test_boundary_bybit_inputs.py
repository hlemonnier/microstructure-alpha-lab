import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import hashlib  # noqa: E402
import json  # noqa: E402
from unittest.mock import patch  # noqa: E402

from lob_forge.boundary_bybit_inputs import BYBIT_VARIANTS, load_bybit_inputs, original_bybit_columns, select_bybit_variant  # noqa: E402


def test_depth_loader_preserves_original_universe_and_keeps_variant_schemas_separate(tmp_path):
    n, day = 40, "2023-06-03"
    clock = np.arange(n) * 1000 + 100000
    current = pd.DataFrame({"decision_time": clock, "bid": 99.0, "ask": 101.0, "bid_qty": 2.0, "ask_qty": 3.0})
    depth = np.zeros((n, 2, 25, 4))
    for level in range(25):
        depth[:, :, level] = [101 + level, 4 + level, 99 - level, 2 + level]
    observations = {"decision_times": clock, "delays_ms": np.array([100, 500]), "depth": depth,
        "publisher_times": clock[:, None] - np.array([101, 501]), "available": np.ones((n, 2), dtype=bool)}
    base_rows, depth_rows = [], []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = tmp_path / f"{symbol}.npz"
        np.savez_compressed(path, **observations)
        base_rows.append({"symbol": symbol, "session_date": day, "features_path": f"{symbol}.parquet", "sha256": "original"})
        depth_rows.append({"symbol": symbol, "session_date": day, "observation_path": path.name,
            "observation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "original_features_sha256": "original"})
    base_manifest, depth_manifest = tmp_path / "base.json", tmp_path / "depth.json"
    base_manifest.write_text(json.dumps({"sessions": base_rows}))
    depth_manifest.write_text(json.dumps({"sessions": depth_rows}))
    keys = [(s, day) for s in ("BTCUSDT", "ETHUSDT")]

    def context_loader(*args):
        return ({k: pd.DataFrame({"frozen": np.arange(20), "context_label_example": 1.0}) for k in keys},
            {k: np.arange(20) % 3 - 1 for k in keys}, {k: clock[10:30].copy() for k in keys})

    with patch("lob_forge.boundary_bybit_inputs.load_context_inputs", side_effect=context_loader), patch("pandas.read_parquet", side_effect=lambda *a, **kw: current.copy()):
        x, y, times = load_bybit_inputs(tmp_path, base_manifest, depth_manifest)
    for k in keys:
        assert original_bybit_columns(x[k]) == ["frozen"]
        np.testing.assert_array_equal(x[k].frozen, np.arange(20))
        np.testing.assert_array_equal(y[k], np.arange(20) % 3 - 1)
        np.testing.assert_array_equal(times[k], clock[10:30])
        schemas = {}
        for variant in BYBIT_VARIANTS:
            chosen = select_bybit_variant(x[k], variant)
            assert not any(c.startswith(("bybit_", "peer_bybit_")) for c in chosen.columns)
            np.testing.assert_array_equal(chosen.frozen, np.arange(20))
            schemas[variant] = list(chosen.columns)
        assert schemas["top1_100"] == schemas["top1_500"]
        assert schemas["top25_100"] == schemas["top25_500"]
        assert len(schemas["top25_100"]) > len(schemas["top1_100"])
