import hashlib
import json
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_tardis_inputs import TARDIS_VARIANTS, load_tardis_inputs, original_tardis_columns, select_tardis_variant  # noqa: E402


def test_native_loader_preserves_every_row_in_source_interval_and_original_fields(tmp_path):
    day = "2023-06-01"
    clock = np.arange(40) * 1000 + 100000
    current = pd.DataFrame({"decision_time": clock, "bid": 99.0, "ask": 101.0, "bid_qty": 2.0, "ask_qty": 3.0})
    depth = np.zeros((30, 2, 25, 4))
    for level in range(25):
        depth[:, :, level] = [101 + level, 4 + level, 99 - level, 2 + level]
    observed = {"decision_times": clock[:30], "delays_ms": np.array([100, 500]), "depth": depth,
        "publisher_times_ms": clock[:30, None] - np.array([102, 502]),
        "capture_times_ns": (clock[:30, None] - np.array([101, 501])) * 1_000_000,
        "available": np.ones((30, 2), dtype=bool)}
    observed["available"][15] = False
    base, source = [], []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = tmp_path / f"{symbol}.npz"
        np.savez_compressed(path, **observed)
        base.append({"symbol": symbol, "session_date": day, "features_path": f"{symbol}.parquet", "sha256": "original"})
        source.append({"symbol": symbol, "session_date": day, "observation_path": path.name,
            "observation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "original_features_sha256": "original"})
    manifest, native = tmp_path / "base.json", tmp_path / "native.json"
    manifest.write_text(json.dumps({"sessions": base}))
    native.write_text(json.dumps({"sessions": source}))
    keys = [(s, day) for s in ("BTCUSDT", "ETHUSDT")]

    def combined_loader(*args):
        return ({k: pd.DataFrame({"frozen": np.arange(25), "foreign_existing": 1.0, "flow_existing": 2.0}) for k in keys},
            {k: np.arange(25) % 3 - 1 for k in keys}, {k: clock[10:35].copy() for k in keys})

    with patch("lob_forge.boundary_tardis_inputs.load_combined_inputs", side_effect=combined_loader), patch("pandas.read_parquet", side_effect=lambda *a, **kw: current.copy()):
        x, y, times = load_tardis_inputs(tmp_path, manifest, native, tmp_path / "unused", tmp_path / "unused")
    for key in keys:
        assert original_tardis_columns(x[key]) == ["frozen", "foreign_existing", "flow_existing"]
        np.testing.assert_array_equal(x[key].frozen, np.arange(20))
        np.testing.assert_array_equal(y[key], np.arange(20) % 3 - 1)
        np.testing.assert_array_equal(times[key], clock[10:30])
        assert list(select_tardis_variant(x[key], "observations").columns) == ["frozen", "flow_existing"]
        schemas = {}
        for variant in TARDIS_VARIANTS:
            selected = select_tardis_variant(x[key], variant)
            np.testing.assert_array_equal(selected.frozen, np.arange(20))
            assert selected.native_depth_venue_available.iloc[5] == 0
            assert selected.peer_native_depth_venue_available.iloc[5] == 0
            schemas[variant] = list(selected.columns)
        assert schemas["top1_100"] == schemas["top1_500"]
        assert schemas["top25_100"] == schemas["top25_500"]
        assert len(schemas["top1_100"]) == 3 + 27 + 7
        assert len(schemas["top25_100"]) == 3 + 88 + 11
