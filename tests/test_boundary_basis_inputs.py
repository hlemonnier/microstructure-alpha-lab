from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from lob_forge.boundary_basis_inputs import load_basis_inputs, original_basis_columns  # noqa: E402


def test_basis_loader_preserves_original_universe_and_joins_only_simultaneous_peer_state(tmp_path):
    clock = np.arange(6, dtype=np.int64) * 1000
    frame = pd.DataFrame({"foreign_venue_mid_basis_bps": [1, 2, 1, 3, 2, 4],
                          "foreign_venue_available": np.ones(6), "aux_last_basis_bps": np.zeros(6),
                          "aux_available": np.ones(6), "original_signal": np.arange(6)})
    keys = [(s, "2023-06-03") for s in ("BTCUSDT", "ETHUSDT")]
    frames = {k: frame.copy() for k in keys}
    frames[keys[1]]["foreign_venue_mid_basis_bps"] *= 2
    original = {k: v.copy(deep=True) for k, v in frames.items()}
    labels = {k: np.array([-1, 0, 1, 1, 0, -1]) for k in keys}
    times = {k: clock.copy() for k in keys}
    with patch("lob_forge.boundary_basis_inputs.load_combined_inputs", return_value=(frames, labels, times)):
        x, y, t = load_basis_inputs(tmp_path, None, None, None)
    for key in keys:
        other = keys[1] if key == keys[0] else keys[0]
        pd.testing.assert_frame_equal(x[key][original_basis_columns(x[key])], original[key], check_exact=True)
        np.testing.assert_array_equal(y[key], labels[key])
        np.testing.assert_array_equal(t[key], clock)
        assert len(x[key].columns) == len(frame.columns) + 38
        np.testing.assert_array_equal(x[key].peer_basis_state_bybit_innovation_z_60s,
                                      x[other].basis_state_bybit_innovation_z_60s)
