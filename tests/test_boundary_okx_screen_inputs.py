import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("joblib")
pytest.importorskip("sklearn")
previous_path = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from run_boundary_okx_counted_screen import GROUPS, model_columns, wide_counted_inputs
finally:
    sys.path[:] = previous_path

from lob_forge.boundary_okx_inputs import PEER_COUNTS, PEER_QUANTITY  # noqa: E402


def _wide():
    clock = np.array([1000, 2000, 3000], dtype=np.int64)
    original = pd.DataFrame({"observation": [1., 2., 3.], "foreign_venue": [4., 5., 6.]})
    own, peer = {}, {}
    for i, (depth, delay) in enumerate(GROUPS, 1):
        group = f"top{depth}_{delay}"
        own[group] = pd.DataFrame({c: np.arange(3, dtype=float) + 10*i for c in (*PEER_QUANTITY, *PEER_COUNTS)}, index=clock)
        peer[group] = own[group] + 100
    frame, variants = wide_counted_inputs(original, clock, own, peer)
    metadata = {"variant_columns": variants, "original_columns": list(original.columns), "observation_columns": ["observation"]}
    return frame, metadata, original


def test_shared_cache_keeps_delay_depth_and_peer_states_distinct_without_losing_rows():
    frame, metadata, original = _wide()
    pd.testing.assert_frame_equal(frame[list(original.columns)], original, check_exact=True)
    np.testing.assert_array_equal(frame.okx_top25_100__count_1_log_orders, [10., 11., 12.])
    np.testing.assert_array_equal(frame.okx_top100_100__count_1_log_orders, [20., 21., 22.])
    np.testing.assert_array_equal(frame.okx_top100_500__count_1_log_orders, [30., 31., 32.])
    np.testing.assert_array_equal(frame.peer_okx_top100_500__count_1_log_orders, [130., 131., 132.])
    for variant, columns in metadata["variant_columns"].items():
        assert len(frame[columns]) == 3 and columns[:2] == list(original.columns)
        if variant.startswith("q"):
            assert not any("__count_" in c for c in columns)


def test_observation_representation_removes_only_existing_auxiliary_venue_features():
    frame, metadata, _ = _wide()
    combined = model_columns(metadata, "n100_500_neural")
    observed = model_columns(metadata, "observations_n100_500_neural")
    assert observed == [c for c in combined if c != "foreign_venue"]
    assert "okx_top100_500__count_1_log_orders" in observed
    assert "peer_okx_top100_500__count_1_log_orders" in observed
    control = model_columns(metadata, "observations_q100_500_hgb")
    pd.testing.assert_frame_equal(frame[control], frame[[c for c in observed if "__count_" not in c]], check_exact=True)
