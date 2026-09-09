import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("joblib")
pytest.importorskip("sklearn")
previous_path = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from run_boundary_midpoint_conversion_screen import BLENDS, MODEL_SPECS, model_columns
finally:
    sys.path[:] = previous_path


def test_every_midpoint_model_and_blend_has_identical_raw_side_counterpart():
    names = {*MODEL_SPECS, *BLENDS}
    metadata = {"original_columns": ["observed", "auxiliary"], "observation_columns": ["observed"]}
    for name in names:
        if "_midpoint_side_" not in name:
            continue
        raw = name.replace("_midpoint_side_", "_raw_side_")
        assert raw in names
        if name in BLENDS:
            assert tuple(n.replace("_midpoint_side_", "_raw_side_") for n in BLENDS[name]) == BLENDS[raw]
        else:
            actual = model_columns(metadata, name)
            expected = model_columns(metadata, raw)
            assert [c.removeprefix("midpoint_") for c in actual] == expected
            assert {k:v for k,v in MODEL_SPECS[name].items() if k != "conversion_mode"} == {k:v for k,v in MODEL_SPECS[raw].items() if k != "conversion_mode"}


def test_native_and_fx_variants_keep_same_model_settings_and_observed_fx_side():
    metadata = {"original_columns": ["observed", "auxiliary"], "observation_columns": ["observed"]}
    for name, spec in MODEL_SPECS.items():
        columns = model_columns(metadata, name)
        assert any(c.endswith("conversion_last_side") for c in columns)
        assert not any("quote_ETHUSDT_100__spot_" in c for c in columns)
        if spec["variant"] == "btc100":
            fx = name.replace("_btc100_", "_fx100_")
            assert set(model_columns(metadata, fx)) < set(columns)
            assert {k:v for k,v in spec.items() if k != "variant"} == {k:v for k,v in MODEL_SPECS[fx].items() if k != "variant"}
