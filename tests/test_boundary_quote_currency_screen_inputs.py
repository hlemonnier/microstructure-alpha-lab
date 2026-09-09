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
    from run_boundary_quote_currency_screen import BLENDS, MODEL_SPECS, model_columns
finally:
    sys.path[:] = previous_path


def test_quote_model_controls_retain_both_rates_and_exclude_unregistered_delays():
    metadata = {"original_columns": ["observation", "existing_spot"], "observation_columns": ["observation"]}
    for name, spec in MODEL_SPECS.items():
        columns = model_columns(metadata, name)
        assert not any("_500__" in c for c in columns)
        assert "quote_BTCUSDT_100__conversion_available" in columns
        assert "quote_ETHUSDT_100__conversion_available" in columns
        assert ("existing_spot" in columns) == (spec["representation"] == "combined")
        if spec["variant"] == "fx100":
            assert not any("__spot_" in c or "__converted_basis_" in c for c in columns)
        elif spec["variant"] == "btc100":
            assert "quote_BTCUSDT_100__spot_last_basis_bps" in columns
            assert "quote_ETHUSDT_100__spot_last_basis_bps" not in columns


def test_every_native_model_and_blend_has_same_family_conversion_control():
    names = {*MODEL_SPECS, *BLENDS}
    for name in names:
        tokens = name.split("_")
        if tokens[1] in ("btc100", "both100"):
            fx = "_".join([tokens[0], "fx100", *tokens[2:]])
            assert fx in names
            if name in BLENDS:
                native_components = BLENDS[name]
                control_components = BLENDS[fx]
                for left, right in zip(native_components, control_components, strict=True):
                    assert left.replace("_btc100_", "_fx100_").replace("_both100_", "_fx100_") == right
            else:
                assert {k:v for k,v in MODEL_SPECS[name].items() if k != "variant"} == {k:v for k,v in MODEL_SPECS[fx].items() if k != "variant"}
