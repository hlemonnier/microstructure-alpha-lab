import json
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_spot_features import SPOT_FEATURES  # noqa: E402
from lob_forge.boundary_spot_inputs import VARIANTS, load_spot_inputs, original_columns, select_spot_variant  # noqa: E402


def test_spot_join_preserves_base_rows_and_uses_simultaneous_peer_observations(tmp_path):
    symbols, day = ("BTCUSDT", "ETHUSDT"), "2023-06-03"
    clocks = np.array([120000,121000])
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"sessions":[{"symbol":s,"session_date":day,"features_path":s+"_base","sha256":"base"} for s in symbols]}))
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"sessions":[{"symbol":s,"session_date":day,"original_features_path":s+"_base","original_features_sha256":"base","features_path":s+"_extra","sha256":"extra"} for s in symbols]}))
    raw = {s: pd.DataFrame({"decision_time":[119000,120000,121000], **{v+"_"+n: np.arange(3)+10*a+100*i for i,v in enumerate(VARIANTS) for n in SPOT_FEATURES}}) for a,s in enumerate(symbols)}

    def base(*args):
        return ({(s,day):pd.DataFrame({"frozen": [5.,6.],"context_label_neutral_300":[.2,.3]}) for s in symbols},
                {(s,day):np.array([-1,1]) for s in symbols}, {(s,day):clocks.copy() for s in symbols})

    with patch("lob_forge.boundary_spot_inputs.load_context_inputs",side_effect=base), patch("lob_forge.boundary_spot_inputs.sha256_file",return_value="extra"), patch("pandas.read_parquet",side_effect=lambda p:raw[p.name.removesuffix("_extra")].copy()):
        x,y,t = load_spot_inputs(tmp_path,original,extra)
    for s in symbols:
        f=x[s,day]
        assert original_columns(f)==["frozen"]
        np.testing.assert_array_equal(f.frozen,[5,6])
        np.testing.assert_array_equal(y[s,day],[-1,1])
        np.testing.assert_array_equal(t[s,day],clocks)
        for i,v in enumerate(VARIANTS):
            selected=select_spot_variant(f,v)
            assert selected.shape==(2,82)
            assert all(not c.startswith(tuple(v+"_" for v in VARIANTS)) for c in selected.columns)
            other=10 if s=="BTCUSDT" else 0
            np.testing.assert_array_equal(selected.peer_aux_available,clocks/1000-119+other+100*i)
