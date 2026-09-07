import zipfile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from lob_forge.boundary_spot_data import read_spot_trades  # noqa: E402


def test_spot_reader_preserves_headerless_first_row_and_maker_side(tmp_path):
    path = tmp_path / "spot.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("spot.csv", "10,100.10000000,2.5,20,21,1685318400100,true,true\n11,100.2,1,22,22,1685318400100,false,true\n")
    frame = read_spot_trades(path, day_start_ms=1685318400000)
    assert len(frame) == 2
    np.testing.assert_array_equal(frame.agg_trade_id, [10,11])
    np.testing.assert_array_equal(frame.is_buyer_maker, [True,False])
    assert frame.price.iloc[0] == 100.1


def test_spot_reader_rejects_clock_units_schema_and_duplicate_aggregate_ids(tmp_path):
    rows = ["10,100,1,20,20,1685318400000000,true,true\n", "10,100,1,20,20,1685318400100,true\n",
            "10,100,1,20,20,1685318400100,unknown,true\n",
            "10,100,1,20,20,1685318400100,true,true\n10,100,1,21,21,1685318400200,false,true\n"]
    for i,row in enumerate(rows):
        path = tmp_path / f"bad_{i}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("spot.csv", row)
        with pytest.raises(ValueError):
            read_spot_trades(path, day_start_ms=1685318400000)
