from pathlib import Path

from lob_forge.fi2010 import fi2010_to_topn_tensor, load_fi2010_matrix, load_fi2010_named_columns


def test_load_fi2010_matrix_with_label(tmp_path: Path) -> None:
    path = tmp_path / "fi2010.csv"
    values = [str(value) for value in range(1, 41)]
    path.write_text(",".join(values + ["1"]) + "\n")

    dataset = load_fi2010_matrix(path, levels=10)
    tensor = fi2010_to_topn_tensor(dataset)

    assert dataset.rows == 1
    assert dataset.samples[0].ask_prices[0] == 1.0
    assert dataset.samples[0].bid_sizes[-1] == 40.0
    assert dataset.samples[0].label == 1
    assert tensor[0][0] == [1.0, 2.0, 3.0, 4.0]


def test_load_fi2010_named_columns(tmp_path: Path) -> None:
    path = tmp_path / "named.csv"
    columns = []
    values = []
    for level in range(1, 3):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
        values.extend([str(level), str(level + 10), str(level + 20), str(level + 30)])
    columns.append("label")
    values.append("-1")
    path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n")

    dataset = load_fi2010_named_columns(path, levels=2, label_column="label")

    assert dataset.rows == 1
    assert dataset.samples[0].ask_sizes == [11.0, 12.0]
    assert dataset.samples[0].bid_prices == [21.0, 22.0]
    assert dataset.samples[0].label == -1
