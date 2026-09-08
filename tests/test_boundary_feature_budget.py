import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_feature_budget import HistoricalFeatureBudget, balanced_bin_information, nested_balanced_context  # noqa: E402
from lob_forge.boundary_tabicl import SYMBOLS, select_balanced_context  # noqa: E402


def test_balanced_information_matches_conditional_entropy_and_ignores_class_duplication():
    counts = np.array([[12, 3, 1], [3, 6, 2], [1, 5, 14]])
    codes, labels, assets = [], [], []
    for asset in (0, 1):
        for code in range(3):
            for label in range(3):
                n = int(counts[code, label]) * (asset + 1)
                codes.extend([code] * n)
                labels.extend([label - 1] * n)
                assets.extend([asset] * n)
    codes, labels, assets = (np.array(v) for v in (codes, labels, assets))
    joint = counts / (3 * counts.sum(axis=0))
    mass = joint.sum(axis=1)
    conditional = joint / mass[:, None]
    expected = np.log(3) + np.sum(mass * np.sum(conditional * np.log(conditional), axis=1))
    actual = balanced_bin_information(codes, labels, assets, bins=3)
    assert actual == pytest.approx(expected, abs=1e-15)
    repetitions = np.where(labels == -1, 7, 1)
    assert balanced_bin_information(np.repeat(codes, repetitions), np.repeat(labels, repetitions), np.repeat(assets, repetitions), bins=3) == pytest.approx(actual, abs=1e-15)
    assert balanced_bin_information(labels + 1, labels, assets, bins=3) == pytest.approx(np.log(3), abs=1e-15)


def test_feature_budget_rejects_duplicate_signal_and_preserves_frozen_query_schema(tmp_path):
    rng = np.random.default_rng(40)
    y = np.tile([-1, 0, 1], 200)
    features = {s: pd.DataFrame({"signal": y, "duplicate": y, "noise": rng.normal(size=len(y)), "constant": 0.}, dtype=float) for s in SYMBOLS}
    selected = HistoricalFeatureBudget.fit(features, dict.fromkeys(SYMBOLS, y), budget=2, quantile_bins=4)
    assert selected.selected_columns == ["signal", "noise"]
    assert selected.diversity_selected == 2
    assert "constant" not in selected.random_columns
    assert selected.ranking[0]["balanced_conditional_information_nats"] == pytest.approx(np.log(3))
    selected.save(tmp_path / "selection.json")
    restored = HistoricalFeatureBudget.load(tmp_path / "selection.json")
    frame = features[SYMBOLS[0]].copy()
    pd.testing.assert_frame_equal(restored.transform(frame, "selected"), selected.transform(frame, "selected"), check_exact=True)
    frame.iloc[300:] = 1e20
    pd.testing.assert_frame_equal(restored.transform(frame, "selected").iloc[:300], selected.transform(features[SYMBOLS[0]], "selected").iloc[:300], check_exact=True)
    with pytest.raises(ValueError, match="schema"):
        restored.transform(frame[list(reversed(frame.columns))], "selected")
    with pytest.raises(ValueError, match="nonconstant"):
        HistoricalFeatureBudget.fit(features, dict.fromkeys(SYMBOLS, y), budget=4)


def test_nested_context_preserves_original_rows_and_complete_six_cell_counts():
    y = {s: np.repeat([-1, 0, 1], 40) for s in SYMBOLS}
    original, _, _ = select_balanced_context(y, requested_rows=60, seed=20260908)
    selected, population, sampled = nested_balanced_context(y, small_rows=60, large_rows=120)
    again, _, _ = nested_balanced_context(y, small_rows=60, large_rows=120)
    for symbol in SYMBOLS:
        np.testing.assert_array_equal(selected[60][symbol], original[symbol])
        np.testing.assert_array_equal(selected[120][symbol], again[120][symbol])
        assert np.isin(original[symbol], selected[120][symbol]).all()
        assert len(np.unique(selected[120][symbol])) == 60
        assert [(y[symbol][selected[120][symbol]] == label).sum() for label in (-1, 0, 1)] == [20, 20, 20]
        np.testing.assert_array_equal(population[symbol], sampled[symbol])
    with pytest.raises(ValueError, match="larger context budget"):
        nested_balanced_context(y, small_rows=60, large_rows=300)
    with pytest.raises(ValueError, match="smaller context budget"):
        nested_balanced_context({s: y[s][:81] for s in SYMBOLS}, small_rows=60, large_rows=120)
