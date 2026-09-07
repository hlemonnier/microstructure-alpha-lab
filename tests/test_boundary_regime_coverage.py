import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts  # noqa: E402


def test_history_width_preserves_the_validation_day_and_chronological_cutoff():
    cohorts = history_cohorts("2023-06-03")
    assert cohorts["recent4_stride4"]["train_dates"] == ["2023-05-29", "2023-05-30", "2023-05-31", "2023-06-01"]
    assert cohorts["wide14_stride4"]["train_dates"][0] == "2023-05-19"
    assert len(cohorts["wide14_stride4"]["train_dates"]) == 14
    for value in cohorts.values():
        assert value["train_dates"][-1] == "2023-06-01"
        assert "2023-06-02" not in value["train_dates"]
        assert "2023-06-03" not in value["train_dates"]


def test_calendar_sampling_is_fixed_by_clock_and_preserves_prefixes():
    clock = np.arange(100, 200) * 1000
    for stride in (4, 14):
        mask = calendar_stride_mask(clock, 0, stride_seconds=stride)
        np.testing.assert_array_equal(clock[mask], [t for t in clock if t % (stride * 1000) == 0])
        np.testing.assert_array_equal(mask[:30], calendar_stride_mask(clock[:30], 0, stride_seconds=stride))
    with pytest.raises(ValueError):
        calendar_stride_mask([-1000, 1000], 0, stride_seconds=4)
    with pytest.raises(ValueError):
        calendar_stride_mask([0, 86400000], 0, stride_seconds=4)
