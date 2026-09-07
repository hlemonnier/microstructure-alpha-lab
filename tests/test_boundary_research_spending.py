import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_research_spending import research_round_interval  # noqa: E402


def test_repeated_research_reserves_nominal_error_budget_and_widens_later_intervals():
    x = np.linspace(-0.05, 0.1, 20)
    first = research_round_interval(x, round_number=1, procedures=2)
    second = research_round_interval(x, round_number=2, procedures=2)
    assert first["round_alpha"] == 0.025
    assert first["marginal_interval_coverage"] == 0.9875
    assert second["lower"] <= first["lower"]
    assert second["upper"] >= first["upper"]
    assert sum(0.05 / 2**j for j in range(1, 50)) < 0.05
    with pytest.raises(ValueError, match="Positive"):
        research_round_interval(x, round_number=0, procedures=2)
    with pytest.raises(ValueError, match="resamples"):
        research_round_interval(x, round_number=20, procedures=2)
