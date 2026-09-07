import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_confirmation_family import family_interval  # noqa: E402
from lob_forge.boundary_confirmation_stats import cluster_interval  # noqa: E402


def test_two_procedure_interval_uses_wider_tails_on_identical_date_resamples():
    values = np.linspace(-0.05, 0.1, 20)
    for block in [1, 3]:
        ordinary = cluster_interval(values, block_length=block)
        family = family_interval(values, block_length=block)
        assert family["lower_97_5"] <= ordinary["lower_95"]
        assert family["upper_97_5"] >= ordinary["upper_95"]
        assert np.isclose(family["mean"], values.mean())
    exact = family_interval(np.full(20, 0.07))
    assert np.isclose(exact["lower_97_5"], 0.07)
    assert np.isclose(exact["upper_97_5"], 0.07)


def test_family_interval_rejects_changed_date_count_or_invalid_dependence_blocks():
    with pytest.raises(ValueError, match="twenty"):
        family_interval(np.ones(19))
    with pytest.raises(ValueError, match="blocks"):
        family_interval(np.ones(20), block_length=2)
