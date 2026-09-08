import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_error_geometry import error_geometry  # noqa: E402


def test_hindsight_union_rescue_harm_and_class_balance_match_manual_counts():
    y = np.array([-1, 0, 0, 0, 1, 1])
    d = np.array([[-1, 0, 1, 1, 1, -1], [0, 0, 0, 0, -1, 1]])
    p = .1 + .7 * np.eye(3)[d + 1]
    result = error_geometry(p, d, y, reference=0, ensemble_members=[0, 1])
    np.testing.assert_allclose(result["balanced_accuracy"], [(1 + 1 / 3 + 1 / 2) / 3, (0 + 1 + 1 / 2) / 3])
    assert result["hindsight_any_correct_balanced_accuracy"] == pytest.approx(1)
    assert result["rescue_balanced_mass"][1] == pytest.approx((2 / 3 + 1 / 2) / 3)
    assert result["harm_balanced_mass"][1] == pytest.approx((1 + 1 / 2) / 3)
    assert result["unanimous_wrong_balanced_mass"] == 0
    assert sum(result["correct_expert_count_balanced_mass"]) == pytest.approx(1)
    assert result["hindsight_ceiling_is_a_prediction"] is False


def test_brier_identity_matches_pair_distance_formula_and_identical_experts():
    rng = np.random.default_rng(12)
    p = rng.dirichlet([1, 2, 1], size=(4, 90))
    y = np.tile([-1, 0, 1], 30)
    d = np.argmax(p, axis=2) - 1
    result = error_geometry(p, d, y, reference=0, ensemble_members=[0, 1, 2, 3])
    brier = result["equal_expert_brier"]
    # Independent pair-distance expression: (2*M^2)^-1 sum_ij ||p_i-p_j||^2.
    pair_ambiguity = sum(np.mean(np.sum((p[i] - p[j]) ** 2, axis=1)) for i in range(4) for j in range(4)) / 32
    assert brier["ambiguity"] == pytest.approx(pair_ambiguity, abs=2e-15)
    gram = np.asarray(result["natural_brier_residual_gram"])
    assert np.linalg.eigvalsh(gram).min() > -1e-12
    assert brier["mean_probability_brier"] == pytest.approx(gram.mean(), abs=2e-15)
    assert brier["average_individual_brier"] - brier["ambiguity"] == pytest.approx(brier["mean_probability_brier"], abs=2e-15)
    same_p, same_d = np.repeat(p[:1], 3, axis=0), np.repeat(d[:1], 3, axis=0)
    same = error_geometry(same_p, same_d, y, reference=0, ensemble_members=[0, 1, 2])
    assert same["equal_expert_brier"]["ambiguity"] < 1e-28
    assert same["hindsight_any_correct_balanced_accuracy"] == pytest.approx(same["balanced_accuracy"][0])
    perfect = np.eye(3)[y + 1][None, :, :]
    perfect_result = error_geometry(perfect, y[None, :], y, reference=0, ensemble_members=[0])
    assert perfect_result["natural_brier_residual_cosine"] == [[None]]


def test_geometry_rejects_missing_classes_and_invalid_probability_or_member_contracts():
    y = np.array([-1, 0, 1])
    p = np.array([[[.7, .2, .1], [.1, .8, .1], [.2, .1, .7]]])
    d = y[None, :]
    for bad in (p * 2, np.full_like(p, np.nan)):
        with pytest.raises(ValueError):
            error_geometry(bad, d, y, reference=0, ensemble_members=[0])
    for member in ([1], [0, 0], [], [.5]):
        with pytest.raises(ValueError):
            error_geometry(p, d, y, reference=0, ensemble_members=member)
    with pytest.raises(ValueError):
        error_geometry(p, d, np.zeros(3, dtype=int), reference=0, ensemble_members=[0])
