import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from lob_forge.boundary_forecasts import BASE_FEATURES  # noqa: E402
from lob_forge.boundary_structural import fit_hurdle, mirror_features, reflection_signs  # noqa: E402


def test_reflection_is_involution_and_path_area_parity_is_correct():
    columns = ["top_imbalance", "log_activity", "flow_x_imbalance", "flow_ewm_15", "path_area_5_0", "path_area_5_3"]
    assert np.array_equal(reflection_signs(columns), [-1, 1, 1, -1, 1, -1])
    frame = pd.DataFrame(np.arange(18).reshape(3, 6), columns=columns)
    assert np.array_equal(mirror_features(mirror_features(frame)), frame)


def test_hurdle_probabilities_coherent_and_balanced_decisions_reflect():
    n = 1200
    x = pd.DataFrame({key: np.zeros(n) for key in BASE_FEATURES})
    x["top_imbalance"] = np.linspace(-1, 1, n)
    x["log_activity"] = np.tile([0.0, 1.0], n // 2)
    labels = np.where(x["log_activity"] == 0, 0, np.where(x["top_imbalance"] > 0, 1, -1))
    model = fit_hurdle("hurdle_logistic_symmetric", x, labels)
    p = model.predict_proba(x)
    reflected = model.predict_proba(mirror_features(x))
    assert np.allclose(p.sum(axis=1), 1)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.array_equal(np.argmax(p / model.priors, axis=1) - 1, -(np.argmax(reflected / model.priors, axis=1) - 1))
    assert ((np.argmax(p / model.priors, axis=1) - 1) == labels).mean() > 0.95
