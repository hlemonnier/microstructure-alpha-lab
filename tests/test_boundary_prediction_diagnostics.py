import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from lob_forge.boundary_prediction_diagnostics import forecast_diagnostics  # noqa: E402


def test_likelihood_decomposition_and_distinct_movement_direction_information():
    y = np.array([-1, 0, 1, -1, 0, 1])
    p = np.array([[0.45, 0.1, 0.45], [0.05, 0.9, 0.05], [0.45, 0.1, 0.45]] * 2)
    result = forecast_diagnostics(p, y)
    assert result["movement_roc_auc"] == 1
    assert result["conditional_direction_roc_auc"] == 0.5
    assert np.isclose(result["conditional_direction_log_loss"], np.log(2))
    assert np.isclose(
        result["three_class_log_loss_with_shared_smoothing"],
        result["movement_binary_log_loss"] + result["direction_log_loss_contribution_per_decision"],
    )


def test_diagnostic_zero_probabilities_single_class_and_label_integrity():
    result = forecast_diagnostics(np.eye(3)[[1, 1]], np.array([0, 0]))
    assert result["conditional_direction_log_loss"] is None
    assert result["conditional_direction_roc_auc"] is None
    assert result["movement_roc_auc"] is None
    assert np.isfinite(result["three_class_log_loss_with_shared_smoothing"])
    with pytest.raises(ValueError, match="unmodified"):
        forecast_diagnostics(np.array([[0.2, 0.6, 0.2]]), np.array([0.2]))
