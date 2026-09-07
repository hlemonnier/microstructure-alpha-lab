import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_aligned_selection import SELECTORS, fit_aligned_selection  # noqa: E402
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural  # noqa: E402
from lob_forge.boundary_pooled import PooledForecaster  # noqa: E402


def test_aligned_checkpoint_selectors_preserve_original_optimization_and_predictions(tmp_path):
    rng = np.random.default_rng(61)
    x = {s: pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd")) for s in SYMBOLS}
    y = {s: np.tile([-1, 0, 1], 20) for s in SYMBOLS}
    vx = {s: pd.DataFrame(rng.normal(size=(24, 4)), columns=list("abcd")) for s in SYMBOLS}
    vy = {s: np.tile([-1, 0, 1], 8) for s in SYMBOLS}
    clocks = {s: np.arange(24) * 1000 + 100000 for s in SYMBOLS}
    initial = {s: np.array([0.15, 0.7, 0.15]) for s in SYMBOLS}
    original, original_history = fit_confirm_neural(x, y, vx, vy)
    models, history = fit_aligned_selection(x, y, vx, vy, clocks, initial)
    assert history["model_fits"] == 1 and history["selection_procedures"] == 3
    assert history["best_epochs"]["registered"] == original_history["best_epoch"]
    np.testing.assert_array_equal([r["mean_training_loss"] for r in history["history"]],
                                  [r["mean_training_loss"] for r in original_history["history"]])
    for asset, s in enumerate(SYMBOLS):
        np.testing.assert_array_equal(models["registered"].predict_proba(vx[s], asset), original.predict_proba(vx[s], asset))
    for selector in SELECTORS:
        models[selector].save(tmp_path / selector)
        restored = PooledForecaster.load(tmp_path / selector)
        for asset, s in enumerate(SYMBOLS):
            np.testing.assert_array_equal(restored.predict_proba(vx[s], asset), models[selector].predict_proba(vx[s], asset))
            np.testing.assert_array_equal(restored.normalizer.quantiles_, original.normalizer.quantiles_)
    changed = {s: np.array([0.4, 0.2, 0.4]) for s in SYMBOLS}
    changed_models, changed_history = fit_aligned_selection(x, y, vx, vy, clocks, changed)
    # Validation decision priors influence selection, never optimization or scaling.
    np.testing.assert_array_equal([r["mean_training_loss"] for r in history["history"]],
                                  [r["mean_training_loss"] for r in changed_history["history"]])
    for asset, s in enumerate(SYMBOLS):
        np.testing.assert_array_equal(changed_models["registered"].predict_proba(vx[s], asset), original.predict_proba(vx[s], asset))
    with pytest.raises(ValueError):
        fit_aligned_selection(x, y, vx, vy, {s: clocks[s][:-1] for s in SYMBOLS}, initial)
