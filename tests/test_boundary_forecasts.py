import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from lob_forge.boundary_forecasts import (  # noqa: E402
    OBSERVED_FIELDS,
    feature_frame,
    path_areas,
    fit_classifier,
    classification_metrics,
)


def _frame(n=80):
    frame = pd.DataFrame({field: np.zeros(n) for field in OBSERVED_FIELDS})
    frame["decision_time"] = np.arange(n) * 1000 + 100000
    frame["bid"] = 100 + np.sin(np.arange(n) / 8)
    frame["ask"] = frame["bid"] + 0.02
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2
    frame["bid_qty"] = 2
    frame["ask_qty"] = 3
    frame["quote_age_ms"] = 100
    frame["mid_return_1"] = frame["mid"].pct_change().fillna(0)
    frame["quote_ofi_normalized"] = np.cos(np.arange(n) / 5)
    frame["label"] = np.tile([-1, 0, 1], n // 3 + 1)[:n]
    return frame


def test_signature_area_preserves_order_and_forgets_old_path():
    xy = np.array([[1.0, 0.0], [0.0, 1.0]])
    yx = xy[::-1]
    assert path_areas(xy, 2)[-1, 0] == 0.5
    assert path_areas(yx, 2)[-1, 0] == -0.5
    whole = np.vstack([[100.0, -50.0], xy])
    assert path_areas(whole, 2)[-1, 0] == 0.5
    assert np.all(path_areas(whole, 1) == 0)


def test_boundary_features_ignore_targets_and_future_rows():
    own = _frame()
    peer = _frame()
    before = feature_frame(own, peer, "signature_cross")
    changed = own.copy()
    changed["label"] = 1e99
    changed["future_mid"] = -1e99
    changed["entry_ask"] = float("nan")
    changed_peer = peer.copy()
    changed_peer.loc[60:, ["bid", "ask", "mid"]] *= 2
    changed_peer.loc[60:, "quote_ofi_normalized"] = 1e20
    after = feature_frame(changed, changed_peer, "signature_cross")
    assert np.allclose(before.iloc[:60], after.iloc[:60], rtol=0, atol=0)
    assert not any(c.startswith(("label", "future_", "entry_")) for c in before.columns)
    prefix = feature_frame(own.iloc[:40], peer.iloc[:40], "signature_cross")
    assert np.allclose(prefix, before.iloc[:40], rtol=0, atol=0)


def test_boundary_features_reject_gaps_and_disable_stale_peer():
    own = _frame()
    peer = _frame()
    peer["quote_age_ms"] = 2000
    result = feature_frame(own, peer, "temporal_cross")
    assert not result["peer_available"].any()
    assert not result["peer_mid_return_1"].any()
    with pytest.raises(ValueError, match="contiguous"):
        feature_frame(own.drop(index=10), peer, "temporal_cross")


def test_classifier_training_state_and_probabilities_are_frozen():
    own = _frame(120)
    x = feature_frame(own, own, "base")
    model = fit_classifier("logistic_0.1", x, own["label"].to_numpy())
    bounds = model.lower.copy(), model.upper.copy(), model.priors.copy()
    p = model.predict_proba(x)
    metrics = classification_metrics(model, p, own["label"].to_numpy())
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert np.allclose(p.sum(axis=1), 1)
    model.predict_proba(x * 1e20)
    assert all(np.array_equal(a, b) for a, b in zip(bounds, [model.lower, model.upper, model.priors]))
    with pytest.raises(ValueError, match="schema mismatch"):
        model.predict_proba(x.iloc[:, :-1])
