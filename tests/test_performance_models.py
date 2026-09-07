"""Learner-contract tests; constructed payoffs are not empirical alpha evidence."""
import copy
import json
import math

import pytest

from lob_forge.baselines import DEFAULT_FEATURES
from lob_forge.edge_model import fit_edge_model, predict_gross_edges_bps
from lob_forge.performance_models import (
    PERFORMANCE_MODEL_NAMES,
    STATE_FEATURES,
    fit_performance_model,
    gross_edge_targets_bps,
    performance_feature_rows,
)


def _row(x: float, gross_bps: float = 1.0) -> dict[str, str]:
    row = {feature: "0" for feature in DEFAULT_FEATURES}
    row.update({
        "top_imbalance": repr(x), "quote_ofi_normalized": repr(x),
        "realized_volatility_5": ".0001", "bid": "100", "ask": "100",
        "bid_qty": repr(1 + x), "ask_qty": repr(1 - x),
        "quote_updates_in_bucket": "10", "quote_age_ms": "20",
        "trade_count": "3", "trade_qty": "2", "trade_notional": "200",
        "entry_bid": "100", "entry_ask": "100",
        "future_bid": repr(100 * (1 + gross_bps / 10_000)),
        "future_ask": repr(100 * (1 + gross_bps / 10_000)),
    })
    return row


def test_performance_features_never_read_entry_future_or_labels() -> None:
    original = _row(.25)
    changed = {**original, "entry_bid": "nan", "entry_ask": "inf",
               "future_bid": "-1", "future_ask": "garbage", "label": "-999",
               "entry_bid_qty": "1e99", "future_lag_ms": "999999999"}
    before = copy.deepcopy(original)
    features = performance_feature_rows([original])[0]
    assert features == performance_feature_rows([changed])[0]
    assert set(features) == set(DEFAULT_FEATURES) | set(STATE_FEATURES)
    assert original == before
    assert math.isclose(float(features["state_ofi_x_imbalance"]), .0625)
    assert math.isclose(float(features["state_log_top_notional"]), math.log1p(200))
    assert all(not key.startswith(("entry_", "future_")) for key in features)


def test_performance_targets_use_executable_entry_quotes_exactly() -> None:
    row = {**_row(.1), "bid": "10", "ask": "11", "entry_bid": "99", "entry_ask": "101",
           "future_bid": "102", "future_ask": "104"}
    long, short = gross_edge_targets_bps([row])[0]
    assert math.isclose(long, 10_000 / 101)
    assert math.isclose(short, -50_000 / 99)
    with pytest.raises(ValueError, match="entry_ask"):
        gross_edge_targets_bps([{key: value for key, value in row.items() if key != "entry_ask"}])
    with pytest.raises(ValueError, match="uncrossed"):
        gross_edge_targets_bps([{**row, "future_bid": "200"}])


def test_performance_ridge_baseline_matches_existing_model_and_freezes_scaler() -> None:
    train = [_row(-.9 + i * .1, -.9 + i * .1) for i in range(19)]
    held = [_row(.333, 99), _row(-.222, -99)]
    baseline = fit_edge_model(train, DEFAULT_FEATURES, l2=1)
    model = fit_performance_model("ridge_default", train)
    expected = [predict_gross_edges_bps(baseline, row) for row in held]
    actual = model.predict_gross_edges(held)
    assert actual == expected
    metadata_before = json.dumps(model.metadata(), sort_keys=True, allow_nan=False)
    model.predict_gross_edges([_row(1e8)])
    assert json.dumps(model.metadata(), sort_keys=True, allow_nan=False) == metadata_before
    unlabeled = [{key: value for key, value in row.items() if not key.startswith(("entry_", "future_"))}
                 for row in held]
    assert model.predict_gross_edges(unlabeled) == actual
    assert model.predict_gross_edges([]) == []


def test_performance_models_reject_nonfinite_and_missing_predictors() -> None:
    for name in PERFORMANCE_MODEL_NAMES:
        with pytest.raises(ValueError, match="finite"):
            fit_performance_model(name, [{**_row(.1), "top_imbalance": "nan"}])
    with pytest.raises(ValueError, match="unknown performance model"):
        fit_performance_model("unregistered_sweep", [_row(.1)])
    with pytest.raises(ValueError, match="empty"):
        fit_performance_model("ridge_state", [])
    with pytest.raises(ValueError, match="quote_age_ms"):
        performance_feature_rows([{key: value for key, value in _row(.1).items() if key != "quote_age_ms"}])
    with pytest.raises(ValueError, match="nonnegative"):
        performance_feature_rows([{**_row(.1), "trade_qty": "-1"}])


def test_performance_enriched_ridge_recovers_predeclared_interaction() -> None:
    train = [_row(x, 5*x*x) for x in [-.95 + i*.05 for i in range(39)]]
    test = [_row(x, 5*x*x) for x in [-.83, -.27, .12, .67]]
    default = fit_performance_model("ridge_default", train).predict_gross_edges(test)
    model = fit_performance_model("ridge_state", train)
    state = model.predict_gross_edges(test)
    truth = gross_edge_targets_bps(test)
    default_error = sum((a[0]-t[0])**2 for a, t in zip(default, truth))
    state_error = sum((a[0]-t[0])**2 for a, t in zip(state, truth))
    assert state_error < .02 * default_error
    assert model.metadata()["standardizer"]["fitted_on"] == "training rows only"


def test_performance_hgb_is_deterministic_and_learns_a_constructed_threshold() -> None:
    pytest.importorskip("sklearn")
    train_x = [-.999 + i*1.998/1599 for i in range(1600)]
    test_x = [-.997 + i*1.994/398 for i in range(399)]
    train = [_row(x, 4 if x > .4 else -4) for x in train_x]
    test = [_row(x, 4 if x > .4 else -4) for x in test_x]
    model = fit_performance_model("hgb_state", train)
    predictions = model.predict_gross_edges(test)
    repeat = fit_performance_model("hgb_state", train).predict_gross_edges(test)
    assert predictions == repeat
    truth = gross_edge_targets_bps(test)
    assert sum((a[0]-t[0])**2 for a, t in zip(predictions, truth)) / len(truth) < .3
    metadata = model.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata["config"]["early_stopping"] is False
    assert metadata["config"]["max_iter"] == 150
    assert metadata["config"]["max_leaf_nodes"] == 7
    assert metadata["config"]["min_samples_leaf"] == 100
    assert metadata["standardizer"] is None
