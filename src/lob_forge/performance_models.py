"""A small, predeclared expected-payoff family for performance research.

These learners predict two executable gross returns, in basis points. They do
not choose an execution policy or establish profitability. Every predictor is
derived from the completed decision snapshot; executable entry/future quotes
are used only as supervised targets. Optional research libraries load lazily.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from lob_forge.baselines import DEFAULT_FEATURES
from lob_forge.edge_model import EdgeModel, fit_edge_model, predict_gross_edges_bps


PERFORMANCE_MODEL_NAMES = ("ridge_default", "ridge_state", "hgb_state")
PERFORMANCE_MODEL_VERSION = "causal_expected_payoff_family_v1"
PERFORMANCE_SEED = 20260907
STATE_FEATURES = (
    "state_spread_bps", "state_log_top_quantity", "state_log_top_notional",
    "state_log_quote_updates", "state_log_quote_age_ms", "state_log_trade_count",
    "state_log_trade_quantity", "state_log_trade_notional", "state_ofi_x_imbalance",
    "state_ofi_x_spread_bps", "state_ofi_x_volatility_bps",
)


@dataclass
class PerformanceModel:
    name: str
    features: list[str]
    base_features: list[str]
    training_rows: int
    config: dict[str, Any]
    ridge: EdgeModel | None = None
    regressors: tuple[Any, Any] | None = None

    def predict_gross_edges(self, rows: Sequence[dict[str, str]]) -> list[tuple[float, float]]:
        if not rows:
            return []
        transformed = _transform_rows(rows, self.base_features, enriched=self.name != "ridge_default")
        if self.ridge is not None:
            predictions = [predict_gross_edges_bps(self.ridge, row) for row in transformed]
        else:
            if self.regressors is None:
                raise ValueError("performance model has no fitted learner")
            matrix = [[float(row[feature]) for feature in self.features] for row in transformed]
            long_predictions = self.regressors[0].predict(matrix)
            short_predictions = self.regressors[1].predict(matrix)
            predictions = [(float(long), float(short)) for long, short in zip(long_predictions, short_predictions)]
        if any(not math.isfinite(value) for pair in predictions for value in pair):
            raise ValueError("performance model produced non-finite gross edges")
        return predictions

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "contract_version": PERFORMANCE_MODEL_VERSION,
            "training_rows": self.training_rows,
            "features": list(self.features),
            "target_order": ["long_gross_bps", "short_gross_bps"],
            "target_formulas": [
                "10000*(future_bid-entry_ask)/entry_ask",
                "10000*(entry_bid-future_ask)/entry_bid",
            ],
            "predictor_information_set": "completed decision snapshot only",
            "config": dict(self.config),
        }
        if self.ridge is not None:
            result["standardizer"] = {
                "fitted_on": "training rows only",
                "means": list(self.ridge.standardizer.means),
                "scales": list(self.ridge.standardizer.scales),
            }
        else:
            result["standardizer"] = None  # Tree splits do not require standardization.
        return result


def fit_performance_model(name: str, train_rows: Sequence[dict[str, str]]) -> PerformanceModel:
    if name not in PERFORMANCE_MODEL_NAMES:
        raise ValueError(f"unknown performance model {name!r}; expected {PERFORMANCE_MODEL_NAMES}")
    if not train_rows:
        raise ValueError("cannot fit a performance model on empty training rows")
    base_features = [feature for feature in DEFAULT_FEATURES if feature in train_rows[0]]
    if not base_features:
        raise ValueError("performance model requires at least one default microstructure feature")
    enriched = name != "ridge_default"
    features = [*base_features, *(STATE_FEATURES if enriched else ())]
    transformed = _transform_rows(train_rows, base_features, enriched=enriched)
    targets = gross_edge_targets_bps(train_rows)
    if name.startswith("ridge_"):
        # The source model provides the exact baseline objective and QR solver.
        # Only target columns are carried across; future fields cannot enter X.
        for row, original in zip(transformed, train_rows):
            row.update({field: original[field] for field in
                        ("entry_ask", "entry_bid", "future_ask", "future_bid")})
        ridge = fit_edge_model(transformed, features, l2=1.0)
        return PerformanceModel(name, features, base_features, len(train_rows), {
            "l2": 1.0,
            "objective": "sum_squared_error + l2*sum(non_intercept_weights**2)",
            "solver": "augmented_householder_qr",
        }, ridge=ridge)

    # The automatic sklearn stopping split shuffles observations. A frozen
    # iteration count keeps model selection in the outer chronological study.
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import HistGradientBoostingRegressor

    config = {
        "loss": "squared_error", "max_iter": 150, "learning_rate": 0.05,
        "max_leaf_nodes": 7, "max_depth": None,
        "min_samples_leaf": max(100, int(0.005 * len(train_rows))),
        "l2_regularization": 10.0, "early_stopping": False,
        "max_bins": 255, "random_state": PERFORMANCE_SEED,
    }
    matrix = [[float(row[feature]) for feature in features] for row in transformed]
    long_model = HistGradientBoostingRegressor(**config)
    short_model = HistGradientBoostingRegressor(**config)
    long_model.fit(matrix, [target[0] for target in targets])
    short_model.fit(matrix, [target[1] for target in targets])
    return PerformanceModel(name, features, base_features, len(train_rows), {
        **config, "sklearn_version": sklearn_version,
    }, regressors=(long_model, short_model))


def gross_edge_targets_bps(rows: Sequence[dict[str, str]]) -> list[tuple[float, float]]:
    """Exact long/short executable returns, excluding the two legs' explicit fees."""
    targets: list[tuple[float, float]] = []
    for row in rows:
        entry_bid = _number(row, "entry_bid", strictly_positive=True)
        entry_ask = _number(row, "entry_ask", strictly_positive=True)
        future_bid = _number(row, "future_bid", strictly_positive=True)
        future_ask = _number(row, "future_ask", strictly_positive=True)
        if entry_ask < entry_bid or future_ask < future_bid:
            raise ValueError("performance targets require uncrossed executable quotes")
        target = (10_000.0 * (future_bid - entry_ask) / entry_ask,
                  10_000.0 * (entry_bid - future_ask) / entry_bid)
        if not all(math.isfinite(value) for value in target):
            raise ValueError("performance targets must be finite")
        targets.append(target)
    return targets


def performance_feature_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Expose the enriched causal feature map, without fitting or reading labels."""
    if not rows:
        return []
    base_features = [feature for feature in DEFAULT_FEATURES if feature in rows[0]]
    return _transform_rows(rows, base_features, enriched=True)


def _transform_rows(
    rows: Sequence[dict[str, str]], base_features: list[str], *, enriched: bool,
) -> list[dict[str, str]]:
    transformed: list[dict[str, str]] = []
    for row in rows:
        values = {feature: _number(row, feature) for feature in base_features}
        if enriched:
            bid = _number(row, "bid", strictly_positive=True)
            ask = _number(row, "ask", strictly_positive=True)
            if ask < bid:
                raise ValueError("performance features require uncrossed decision quotes")
            bid_quantity = _number(row, "bid_qty", nonnegative=True)
            ask_quantity = _number(row, "ask_qty", nonnegative=True)
            ofi = _number(row, "quote_ofi_normalized")
            imbalance = _number(row, "top_imbalance")
            volatility_bps = 10_000.0 * _number(row, "realized_volatility_5", nonnegative=True)
            spread_bps = 10_000.0 * (ask - bid) / ((ask + bid) / 2.0)
            values.update({
                "state_spread_bps": spread_bps,
                "state_log_top_quantity": math.log1p(bid_quantity + ask_quantity),
                "state_log_top_notional": math.log1p(bid * bid_quantity + ask * ask_quantity),
                "state_log_quote_updates": math.log1p(_number(row, "quote_updates_in_bucket", nonnegative=True)),
                "state_log_quote_age_ms": math.log1p(_number(row, "quote_age_ms", nonnegative=True)),
                "state_log_trade_count": math.log1p(_number(row, "trade_count", nonnegative=True)),
                "state_log_trade_quantity": math.log1p(_number(row, "trade_qty", nonnegative=True)),
                "state_log_trade_notional": math.log1p(_number(row, "trade_notional", nonnegative=True)),
                "state_ofi_x_imbalance": ofi * imbalance,
                "state_ofi_x_spread_bps": ofi * spread_bps,
                "state_ofi_x_volatility_bps": ofi * volatility_bps,
            })
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("derived performance features must be finite")
        transformed.append({feature: repr(value) for feature, value in values.items()})
    return transformed


def _number(
    row: dict[str, str], field: str, *, strictly_positive: bool = False, nonnegative: bool = False,
) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"performance input {field!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"performance input {field!r} must be finite")
    if strictly_positive and value <= 0:
        raise ValueError(f"performance input {field!r} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"performance input {field!r} must be nonnegative")
    return value
