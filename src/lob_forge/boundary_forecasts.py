"""Causal temporal/cross-asset forecast representations for registered research.

All targets are kept outside this feature map. Path areas retain temporal order;
class-prior adjustments affect decisions only, not reported posterior log loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lob_forge.baselines import DEFAULT_FEATURES

BASE_FEATURES = tuple(DEFAULT_FEATURES)
OBSERVED_FIELDS = (
    "decision_time",
    "bid",
    "ask",
    "bid_qty",
    "ask_qty",
    "mid",
    "quote_age_ms",
    "quote_updates_in_bucket",
    "quote_ofi_normalized",
    "quote_ofi_5_normalized",
    "top_imbalance",
    "top_imbalance_mean_5",
    "mid_return_1",
    "mid_return_5",
    "realized_volatility_5",
    "trade_imbalance",
    "buy_qty",
    "sell_qty",
    "trade_qty",
    "trade_notional",
    "trade_count",
    "depth_imbalance_1pct",
    "notional_imbalance_1pct",
    "depth_imbalance_5pct",
    "notional_imbalance_5pct",
)
PEER_FIELDS = (
    "quote_ofi_normalized",
    "top_imbalance",
    "mid_return_1",
    "mid_return_5",
    "trade_imbalance",
    "quote_updates_in_bucket",
)


def path_areas(increments: np.ndarray, window: int) -> np.ndarray:
    """Level-two log-signature antisymmetric areas of causal trailing increments."""
    x = np.asarray(increments, dtype=np.float64)
    if x.ndim != 2 or window < 1 or not np.isfinite(x).all():
        raise ValueError("finite two-dimensional increments and positive window required")
    n, dimension = x.shape
    cumulative = np.vstack([np.zeros((1, dimension)), np.cumsum(x, axis=0)])
    left = np.maximum(0, np.arange(n) + 1 - window)
    delta = cumulative[1:] - cumulative[left]
    columns = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            prefix_area = np.r_[0.0, np.cumsum(0.5 * (cumulative[:-1, i] * x[:, j] - cumulative[:-1, j] * x[:, i]))]
            columns.append(
                prefix_area[1:]
                - prefix_area[left]
                - 0.5 * (cumulative[left, i] * delta[:, j] - cumulative[left, j] * delta[:, i])
            )
    return np.column_stack(columns) if columns else np.empty((n, 0))


def _observed(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(OBSERVED_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing decision-time fields: {sorted(missing)}")
    observed = frame.loc[:, OBSERVED_FIELDS].astype(float).copy()
    if not np.isfinite(observed.to_numpy()).all():
        raise ValueError("decision-time observations must be finite")
    times = observed["decision_time"].to_numpy(dtype=np.int64)
    if len(times) > 1 and not np.all(np.diff(times) == 1000):
        raise ValueError("registered temporal representation requires contiguous one-second sessions")
    if ((observed["ask"] < observed["bid"]) | (observed["bid"] <= 0)).any():
        raise ValueError("decision quotes must be positive and uncrossed")
    return observed


def feature_frame(frame: pd.DataFrame, peer: pd.DataFrame, representation: str) -> pd.DataFrame:
    """Transform one asset/session; caller must never concatenate sessions first."""
    own = _observed(frame)
    if representation == "base":
        return own.loc[:, BASE_FEATURES].copy()
    if representation not in ("temporal_cross", "signature_cross"):
        raise ValueError("unknown registered representation")
    other = _observed(peer)
    joined = pd.merge_asof(
        own[["decision_time"]],
        other[["decision_time", "quote_age_ms", *PEER_FIELDS]],
        on="decision_time",
        direction="backward",
        allow_exact_matches=True,
    )
    # A peer feature is usable only after its whole bucket closed. Equal decision
    # times mean both observations are strictly earlier than that shared boundary.
    peer_times = pd.merge_asof(
        own[["decision_time"]],
        other[["decision_time"]].rename(columns={"decision_time": "peer_time"}),
        left_on="decision_time",
        right_on="peer_time",
        direction="backward",
    )
    peer_age = own["decision_time"].to_numpy() - peer_times["peer_time"].to_numpy()
    valid_peer = (peer_age <= 1000) & (joined["quote_age_ms"].to_numpy() + peer_age <= 1000)
    result = own.loc[:, BASE_FEATURES].copy()
    result["spread_bps"] = (own["ask"] - own["bid"]) / own["mid"] * 10000
    result["log_depth"] = np.log1p(own["bid_qty"] + own["ask_qty"])
    result["log_depth_notional"] = np.log1p(own["bid"] * own["bid_qty"] + own["ask"] * own["ask_qty"])
    result["log_activity"] = np.log1p(own["quote_updates_in_bucket"])
    result["log_trade_qty"] = np.log1p(own["trade_qty"])
    result["log_trade_count"] = np.log1p(own["trade_count"])
    result["quote_age_ms"] = own["quote_age_ms"]
    result["peer_available"] = valid_peer.astype(float)
    for field in PEER_FIELDS:
        result["peer_" + field] = np.where(valid_peer, joined[field].to_numpy(), 0.0)
    flow = own["quote_ofi_normalized"]
    signed_trade = (own["buy_qty"] - own["sell_qty"]) / (own["bid_qty"] + own["ask_qty"]).clip(lower=1e-9)
    price_return = own["mid_return_1"] * 10000
    for span in (2, 5, 15, 60):
        for name, series in [
            ("flow", flow),
            ("signed_trade", signed_trade),
            ("return_bps", price_return),
            ("imbalance", own["top_imbalance"]),
            ("peer_return_bps", result["peer_mid_return_1"] * 10000),
            ("peer_flow", result["peer_quote_ofi_normalized"]),
        ]:
            result[f"{name}_ewm_{span}"] = series.ewm(span=span, adjust=False).mean()
        result[f"return_vol_{span}"] = price_return.pow(2).rolling(span, min_periods=1).mean().pow(0.5)
        result[f"flow_activity_{span}"] = flow.abs().rolling(span, min_periods=1).mean()
        result[f"cross_return_{span}"] = result[f"return_bps_ewm_{span}"] - result[f"peer_return_bps_ewm_{span}"]
    for lag in (1, 2, 5, 15):
        result[f"flow_lag_{lag}"] = flow.shift(lag).fillna(0)
        result[f"imbalance_lag_{lag}"] = own["top_imbalance"].shift(lag).fillna(0)
        result[f"peer_return_lag_{lag}"] = result["peer_mid_return_1"].shift(lag).fillna(0) * 10000
    result["flow_acceleration"] = result["flow_ewm_2"] - result["flow_ewm_15"]
    result["flow_x_imbalance"] = flow * own["top_imbalance"]
    if representation == "signature_cross":
        increments = np.column_stack(
            [price_return, flow, signed_trade, result["peer_mid_return_1"] * 10000, np.ones(len(own))]
        )
        additions = {}
        for window in (5, 15, 60):
            areas = path_areas(increments, window)
            for column in range(areas.shape[1]):
                additions[f"path_area_{window}_{column}"] = areas[:, column] / window
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("causal feature map produced nonfinite values")
    return result


@dataclass
class ForecastClassifier:
    name: str
    estimator: Any
    columns: list[str]
    lower: np.ndarray
    upper: np.ndarray
    priors: np.ndarray

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.columns:
            raise ValueError("forecast feature schema mismatch")
        matrix = np.clip(features.to_numpy(dtype=float), self.lower, self.upper)
        raw = self.estimator.predict_proba(matrix)
        result = np.zeros((len(features), 3))
        for col, label in enumerate(self.estimator.classes_):
            result[:, int(label) + 1] = raw[:, col]
        return result


def fit_classifier(
    name: str, features: pd.DataFrame, labels: np.ndarray, *, seed: int = 20260907
) -> ForecastClassifier:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=int)
    if set(np.unique(y)) != {-1, 0, 1}:
        raise ValueError("training session set must contain all three classes")
    x = features.to_numpy(dtype=float)
    if not np.isfinite(x).all() or len(x) != len(y):
        raise ValueError("finite aligned training features and labels required")
    lower, upper = np.quantile(x, [0.001, 0.999], axis=0)
    # Quantile clipping is fit on training only; test outliers cannot alter it.
    x = np.clip(x, lower, upper)
    if name.startswith("logistic_"):
        c = float(name.removeprefix("logistic_"))
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=seed))
    elif name.startswith("hgb_"):
        leaves = int(name.removeprefix("hgb_"))
        estimator = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=leaves,
            min_samples_leaf=100,
            l2_regularization=10,
            early_stopping=False,
            random_state=seed,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=50, max_features=0.7, n_jobs=2, random_state=seed
        )
    else:
        raise ValueError("unknown registered classifier")
    estimator.fit(x, y)
    priors = np.array([(y == label).mean() for label in [-1, 0, 1]])
    return ForecastClassifier(name, estimator, list(features.columns), lower, upper, priors)


def classification_metrics(model: ForecastClassifier, probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("finite normalized probabilities aligned to three-class labels required")
    balanced_decisions = np.argmax(p / model.priors, axis=1) - 1
    natural_decisions = np.argmax(p, axis=1) - 1
    return {
        "rows": len(y),
        "balanced_accuracy": balanced_accuracy_score(y, balanced_decisions),
        "natural_accuracy": accuracy_score(y, natural_decisions),
        "log_loss": log_loss(y, p, labels=[-1, 0, 1]),
        "macro_f1": f1_score(y, balanced_decisions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, balanced_decisions, labels=[-1, 0, 1]).tolist(),
        "class_counts": [(y == label).sum().item() for label in [-1, 0, 1]],
    }
