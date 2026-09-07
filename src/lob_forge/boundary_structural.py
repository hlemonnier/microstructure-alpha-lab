"""Structured movement/direction classifiers and reflection-equivariant priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lob_forge.boundary_forecasts import BASE_FEATURES, ForecastClassifier, fit_classifier


def reflection_signs(columns: list[str]) -> np.ndarray:
    signs = []
    for name in columns:
        if name.startswith("path_area_"):
            signs.append(-1 if int(name.rsplit("_", 1)[1]) in (3, 6, 8, 9) else 1)
        elif (
            name == "flow_x_imbalance"
            or "activity" in name
            or name.startswith("log_")
            or name.startswith("return_vol_")
        ):
            signs.append(1)
        elif (
            "imbalance" in name
            or name.startswith(
                (
                    "quote_ofi",
                    "mid_return",
                    "flow_",
                    "signed_trade_",
                    "return_bps_",
                    "cross_return_",
                    "peer_return_",
                    "peer_flow_",
                )
            )
            or name in ("peer_quote_ofi_normalized", "peer_mid_return_1", "peer_mid_return_5")
        ):
            signs.append(-1)
        else:
            signs.append(1)
    return np.array(signs, dtype=float)


def mirror_features(features: pd.DataFrame) -> pd.DataFrame:
    return features * reflection_signs(list(features.columns))


def _matrix_bounds(features: pd.DataFrame, symmetric: bool):
    lower, upper = np.quantile(features.to_numpy(dtype=float), [0.001, 0.999], axis=0)
    if symmetric:
        odd = reflection_signs(list(features.columns)) < 0
        maximum = np.maximum(abs(lower), abs(upper))
        lower[odd], upper[odd] = -maximum[odd], maximum[odd]
    return lower, upper


@dataclass
class HurdleClassifier:
    name: str
    movement: Any
    direction: Any
    columns: list[str]
    direction_columns: list[str]
    lower: np.ndarray
    upper: np.ndarray
    direction_lower: np.ndarray
    direction_upper: np.ndarray
    priors: np.ndarray
    symmetric: bool

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.columns:
            raise ValueError("structured feature schema mismatch")
        x = np.clip(features.to_numpy(dtype=float), self.lower, self.upper)
        z = np.clip(features[self.direction_columns].to_numpy(dtype=float), self.direction_lower, self.direction_upper)
        moving = self.movement.predict_proba(x)[:, 1]
        up = self.direction.predict_proba(z)[:, 1]
        if self.symmetric:
            reflected = mirror_features(features)
            rx = np.clip(reflected.to_numpy(dtype=float), self.lower, self.upper)
            rz = np.clip(
                reflected[self.direction_columns].to_numpy(dtype=float), self.direction_lower, self.direction_upper
            )
            moving = 0.5 * (moving + self.movement.predict_proba(rx)[:, 1])
            up = 0.5 * (up + 1 - self.direction.predict_proba(rz)[:, 1])
        probabilities = np.column_stack([moving * (1 - up), 1 - moving, moving * up])
        if self.symmetric:
            directional_prior = (self.priors[0] + self.priors[2]) / 2
            augmented_priors = np.array([directional_prior, self.priors[1], directional_prior])
            probabilities *= self.priors / augmented_priors
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


def fit_hurdle(name: str, features: pd.DataFrame, labels: np.ndarray, *, seed=20260907):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name not in ("hurdle_logistic", "hurdle_logistic_symmetric", "hurdle_boosted", "hurdle_boosted_symmetric"):
        raise ValueError("unknown registered hurdle variant")
    y = np.asarray(labels, dtype=int)
    if set(np.unique(y)) != {-1, 0, 1}:
        raise ValueError("allthreeclassesrequiredforhurdletraining")
    symmetric = name.endswith("_symmetric")
    lower, upper = _matrix_bounds(features, symmetric)
    direction_columns = list(BASE_FEATURES)
    dlower, dupper = _matrix_bounds(features[direction_columns], symmetric)
    priors = np.array([(y == label).mean() for label in [-1, 0, 1]])
    train = pd.concat([features, mirror_features(features)], ignore_index=True) if symmetric else features
    training_labels = np.r_[y, -y] if symmetric else y
    x = np.clip(train.to_numpy(dtype=float), lower, upper)
    moving = training_labels != 0
    z = np.clip(train[direction_columns].to_numpy(dtype=float), dlower, dupper)[moving]
    config = dict(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=10,
        early_stopping=False,
        random_state=seed,
    )
    movement = HistGradientBoostingClassifier(**config).fit(x, moving.astype(int))
    if "logistic" in name:
        direction = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000, random_state=seed))
    else:
        direction = HistGradientBoostingClassifier(**config)
    direction.fit(z, (training_labels[moving] > 0).astype(int))
    return HurdleClassifier(
        name,
        movement,
        direction,
        list(features.columns),
        direction_columns,
        lower,
        upper,
        dlower,
        dupper,
        priors,
        symmetric,
    )


@dataclass
class SymmetricClassifier:
    name: str
    base: ForecastClassifier
    priors: np.ndarray

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        direct = self.base.predict_proba(features)
        reflected = self.base.predict_proba(mirror_features(features))[:, ::-1]
        symmetric = (direct + reflected) / 2
        natural = symmetric * (self.priors / self.base.priors)
        return natural / natural.sum(axis=1, keepdims=True)


def fit_symmetric_classifier(features: pd.DataFrame, labels: np.ndarray, *, seed=20260907):
    y = np.asarray(labels, dtype=int)
    train = pd.concat([features, mirror_features(features)], ignore_index=True)
    base = fit_classifier("hgb_7", train, np.r_[y, -y], seed=seed)
    return SymmetricClassifier("symmetric_hgb", base, np.array([(y == label).mean() for label in [-1, 0, 1]]))
