"""Prospective pooled boosting with an explicit class-weight posterior contract.

This is a research candidate, not a promoted default. Natural probabilities are
recovered separately for each asset after fitting class-balanced cross entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def assert_matching_asset_priors(tree_priors, neural_priors):
    """Compare numerical priors across the tree array and neural mapping formats."""
    if len(tree_priors) != len(neural_priors) or set(neural_priors) != set(range(len(tree_priors))):
        raise ValueError("Tree and neural checkpoints require the same indexed assets")
    for asset, values in enumerate(tree_priors):
        np.testing.assert_array_equal(values, neural_priors[asset])


def asset_class_weights(labels, assets, *, class_balanced):
    y, a = np.asarray(labels), np.asarray(assets)
    if y.ndim != 1 or a.shape != y.shape or not len(y) or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Nonempty aligned three-class labels and asset identifiers required")
    if not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
        raise ValueError("Nonnegative integer asset identifiers required")
    unique = np.unique(a)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Asset identifiers must be contiguous from zero")
    priors, weights = [], np.zeros(len(y), dtype=float)
    for asset in unique:
        mask = a == asset
        counts = np.array([np.count_nonzero(mask & (y == label)) for label in [-1, 0, 1]])
        if (counts == 0).any():
            raise ValueError("Every training asset must contain all three classes")
        priors.append(counts / counts.sum())
        if class_balanced:
            for index, label in enumerate([-1, 0, 1]):
                weights[mask & (y == label)] = len(y) / (len(unique) * 3 * counts[index])
        else:
            weights[mask] = len(y) / (len(unique) * counts.sum())
    return np.asarray(priors), weights


def recover_natural_posterior(weighted_probabilities, priors, *, class_balanced):
    q, prior = np.asarray(weighted_probabilities, dtype=float), np.asarray(priors, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Finite nonnegative three-class probabilities required")
    if not np.allclose(q.sum(axis=1), 1) or prior.shape != (3,) or not np.isfinite(prior).all():
        raise ValueError("Normalized posteriors and three asset priors required")
    if (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError("Strictly positive normalized training priors required")
    natural = q * prior if class_balanced else q.copy()
    return natural / natural.sum(axis=1, keepdims=True)


@dataclass
class WeightedBoostForecaster:
    estimator: Any
    columns: list[str]
    symbols: list[str]
    lower: np.ndarray
    upper: np.ndarray
    priors: np.ndarray
    class_balanced: bool

    def predict_proba(self, frame: pd.DataFrame, symbol: str) -> np.ndarray:
        if list(frame.columns) != self.columns or symbol not in self.symbols:
            raise ValueError("Prediction feature schema and asset must match training")
        x = frame.to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Finite observed features required")
        asset = self.symbols.index(symbol)
        matrix = np.column_stack([np.clip(x, self.lower, self.upper), np.full(len(x), asset)])
        raw = self.estimator.predict_proba(matrix)
        if not np.array_equal(self.estimator.classes_, [-1, 0, 1]):
            raise ValueError("Fitted class ordering must be negative, neutral, positive")
        return recover_natural_posterior(raw, self.priors[asset], class_balanced=self.class_balanced)


def fit_weighted_boost(training_features, training_labels, *, leaves, class_balanced, seed=20260907):
    from sklearn.ensemble import HistGradientBoostingClassifier

    symbols = sorted(training_features)
    if not symbols or set(training_labels) != set(symbols) or leaves not in {7, 31}:
        raise ValueError("Aligned asset mappings and a registered leaf count required")
    columns = list(training_features[symbols[0]].columns)
    if any(list(training_features[symbol].columns) != columns for symbol in symbols):
        raise ValueError("All training assets require the same observed feature schema")
    if any(len(training_features[symbol]) != len(training_labels[symbol]) for symbol in symbols):
        raise ValueError("Each asset requires aligned training features and labels")
    x = pd.concat([training_features[symbol] for symbol in symbols], ignore_index=True).to_numpy(dtype=float)
    y = np.concatenate([training_labels[symbol] for symbol in symbols])
    assets = np.concatenate([np.full(len(training_labels[symbol]), index) for index, symbol in enumerate(symbols)])
    if not np.isfinite(x).all():
        raise ValueError("Finite training features required")
    priors, weights = asset_class_weights(y, assets, class_balanced=class_balanced)
    lower, upper = np.quantile(x, [0.001, 0.999], axis=0)
    matrix = np.column_stack([np.clip(x, lower, upper), assets])
    estimator = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves,
        min_samples_leaf=100, l2_regularization=10, early_stopping=False, random_state=seed,
    )
    estimator.fit(matrix, y, sample_weight=weights)
    return WeightedBoostForecaster(estimator, columns, symbols, lower, upper, priors, class_balanced)
