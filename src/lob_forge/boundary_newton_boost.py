"""A fixed XGBoost alternative with explicit asset/class posterior recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lob_forge.boundary_weighted_boost import asset_class_weights, recover_natural_posterior


@dataclass
class NewtonBoostForecaster:
    booster: Any
    columns: list[str]
    symbols: list[str]
    lower: np.ndarray
    upper: np.ndarray
    priors: np.ndarray
    class_balanced: bool
    parameters: dict

    def predict_proba(self, frame, symbol):
        if list(frame.columns) != self.columns or symbol not in self.symbols:
            raise ValueError("Prediction feature schema and asset must match training")
        x = frame.to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Finite observed features required")
        asset = self.symbols.index(symbol)
        matrix = np.ascontiguousarray(np.column_stack([
            np.clip(x, self.lower, self.upper), np.full(len(x), asset),
        ]), dtype=np.float32)
        q = self.booster.inplace_predict(matrix, strict_shape=True)
        return recover_natural_posterior(q, self.priors[asset], class_balanced=self.class_balanced)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path / "booster.ubj")
        metadata = {"columns": self.columns, "symbols": self.symbols,
                    "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                    "priors": self.priors.tolist(), "class_balanced": self.class_balanced,
                    "parameters": self.parameters, "label_mapping": [-1, 0, 1]}
        (path / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        import xgboost as xgb

        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text())
        booster = xgb.Booster()
        booster.load_model(path / "booster.ubj")
        booster.set_param({"nthread": 2, "device": "cpu"})
        if metadata.pop("label_mapping") != [-1, 0, 1] or booster.num_features() != len(metadata["columns"]) + 1:
            raise ValueError("Checkpoint class mapping and feature count must match")
        for name in ("lower", "upper", "priors"):
            metadata[name] = np.asarray(metadata[name], dtype=float)
        return cls(booster=booster, **metadata)


def fit_newton_boost(training_features, training_labels, *, max_depth, class_balanced,
                     rounds=400, min_child_weight=30, seed=20260907):
    """Fit once with fixed rounds; no assessment or validation data are accepted.

    min_child_weight bounds summed Hessian mass, not the number of rows. The
    class-balanced loss has optimum q_k proportional to p_k / pi_asset,k;
    multiplying by the corresponding training prior recovers natural p.
    """
    import xgboost as xgb

    symbols = sorted(training_features)
    if not symbols or set(training_labels) != set(symbols):
        raise ValueError("Aligned nonempty asset mappings required")
    if max_depth not in (3, 6) or not isinstance(rounds, int) or rounds < 1 or min_child_weight < 0:
        raise ValueError("Use the registered tree depth and positive training budget")
    columns = list(training_features[symbols[0]].columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("A nonempty unique observed feature schema is required")
    if any(list(training_features[s].columns) != columns or len(training_features[s]) != len(training_labels[s]) for s in symbols):
        raise ValueError("Every asset requires aligned labels and the same feature schema")
    x = pd.concat([training_features[s] for s in symbols], ignore_index=True).to_numpy(dtype=float)
    y = np.concatenate([training_labels[s] for s in symbols])
    assets = np.concatenate([np.full(len(training_labels[s]), i) for i, s in enumerate(symbols)])
    if not np.isfinite(x).all():
        raise ValueError("Finite training features required")
    priors, weights = asset_class_weights(y, assets, class_balanced=class_balanced)
    lower, upper = np.quantile(x, [0.001, 0.999], axis=0)
    matrix = np.ascontiguousarray(np.column_stack([np.clip(x, lower, upper), assets]), dtype=np.float32)
    parameters = {"objective": "multi:softprob", "num_class": 3, "tree_method": "hist",
                  "device": "cpu", "nthread": 2, "max_depth": max_depth, "max_bin": 256,
                  "eta": 0.05, "min_child_weight": min_child_weight, "lambda": 10,
                  "alpha": 0, "gamma": 0, "subsample": 1, "colsample_bytree": 1,
                  "grow_policy": "depthwise", "seed": seed, "base_score": 0.5,
                  "disable_default_eval_metric": True, "verbosity": 0}
    data = xgb.DMatrix(matrix, label=y + 1, weight=weights, nthread=2)
    booster = xgb.train(parameters, data, num_boost_round=rounds)
    return NewtonBoostForecaster(booster, columns, symbols, lower, upper, priors,
                                  class_balanced, {**parameters, "rounds": rounds})
