"""Learn recent conditional log-probability corrections to a frozen forecast."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_weighted_boost import asset_class_weights, recover_natural_posterior

RESIDUAL_SEMANTICS = "frozen_posterior_plus_causal_residual_logits_v1"
CHECKPOINT_ROUNDS = (0, 16, 64, 128, 256)


def checked_base_probabilities(values, rows):
    p = np.asarray(values, dtype=float)
    if p.shape != (rows, 3) or not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Aligned strictly positive natural base probabilities required")
    return p


def residual_base_margin(probabilities, priors, *, class_balanced):
    """At zero correction, recovery with pi preserves the natural base.

    Weighted CE targets q proportional to p/pi. Therefore its initial margin
    is log(p_base)-log(pi), not log(p_base). Natural CE uses log(p_base).
    """
    p = checked_base_probabilities(probabilities, len(probabilities))
    prior = np.asarray(priors, dtype=float)
    if prior.shape not in ((3,), p.shape) or not np.isfinite(prior).all() or (prior <= 0).any() or not np.allclose(prior.sum(axis=-1), 1):
        raise ValueError("Positive normalized training priors matching the base probabilities required")
    return np.log(p) - (np.log(prior) if class_balanced else 0)


def _augmented(frame, probabilities):
    raw = frame.to_numpy(dtype=float)
    p = checked_base_probabilities(probabilities, len(raw))
    if not np.isfinite(raw).all():
        raise ValueError("Finite original observations required")
    odds = np.log(p[:, [0, 2]]) - np.log(p[:, [1]])
    return np.column_stack([raw, odds]), p


@dataclass
class ResidualBoostForecaster:
    booster: Any
    columns: list[str]
    symbols: list[str]
    lower: np.ndarray
    upper: np.ndarray
    priors: np.ndarray
    class_balanced: bool
    parameters: dict
    selected_rounds: int

    def predict_proba(self, frame, symbol, base_probabilities, *, rounds=None):
        if list(frame.columns) != self.columns or symbol not in self.symbols:
            raise ValueError("Residual prediction schema and asset must match training")
        count = self.selected_rounds if rounds is None else rounds
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 0 <= count <= self.parameters["rounds"]:
            raise ValueError("Prediction rounds must lie inside the fitted correction path")
        raw, p = _augmented(frame, base_probabilities)
        if count == 0:
            # XGBoost iteration_range=(0, 0) means ALL trees, not zero trees.
            return p.copy()
        asset = self.symbols.index(symbol)
        matrix = np.ascontiguousarray(np.column_stack([np.clip(raw, self.lower, self.upper), np.full(len(raw), asset)]), dtype=np.float32)
        margin = np.ascontiguousarray(residual_base_margin(p, self.priors[asset], class_balanced=self.class_balanced), dtype=np.float32)
        q = self.booster.inplace_predict(matrix, base_margin=margin, iteration_range=(0, int(count)), strict_shape=True)
        return recover_natural_posterior(q, self.priors[asset], class_balanced=self.class_balanced)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path / "booster.ubj")
        metadata = {"semantics": RESIDUAL_SEMANTICS, "columns": self.columns, "symbols": self.symbols,
            "lower": self.lower.tolist(), "upper": self.upper.tolist(), "priors": self.priors.tolist(),
            "class_balanced": self.class_balanced, "parameters": self.parameters, "selected_rounds": self.selected_rounds,
            "label_mapping": [-1, 0, 1], "extra_inputs": ["base_log_odds_negative_neutral", "base_log_odds_positive_neutral", "asset"]}
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory):
        import xgboost as xgb

        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text())
        booster = xgb.Booster()
        booster.load_model(path / "booster.ubj")
        booster.set_param({"nthread": 2, "device": "cpu"})
        if metadata.pop("semantics") != RESIDUAL_SEMANTICS or metadata.pop("label_mapping") != [-1, 0, 1]:
            raise ValueError("Residual checkpoint semantics and class ordering must match")
        if metadata.pop("extra_inputs") != ["base_log_odds_negative_neutral", "base_log_odds_positive_neutral", "asset"]:
            raise ValueError("Residual checkpoint base-forecast inputs changed")
        if booster.num_features() != len(metadata["columns"]) + 3 or booster.num_boosted_rounds() != metadata["parameters"]["rounds"]:
            raise ValueError("Residual checkpoint feature count or fitted path changed")
        if not 0 <= metadata["selected_rounds"] <= booster.num_boosted_rounds():
            raise ValueError("Selected residual checkpoint lies outside the fitted path")
        for name in ("lower", "upper", "priors"):
            metadata[name] = np.asarray(metadata[name], dtype=float)
        return cls(booster=booster, **metadata)


def fit_residual_boost(training_features, training_labels, base_probabilities, *, max_depth,
                       class_balanced, rounds=256, min_child_weight=30, seed=20260907):
    """Only past training rows and forecasts from an earlier frozen model enter."""
    import xgboost as xgb

    symbols = sorted(training_features)
    if not symbols or set(training_labels) != set(symbols) or set(base_probabilities) != set(symbols):
        raise ValueError("Aligned observations, labels and base forecasts for every asset required")
    if max_depth not in (2, 4) or isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1 or min_child_weight < 0:
        raise ValueError("Registered residual tree depth and positive training budget required")
    columns = list(training_features[symbols[0]].columns)
    if not columns or len(set(columns)) != len(columns) or any(
        list(training_features[s].columns) != columns or len(training_features[s]) != len(training_labels[s]) for s in symbols
    ):
        raise ValueError("Aligned training rows and identical unambiguous feature schemas required")
    frames, probabilities = zip(*[_augmented(training_features[s], base_probabilities[s]) for s in symbols])
    x, p = np.concatenate(frames), np.concatenate(probabilities)
    y = np.concatenate([training_labels[s] for s in symbols])
    assets = np.concatenate([np.full(len(training_labels[s]), i) for i, s in enumerate(symbols)])
    priors, weights = asset_class_weights(y, assets, class_balanced=class_balanced)
    lower, upper = np.quantile(x, [0.001, 0.999], axis=0)
    matrix = np.ascontiguousarray(np.column_stack([np.clip(x, lower, upper), assets]), dtype=np.float32)
    margins = np.ascontiguousarray(residual_base_margin(p, priors[assets], class_balanced=class_balanced), dtype=np.float32)
    parameters = {"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "device": "cpu", "nthread": 2,
        "max_depth": max_depth, "max_bin": 256, "eta": 0.025, "min_child_weight": min_child_weight, "lambda": 10,
        "alpha": 0, "gamma": 0, "subsample": 1, "colsample_bytree": 1, "grow_policy": "depthwise", "seed": seed,
        "base_score": 0.5, "disable_default_eval_metric": True, "verbosity": 0}
    data = xgb.DMatrix(matrix, label=y + 1, weight=weights, base_margin=margins, nthread=2)
    booster = xgb.train(parameters, data, num_boost_round=rounds)
    return ResidualBoostForecaster(booster, columns, symbols, lower, upper, priors, class_balanced,
                                   {**parameters, "rounds": rounds}, rounds)


def select_residual_rounds(model, validation_features, validation_labels, base_probabilities, decision_priors, *, candidates=CHECKPOINT_ROUNDS):
    """Choose once on earlier validation data, including the unchanged base."""
    symbols = model.symbols
    if any(set(v) != set(symbols) for v in (validation_features, validation_labels, base_probabilities, decision_priors)):
        raise ValueError("Aligned earlier validation partitions for every asset required")
    if not candidates or candidates[0] != 0 or tuple(sorted(set(candidates))) != tuple(candidates):
        raise ValueError("Distinct increasing correction checkpoints including zero required")
    history, best, chosen = [], (-np.inf, -np.inf, -np.inf), None
    for count in candidates:
        metrics = {s: classification_metrics(SimpleNamespace(priors=decision_priors[s]),
            model.predict_proba(validation_features[s], s, base_probabilities[s], rounds=count), validation_labels[s]) for s in symbols}
        key = (float(np.mean([m["balanced_accuracy"] for m in metrics.values()])),
               -float(np.mean([m["log_loss"] for m in metrics.values()])), -count)
        if key > best:
            best, chosen = key, count
        history.append({"rounds": count, "validation": metrics})
    model.selected_rounds = int(chosen)
    return {"selected_rounds": int(chosen), "candidates": list(candidates), "history": history,
            "selector": "mean asset balanced accuracy, then natural log loss, then fewer rounds"}
