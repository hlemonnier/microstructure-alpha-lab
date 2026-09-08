"""Fixed-budget symmetric, ordered and Langevin-regularized classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lob_forge.boundary_weighted_boost import asset_class_weights, recover_natural_posterior

SEMANTICS = "chronological_symmetric_multiclass_balanced_to_natural_v1"
VARIANTS = {"plain6": {"boosting_type": "Plain", "depth": 6, "posterior_sampling": False},
    "ordered6": {"boosting_type": "Ordered", "depth": 6, "posterior_sampling": False},
    "plain8": {"boosting_type": "Plain", "depth": 8, "posterior_sampling": False},
    "langevin6": {"boosting_type": "Plain", "depth": 6, "posterior_sampling": True}}
PARAMETERS = {"iterations": 1000, "learning_rate": .03, "l2_leaf_reg": 10, "loss_function": "MultiClass", "classes_count": 3,
    "grow_policy": "SymmetricTree", "border_count": 128, "bootstrap_type": "Bayesian", "bagging_temperature": 1,
    "random_strength": 1, "rsm": 1, "leaf_estimation_method": "Newton", "leaf_estimation_iterations": 1,
    "leaf_estimation_backtracking": "No", "score_function": "Cosine", "has_time": True, "nan_mode": "Forbidden",
    "random_seed": 20260908, "task_type": "CPU", "thread_count": 2, "use_best_model": False, "allow_writing_files": False, "verbose": False}


def chronological_training_pool(features, labels, times):
    symbols = sorted(features)
    if symbols != ["BTCUSDT", "ETHUSDT"] or set(labels) != set(symbols) or set(times) != set(symbols):
        raise ValueError("Both original assets and aligned historical inputs required")
    columns = list(features[symbols[0]].columns)
    if not columns or len(columns) != len(set(columns)) or any(list(features[s].columns) != columns for s in symbols):
        raise ValueError("A common nonempty unique feature schema required")
    for symbol in symbols:
        y, clock = np.asarray(labels[symbol]), np.asarray(times[symbol])
        if (y.ndim != 1 or clock.shape != y.shape or len(features[symbol]) != len(y)
            or not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(clock.dtype, np.integer)
            or (clock < 0).any() or (clock > np.iinfo(np.int64).max).any() or len(np.unique(clock)) != len(clock)):
            raise ValueError("Aligned integer labels and unique nonnegative integer clocks required")
    raw = pd.concat([features[s] for s in symbols], ignore_index=True).to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Finite historical training observations required")
    target = np.concatenate([labels[s] for s in symbols])
    assets = np.concatenate([np.full(len(labels[s]), a, dtype=np.int64) for a, s in enumerate(symbols)])
    clock = np.concatenate([times[s] for s in symbols]).astype(np.int64)
    priors, weights = asset_class_weights(target, assets, class_balanced=True)
    lower, upper = np.quantile(raw, [.001, .999], axis=0)
    matrix = np.column_stack([np.clip(raw, lower, upper), assets]).astype(np.float32)
    order = np.lexsort((assets, clock))
    return {"matrix": matrix[order], "labels": target[order], "weights": weights[order], "assets": assets[order], "times": clock[order],
        "priors": priors, "lower": lower, "upper": upper, "columns": columns, "symbols": symbols}


def native_booster(variant, *, iterations=1000):
    import catboost

    if catboost.__version__ != "1.2.10":
        raise ValueError("The frozen stochastic-boosting family requires CatBoost 1.2.10")
    if variant not in VARIANTS or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("A registered stochastic-boosting variant and positive tree count required")
    params = {**PARAMETERS, **VARIANTS[variant], "iterations": iterations}
    if variant == "ordered6":
        params.update(fold_len_multiplier=2, approx_on_full_history=False)
    return catboost.CatBoostClassifier(**params)


def verify_native_booster(booster, variant, *, iterations, training_rows):
    actual = booster.get_all_params()
    # The usual get_all_params summary omits has_time. Verify the effective
    # serialized data-processing/system configuration, not constructor intent.
    native = json.loads(booster.get_metadata()["params"])
    verified = {"has_time": native["data_processing_options"].get("has_time"),
        "thread_count": native["system_options"].get("thread_count"), "task_type": native.get("task_type")}
    if verified != {"has_time": True, "thread_count": 2, "task_type": "CPU"}:
        raise ValueError(f"Effective chronological CPU configuration changed: {verified}")
    if booster.tree_count_ != iterations or not np.array_equal(booster.classes_, [0, 1, 2]):
        raise ValueError("Native final tree count and negative/neutral/positive label ordering changed")
    for name, expected in {"loss_function": "MultiClass", "classes_count": 3, "grow_policy": "SymmetricTree", **VARIANTS[variant]}.items():
        if actual.get(name) != expected:
            raise ValueError(f"Native stochastic-boosting setting changed: {name}={actual.get(name)!r}")
    if variant == "ordered6" and (actual.get("fold_len_multiplier") != 2 or actual.get("approx_on_full_history") is not False):
        raise ValueError("Native ordered historical-fold settings changed")
    if variant == "langevin6" and (actual.get("langevin") is not True or actual.get("diffusion_temperature") != training_rows
        or not np.isclose(actual.get("model_shrink_rate", np.nan), 1 / (2 * training_rows), rtol=1e-6, atol=0)):
        raise ValueError("Native Langevin noise/shrinkage settings do not match historical row count")
    return {**actual, "verified_serialized_configuration": verified}


@dataclass
class StochasticBoostForecaster:
    booster: Any
    columns: list[str]
    symbols: list[str]
    lower: np.ndarray
    upper: np.ndarray
    priors: np.ndarray
    variant: str
    training_rows: int
    iterations: int
    effective_parameters: dict

    def matrix(self, frame, symbol):
        if list(frame.columns) != self.columns or symbol not in self.symbols:
            raise ValueError("Exact historical prediction schema and asset required")
        raw = frame.to_numpy(dtype=float)
        if not np.isfinite(raw).all() or not len(raw):
            raise ValueError("Nonempty finite query observations required")
        return np.column_stack([np.clip(raw, self.lower, self.upper), np.full(len(raw), self.symbols.index(symbol))]).astype(np.float32)

    def predict_proba(self, frame, symbol):
        q = self.booster.predict_proba(self.matrix(frame, symbol), thread_count=2)
        return recover_natural_posterior(q, self.priors[self.symbols.index(symbol)], class_balanced=True)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True)
        self.booster.save_model(str(directory / "model.cbm"))
        metadata = {"semantics": SEMANTICS, "columns": self.columns, "symbols": self.symbols,
            "lower": self.lower.tolist(), "upper": self.upper.tolist(), "priors": self.priors.tolist(), "variant": self.variant,
            "training_rows": self.training_rows, "iterations": self.iterations, "effective_parameters": self.effective_parameters}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.pop("semantics") != SEMANTICS:
            raise ValueError("Incompatible stochastic-boosting prior semantics")
        booster = native_booster(metadata["variant"], iterations=metadata["iterations"])
        booster.load_model(str(directory / "model.cbm"))
        verify_native_booster(booster, metadata["variant"], iterations=metadata["iterations"], training_rows=metadata["training_rows"])
        for name in ("lower", "upper", "priors"):
            metadata[name] = np.asarray(metadata[name], dtype=float)
        return cls(booster=booster, **metadata)


def fit_stochastic_boost(training_features, training_labels, training_times, *, variant, iterations=1000):
    from catboost import Pool

    packed = chronological_training_pool(training_features, training_labels, training_times)
    booster = native_booster(variant, iterations=iterations)
    pool = Pool(packed["matrix"], label=packed["labels"] + 1, weight=packed["weights"], timestamp=packed["times"].astype(np.uint64))
    booster.fit(pool)
    actual = verify_native_booster(booster, variant, iterations=iterations, training_rows=len(packed["labels"]))
    return StochasticBoostForecaster(booster, packed["columns"], packed["symbols"], packed["lower"], packed["upper"], packed["priors"],
        variant, len(packed["labels"]), iterations, actual)
