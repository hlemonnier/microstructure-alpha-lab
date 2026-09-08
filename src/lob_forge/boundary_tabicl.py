"""Frozen in-context classification with an explicit case-control posterior.

TabICL is an optional research dependency. Its public pretrained weights remain
fixed: fit constructs a context and attention cache, without optimizer updates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lob_forge.binance_vision import sha256_file

SYMBOLS = ("BTCUSDT", "ETHUSDT")
SEMANTICS = "tabicl_case_control_prior_v1"
TABICL_SETTINGS = {
    "n_estimators": 2, "norm_methods": ["none", "power"], "batch_size": 1,
    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt", "kv_cache": True,
    "random_state": 20260908, "device": "cpu", "n_jobs": 2,
    "use_amp": False, "use_fa3": False, "offload_mode": "cpu",
    "softmax_temperature": .9, "average_logits": True,
    "feat_shuffle_method": "latin", "class_shuffle_method": "shift",
    "outlier_threshold": 4., "allow_auto_download": False,
}


def _prior(values):
    p = np.asarray(values, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("Strictly positive normalized three-class priors required")
    return p


def case_control_posterior(probabilities, sampled_prior, population_prior):
    """Undo uniform within-class sampling, conditional on the observed asset.

    If S denotes selection into the context, S is independent of X given Y and
    asset. Bayes gives p_pool(y|x,a) proportional to
    p_context(y|x,a) * pi_pool(y|a) / pi_context(y|a).
    This identity does not assert that the fitted context posterior is calibrated.
    """
    q = np.asarray(probabilities, dtype=float)
    sampled, population = _prior(sampled_prior), _prior(population_prior)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(axis=1), 1):
        raise ValueError("Finite normalized three-class context probabilities required")
    unnormalized = q * (population / sampled)
    return unnormalized / unnormalized.sum(axis=1, keepdims=True)


def select_balanced_context(labels, *, requested_rows=3072, seed=20260908):
    """Return deterministic unique positions; never receives assessment data."""
    if set(labels) != set(SYMBOLS) or isinstance(requested_rows, (bool, np.bool_)) or not isinstance(requested_rows, (int, np.integer)) or requested_rows < 6 or requested_rows % 6:
        raise ValueError("Both assets and a positive context budget divisible by six required")
    arrays, counts, priors = {}, {}, {}
    for symbol in SYMBOLS:
        y = np.asarray(labels[symbol])
        if y.ndim != 1 or not len(y) or not np.isin(y, [-1, 0, 1]).all():
            raise ValueError("Nonempty three-class labels required in every context pool")
        arrays[symbol] = y
        counts[symbol] = np.array([np.count_nonzero(y == label) for label in (-1, 0, 1)])
        if (counts[symbol] == 0).any():
            raise ValueError("Every asset context pool must contain all three classes")
        priors[symbol] = counts[symbol] / len(y)
    per_cell = min(requested_rows // 6, min(int(c.min()) for c in counts.values()))
    rng = np.random.default_rng(seed)
    selected = {s: np.sort(np.concatenate([
        rng.choice(np.flatnonzero(arrays[s] == label), size=per_cell, replace=False)
        for label in (-1, 0, 1)])) for s in SYMBOLS}
    sampled = {s: np.array([(arrays[s][selected[s]] == label).mean() for label in (-1, 0, 1)]) for s in SYMBOLS}
    return selected, priors, sampled


@dataclass
class TabiclForecaster:
    estimator: Any
    columns: list[str]
    population_priors: dict[str, np.ndarray]
    sampled_priors: dict[str, np.ndarray]
    checkpoint: str
    checkpoint_sha256: str

    def __post_init__(self):
        if not self.columns or len(set(self.columns)) != len(self.columns) or any(
            set(p) != set(SYMBOLS) for p in (self.population_priors, self.sampled_priors)
        ):
            raise ValueError("Unique observed columns and both asset prior mappings required")
        for priors in (self.population_priors, self.sampled_priors):
            for symbol in SYMBOLS:
                _prior(priors[symbol])

    def matrix(self, frame, symbol):
        if symbol not in SYMBOLS or list(frame.columns) != self.columns:
            raise ValueError("Prediction feature schema and asset must match the context")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Finite observed features required")
        return np.column_stack([values, np.full(len(values), SYMBOLS.index(symbol))])

    def predict_proba(self, frame, symbol, *, batch_size=64):
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("A positive integer query batch size required")
        if not np.array_equal(self.estimator.classes_, [-1, 0, 1]):
            raise ValueError("Context class ordering must be negative, neutral, positive")
        matrix = self.matrix(frame, symbol)
        if not len(matrix):
            return np.empty((0, 3), dtype=float)
        q = np.concatenate([self.estimator.predict_proba(matrix[start:start + batch_size])
                            for start in range(0, len(matrix), batch_size)])
        if q.shape != (len(matrix), 3):
            raise ValueError("One three-class posterior required for every query")
        return case_control_posterior(q, self.sampled_priors[symbol], self.population_priors[symbol])

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if sha256_file(Path(self.checkpoint)) != self.checkpoint_sha256:
            raise ValueError("Pinned local pretrained checkpoint changed")
        self.estimator.save(folder / "context.pkl", save_model_weights=False, save_training_data=True, save_kv_cache=True)
        metadata = {"semantics": SEMANTICS, "columns": self.columns,
            "population_priors": {s: np.asarray(p).tolist() for s, p in self.population_priors.items()},
            "sampled_priors": {s: np.asarray(p).tolist() for s, p in self.sampled_priors.items()},
            "checkpoint": self.checkpoint, "checkpoint_sha256": self.checkpoint_sha256,
            "context_sha256": sha256_file(folder / "context.pkl")}
        (folder / "model.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, folder):
        from tabicl import TabICLClassifier

        folder = Path(folder)
        metadata = json.loads((folder / "model.json").read_text())
        if metadata["semantics"] != SEMANTICS or sha256_file(Path(metadata["checkpoint"])) != metadata["checkpoint_sha256"] or sha256_file(folder / "context.pkl") != metadata["context_sha256"]:
            raise ValueError("Pinned pretrained weights, local context or posterior semantics changed")
        estimator = TabICLClassifier.load(folder / "context.pkl", device="cpu")
        return cls(estimator, metadata["columns"],
                   {s: np.array(p) for s, p in metadata["population_priors"].items()},
                   {s: np.array(p) for s, p in metadata["sampled_priors"].items()},
                   metadata["checkpoint"], metadata["checkpoint_sha256"])


def fit_tabicl_context(features, labels, population_priors, *, checkpoint, checkpoint_sha256):
    """Build the frozen transformer cache using context data only."""
    from tabicl import TabICLClassifier
    import torch

    if set(features) != set(SYMBOLS) or set(labels) != set(SYMBOLS):
        raise ValueError("Both registered assets required")
    if sha256_file(Path(checkpoint)) != checkpoint_sha256:
        raise ValueError("Pinned local pretrained checkpoint changed")
    columns = list(features[SYMBOLS[0]].columns)
    sampled = {}
    for symbol in SYMBOLS:
        y = np.asarray(labels[symbol])
        if list(features[symbol].columns) != columns or y.ndim != 1 or len(y) != len(features[symbol]) or not len(y) or not np.isin(y, [-1, 0, 1]).all():
            raise ValueError("Aligned context feature schemas and three-class labels required")
        sampled[symbol] = _prior(np.array([(y == label).mean() for label in (-1, 0, 1)]))
        if not np.array_equal(sampled[symbol], np.full(3, 1 / 3)):
            raise ValueError("The registered context has equal asset/class cell counts")
    if len(labels[SYMBOLS[0]]) != len(labels[SYMBOLS[1]]):
        raise ValueError("The registered context has equal asset/class cell counts")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    estimator = TabICLClassifier(**TABICL_SETTINGS, model_path=str(Path(checkpoint).resolve()))
    model = TabiclForecaster(estimator, columns, population_priors, sampled, str(Path(checkpoint).resolve()), checkpoint_sha256)
    matrix = np.concatenate([model.matrix(features[s], s) for s in SYMBOLS])
    estimator.fit(matrix, np.concatenate([labels[s] for s in SYMBOLS]))
    return model
