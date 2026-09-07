"""Causal sequence construction and balanced-posterior neural forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lob_forge.ml_models import build_torch_sequence_classifier


def causal_window_indices(times: np.ndarray, window: int) -> np.ndarray:
    """Reset on a gap/session boundary; replicate only the first observed state."""
    t = np.asarray(times, dtype=np.int64)
    if t.ndim != 1 or window < 1 or (len(t) > 1 and (np.diff(t) <= 0).any()):
        raise ValueError("Strictly increasing timestamps and a positive window required")
    if not len(t):
        return np.empty((0, window), dtype=np.int64)
    rows = np.arange(len(t))
    starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(t) != 1000], rows, 0))
    return np.maximum(rows[:, None] - np.arange(window - 1, -1, -1), starts[:, None])


@dataclass
class SequenceNormalizer:
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix):
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Finite nonempty training matrix required")
        lower, upper = np.quantile(x, [0.001, 0.999], axis=0)
        clipped = np.clip(x, lower, upper)
        scale = clipped.std(axis=0)
        scale[scale < 1e-12] = 1
        return cls(lower, upper, clipped.mean(axis=0), scale)

    def transform(self, matrix):
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Finite matrix matching the training feature schema required")
        return ((np.clip(x, self.lower, self.upper) - self.mean) / self.scale).astype(np.float32)


def natural_posteriors(balanced, priors):
    q, pi = np.asarray(balanced, dtype=float), np.asarray(priors, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or pi.shape != (3,) or (pi <= 0).any():
        raise ValueError("Three positive training priors and three-class posteriors required")
    p = q * pi
    return p / p.sum(axis=1, keepdims=True)


@dataclass
class NeuralForecaster:
    network: Any
    normalizer: SequenceNormalizer
    columns: list[str]
    priors: np.ndarray
    architecture: str
    window: int
    hidden_size: int

    def predict_proba(self, matrix, indices, batch_size=1024):
        import torch

        x = self.normalizer.transform(matrix)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch = torch.from_numpy(x[indices[start : start + batch_size]])
                outputs.append(torch.softmax(self.network(batch), dim=1).cpu().numpy())
        return natural_posteriors(np.concatenate(outputs), self.priors)

    def save(self, path):
        import torch

        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "architecture": self.architecture,
                "window": self.window,
                "hidden_size": self.hidden_size,
                "columns": self.columns,
                "priors": self.priors.tolist(),
                "normalizer": {
                    key: getattr(self.normalizer, key).tolist() for key in ["lower", "upper", "mean", "scale"]
                },
            },
            path,
        )

    @classmethod
    def load(cls, path):
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        network = build_torch_sequence_classifier(
            window=state["window"],
            feature_count=len(state["columns"]),
            model_name=state["architecture"],
            hidden_size=state["hidden_size"],
        )
        network.load_state_dict(state["state_dict"])
        network.eval()
        return cls(
            network,
            SequenceNormalizer(**{key: np.array(value) for key, value in state["normalizer"].items()}),
            state["columns"],
            np.array(state["priors"]),
            state["architecture"],
            state["window"],
            state["hidden_size"],
        )
