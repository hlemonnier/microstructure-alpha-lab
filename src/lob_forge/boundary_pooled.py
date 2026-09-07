"""Cross-asset neural forecasts with a compact parameter-sharing ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lob_forge.boundary_neural import natural_posteriors


def build_member_network(feature_count, *, hidden_size=64, members=1):
    import math
    import torch
    from torch import nn

    if feature_count < 1 or hidden_size < 1 or members < 1:
        raise ValueError("Positive architecture dimensions required")

    class MemberLinear(nn.Module):
        def __init__(self, inputs, outputs, *, first=False):
            super().__init__()
            self.shared = nn.Linear(inputs, outputs, bias=False)
            self.input_scale = nn.Parameter(torch.randn(members, inputs) if first else torch.ones(members, inputs))
            self.output_scale = nn.Parameter(torch.ones(members, outputs))
            self.bias = nn.Parameter(torch.zeros(members, outputs))

        def forward(self, x):
            return self.shared(x * self.input_scale) * self.output_scale + self.bias

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if members == 1:
                self.single = nn.Sequential(
                    nn.Linear(feature_count, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 3),
                )
            else:
                self.first = MemberLinear(feature_count, hidden_size, first=True)
                self.second = MemberLinear(hidden_size, hidden_size)
                self.head = nn.Parameter(torch.empty(members, hidden_size, 3))
                nn.init.uniform_(self.head, -1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
                self.bias = nn.Parameter(torch.zeros(members, 3))

        def forward(self, x):
            if members == 1:
                return self.single(x).unsqueeze(1)
            h = x.unsqueeze(1).expand(-1, members, -1)
            h = torch.relu(self.first(h))
            h = torch.relu(self.second(h))
            return torch.einsum("bkh,khc->bkc", h, self.head) + self.bias

    return Network()


def balanced_asset_weights(labels, asset_ids):
    y, assets = np.asarray(labels, dtype=int), np.asarray(asset_ids, dtype=int)
    if y.shape != assets.shape or not len(y) or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Aligned three-class labels and asset IDs required")
    weights = np.empty(len(y), dtype=np.float32)
    priors = {}
    distinct = np.unique(assets)
    for asset in distinct:
        rows = assets == asset
        pi = np.array([(y[rows] == label).mean() for label in [-1, 0, 1]])
        if (pi <= 0).any():
            raise ValueError("Every training asset must contain all three classes")
        priors[int(asset)] = pi
        weights[rows] = len(y) / (len(distinct) * rows.sum() * 3 * pi[y[rows] + 1])
    return priors, weights


@dataclass
class PooledForecaster:
    network: Any
    normalizer: Any
    active: np.ndarray
    columns: list[str]
    priors: dict[int, np.ndarray]
    members: int
    hidden_size: int

    def matrix(self, features, asset_id):
        if list(features.columns) != self.columns or asset_id not in self.priors:
            raise ValueError("Prediction schema or training asset mismatch")
        raw = features.to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Finite prediction inputs required")
        x = self.normalizer.transform(raw[:, self.active])
        return np.column_stack([x, np.full(len(x), 2 * asset_id - 1)]).astype(np.float32)

    def predict_proba(self, features, asset_id, *, batch_size=2048):
        import torch

        x = self.matrix(features, asset_id)
        self.network.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                logits = self.network(torch.from_numpy(x[start : start + batch_size]))
                result.append(torch.softmax(logits, dim=-1).mean(dim=1).numpy())
        return natural_posteriors(np.concatenate(result), self.priors[asset_id])

    def save(self, directory):
        import joblib
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.normalizer, directory / "normalizer.joblib")
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "active": self.active.tolist(),
                "columns": self.columns,
                "priors": {key: value.tolist() for key, value in self.priors.items()},
                "members": self.members,
                "hidden_size": self.hidden_size,
            },
            directory / "model.pt",
        )

    @classmethod
    def load(cls, directory):
        import joblib
        import torch

        state = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        active = np.array(state["active"], dtype=bool)
        network = build_member_network(
            int(active.sum()) + 1, hidden_size=state["hidden_size"], members=state["members"]
        )
        network.load_state_dict(state["state_dict"])
        network.eval()
        return cls(
            network,
            joblib.load(directory / "normalizer.joblib"),
            active,
            state["columns"],
            {int(key): np.array(value) for key, value in state["priors"].items()},
            state["members"],
            state["hidden_size"],
        )
