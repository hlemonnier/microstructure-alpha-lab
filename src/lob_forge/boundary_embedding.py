"""Train-only numerical embeddings and shared-parameter forecast ensembles."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import balanced_asset_weights, build_member_network

EMBEDDING_SEMANTICS = "clipped_quantile_piecewise_natural_member_mean_v1"
ARCHITECTURES = {"scalar1": (False, 1), "scalar8": (False, 8), "piecewise1": (True, 1), "piecewise8": (True, 8)}


@dataclass
class QuantileBins:
    edges: list[np.ndarray]

    @classmethod
    def fit(cls, values, *, bins=32):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("Nonempty finite active training features required")
        if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 256:
            raise ValueError("A positive bounded number of quantile bins required")
        edges = [np.unique(np.quantile(x[:, j], np.linspace(0, 1, bins + 1))) for j in range(x.shape[1])]
        if any(len(e) < 2 or not np.isfinite(np.diff(e)).all() or (np.diff(e) <= 0).any() for e in edges):
            raise ValueError("Remove constant columns before fitting nonzero-width bins")
        return cls(edges)

    def transform(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.edges) or not np.isfinite(x).all():
            raise ValueError("Finite inputs with the fitted feature schema required")
        indices = np.empty(x.shape, dtype=np.int64)
        fractions = np.empty(x.shape, dtype=np.float32)
        for feature, edges in enumerate(self.edges):
            clipped = np.clip(x[:, feature], edges[0], edges[-1])
            loc = np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, len(edges) - 2)
            indices[:, feature] = loc
            fractions[:, feature] = (clipped - edges[loc]) / (edges[loc + 1] - edges[loc])
        return indices, fractions


def build_piecewise_embedding(bin_counts, *, dimensions=4):
    import math
    import torch
    from torch import nn

    if not bin_counts or any(not isinstance(n, int) or n < 1 for n in bin_counts) or dimensions < 1:
        raise ValueError("Positive feature-bin counts and embedding width required")

    class PiecewiseEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(len(bin_counts), max(bin_counts), dimensions))
            self.bias = nn.Parameter(torch.empty(len(bin_counts), dimensions))
            with torch.no_grad():
                for feature, count in enumerate(bin_counts):
                    nn.init.uniform_(self.weight[feature, :count], -1 / math.sqrt(count), 1 / math.sqrt(count))
                    nn.init.uniform_(self.bias[feature], -1 / math.sqrt(count), 1 / math.sqrt(count))
            self.register_buffer("feature_ids", torch.arange(len(bin_counts)), persistent=False)

        def forward(self, indices, fractions):
            # This equals dense cumulative-bin encoding followed by a separate
            # linear projection per feature, including the weight gradients.
            prefix = torch.cat([torch.zeros_like(self.weight[:, :1]), self.weight.cumsum(dim=1)], dim=1)
            return prefix[self.feature_ids, indices] + fractions[..., None] * self.weight[self.feature_ids, indices] + self.bias

    return PiecewiseEmbedding()


def build_embedding_network(bin_counts, *, dimensions=4, members=1, hidden_size=64):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = build_piecewise_embedding(bin_counts, dimensions=dimensions)
            self.backbone = build_member_network(len(bin_counts) * dimensions + 1, members=members, hidden_size=hidden_size)

        def forward(self, indices, fractions, asset_values):
            values = self.embedding(indices, fractions).flatten(1)
            return self.backbone(torch.cat([values, asset_values[:, None]], dim=1))

    return Network()


def mean_member_natural_probabilities(weighted, prior):
    q, pi = np.asarray(weighted, dtype=float), np.asarray(prior, dtype=float)
    if q.ndim != 3 or q.shape[2] != 3 or not q.shape[1] or not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(axis=2), 1):
        raise ValueError("Normalized finite member-by-class probability vectors required")
    if pi.shape != (3,) or not np.isfinite(pi).all() or (pi <= 0).any() or not np.isclose(pi.sum(), 1):
        raise ValueError("Three positive normalized training asset priors required")
    p = q * pi
    p /= p.sum(axis=2, keepdims=True)
    return p.mean(axis=1)


def _forward(network, prepared, rows):
    import torch

    if len(prepared) == 1:
        return network(torch.from_numpy(prepared[0][rows]))
    return network(*(torch.from_numpy(v[rows]) for v in prepared))


@dataclass
class EmbeddingForecaster:
    network: object
    normalizer: object
    active: np.ndarray
    columns: list[str]
    priors: dict[int, np.ndarray]
    architecture: str
    dimensions: int = 4
    hidden_size: int = 64

    def prepare(self, features, assets):
        if list(features.columns) != self.columns:
            raise ValueError("Embedding prediction schema must match training")
        raw = features.to_numpy(dtype=float)
        assets = np.broadcast_to(np.asarray(assets), (len(raw),))
        if not np.isfinite(raw).all() or not np.issubdtype(assets.dtype, np.integer) or not np.isin(assets, list(self.priors)).all():
            raise ValueError("Finite features and known integer training assets required")
        x = raw[:, self.active]
        asset_values = (2 * assets - 1).astype(np.float32)
        if ARCHITECTURES[self.architecture][0]:
            indices, fractions = self.normalizer.transform(x)
            return indices, fractions, asset_values
        return (np.column_stack([self.normalizer.transform(x), asset_values]).astype(np.float32),)

    def predict_proba(self, features, asset_id, *, batch_size=2048):
        import torch

        prepared = self.prepare(features, asset_id)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                logits = _forward(self.network, prepared, slice(start, start + batch_size))
                outputs.append(mean_member_natural_probabilities(torch.softmax(logits, dim=-1).numpy(), self.priors[asset_id]))
        return np.concatenate(outputs)

    def save(self, directory):
        import joblib
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.normalizer, directory / "normalizer.joblib")
        torch.save({"semantics": EMBEDDING_SEMANTICS, "state_dict": self.network.state_dict(), "active": self.active.tolist(),
            "columns": self.columns, "priors": {k: p.tolist() for k, p in self.priors.items()}, "architecture": self.architecture,
            "dimensions": self.dimensions, "hidden_size": self.hidden_size}, directory / "model.pt")

    @classmethod
    def load(cls, directory):
        import joblib
        import torch

        state = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        if state["semantics"] != EMBEDDING_SEMANTICS or state["architecture"] not in ARCHITECTURES:
            raise ValueError("Embedding checkpoint probability and architecture contract changed")
        normalizer = joblib.load(directory / "normalizer.joblib")
        active = np.asarray(state["active"], dtype=bool)
        embedded, members = ARCHITECTURES[state["architecture"]]
        network = build_embedding_network([len(e) - 1 for e in normalizer.edges], dimensions=state["dimensions"], members=members,
            hidden_size=state["hidden_size"]) if embedded else build_member_network(int(active.sum()) + 1, members=members, hidden_size=state["hidden_size"])
        network.load_state_dict(state["state_dict"])
        network.eval()
        return cls(network, normalizer, active, state["columns"], {int(k): np.asarray(p) for k, p in state["priors"].items()},
                   state["architecture"], state["dimensions"], state["hidden_size"])


def fit_embedding_neural(training_features, training_labels, validation_features, validation_labels, *, architecture):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from threadpoolctl import threadpool_limits

    if architecture not in ARCHITECTURES or any(set(m) != set(SYMBOLS) for m in
        (training_features, training_labels, validation_features, validation_labels)):
        raise ValueError("A registered architecture and both training/validation assets required")
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    columns = list(training_features[SYMBOLS[0]].columns)
    if any(list(f.columns) != columns for f in [*training_features.values(), *validation_features.values()]):
        raise ValueError("Matched training and validation feature schemas required")
    if any(len(training_features[s]) != len(training_labels[s]) or len(validation_features[s]) != len(validation_labels[s]) for s in SYMBOLS):
        raise ValueError("Aligned training and validation labels required")
    train_x = pd.concat([training_features[s] for s in SYMBOLS], ignore_index=True)
    train_y = np.concatenate([training_labels[s] for s in SYMBOLS])
    assets = np.concatenate([np.full(len(training_labels[s]), i) for i, s in enumerate(SYMBOLS)])
    priors, weights = balanced_asset_weights(train_y, assets)
    raw = train_x.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Finite observed training features required")
    active = np.ptp(raw, axis=0) > 0
    if not active.any():
        raise ValueError("At least one nonconstant training feature required")
    embedded, members = ARCHITECTURES[architecture]
    if embedded:
        normalizer = QuantileBins.fit(raw[:, active], bins=32)
        network = build_embedding_network([len(e) - 1 for e in normalizer.edges], dimensions=4, members=members, hidden_size=CONFIG["hidden_size"])
    else:
        normalizer = QuantileTransformer(n_quantiles=min(1024, len(train_x)), output_distribution="normal",
            subsample=min(100000, len(train_x)), random_state=CONFIG["seed"])
        normalizer.fit(raw[:, active])
        network = build_member_network(int(active.sum()) + 1, members=members, hidden_size=CONFIG["hidden_size"])
    learner = EmbeddingForecaster(network, normalizer, active, columns, priors, architecture)
    prepared = learner.prepare(train_x, assets)
    target, sample_weights = torch.tensor(train_y + 1, dtype=torch.long), torch.from_numpy(weights)
    optimizer = torch.optim.AdamW(network.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best_key, best_state, best_epoch, history = (-np.inf, -np.inf), None, None, []
    with threadpool_limits(limits=CONFIG["threads"]):
        for epoch in range(1, CONFIG["epochs"] + 1):
            network.train()
            order = rng.permutation(len(train_y))
            losses = []
            for start in range(0, len(order), CONFIG["batch_size"]):
                rows = order[start:start + CONFIG["batch_size"]]
                logits = _forward(network, prepared, rows)
                repeated_y = target[rows, None].expand(-1, members).reshape(-1)
                member_losses = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), repeated_y, reduction="none").reshape(len(rows), members)
                loss = (member_losses.mean(dim=1) * sample_weights[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite embedding learner loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), CONFIG["gradient_norm_limit"])
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = {s: classification_metrics(SimpleNamespace(priors=priors[i]), learner.predict_proba(validation_features[s], i),
                validation_labels[s]) for i, s in enumerate(SYMBOLS)}
            key = (float(np.mean([m["balanced_accuracy"] for m in validation.values()])), -float(np.mean([m["log_loss"] for m in validation.values()])))
            if key > best_key:
                best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": validation})
    network.load_state_dict(best_state)
    network.eval()
    return learner, {"architecture": architecture, "members": members, "embedding_dimensions": 4 if embedded else None,
        "bin_counts": [len(e) - 1 for e in normalizer.edges] if embedded else None, "best_epoch": best_epoch,
        "history": history, "config": dict(CONFIG), "parameter_count": sum(p.numel() for p in network.parameters()),
        "posterior_contract": "recover each natural member posterior then average"}
