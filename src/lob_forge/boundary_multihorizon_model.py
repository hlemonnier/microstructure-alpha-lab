"""Joint future-horizon supervision with an unchanged five-second forecast path."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_multihorizon_targets import HORIZONS_MS
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network

MULTIHORIZON_MODEL_SEMANTICS = "independent_head_class_weights_shared_backbone_v1"


def multihorizon_weights(labels, available, assets):
    y, valid, a = np.asarray(labels), np.asarray(available), np.asarray(assets)
    if y.ndim != 2 or y.shape[1] != len(HORIZONS_MS) or valid.shape != y.shape or valid.dtype != bool or a.shape != (len(y),):
        raise ValueError("Aligned five-head targets, boolean availability and asset IDs required")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(a.dtype, np.integer) or set(np.unique(a)) != {0, 1}:
        raise ValueError("Integer labels and both registered assets required")
    if not valid[:, 0].all() or not np.isin(y[valid], [-1, 0, 1]).all() or (y[~valid] != -2).any():
        raise ValueError("Primary targets must be complete and unavailable auxiliary labels must be masked")
    weights, priors = np.zeros(y.shape, dtype=np.float32), []
    for head in range(y.shape[1]):
        mask = valid[:, head]
        if set(a[mask]) != {0, 1}:
            raise ValueError("Every head requires valid outcomes from both assets")
        pi, w = balanced_asset_weights(y[mask, head], a[mask])
        weights[mask, head] = w * (len(y) / mask.sum())
        priors.append(pi)
    return priors, weights


def build_multihorizon_network(feature_count, hidden_size=64):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            # Construct every original parameter before any auxiliary draws.
            self.base = build_member_network(feature_count, hidden_size=hidden_size, members=1)
            self.auxiliary = nn.ModuleList([nn.Linear(hidden_size, 3) for _ in HORIZONS_MS[1:]])

        def forward(self, x):
            return self.base(x)

        def all_horizons(self, x):
            hidden = self.base.single[:-1](x)
            return torch.stack([self.base.single[-1](hidden), *(head(hidden) for head in self.auxiliary)], dim=1)

    return Network()


def multihorizon_loss(logits, targets, sample_weights, *, auxiliary_weight):
    import torch

    if auxiliary_weight not in (0, .25, .5):
        raise ValueError("Use a registered auxiliary loss weight")
    expected_heads = 1 if auxiliary_weight == 0 else len(HORIZONS_MS)
    if logits.ndim != 3 or logits.shape[1:] != (expected_heads, 3) or targets.shape != logits.shape[:2] or sample_weights.shape != targets.shape:
        raise ValueError("Aligned logits, class targets and head sample weights required")
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="none").reshape(targets.shape)
    if auxiliary_weight == 0:
        return (loss.mean(dim=1) * sample_weights[:, 0]).mean()
    weighted = loss * sample_weights
    return ((1 - auxiliary_weight) * weighted[:, 0] + auxiliary_weight * weighted[:, 1:].mean(dim=1)).mean()


class MultihorizonForecaster(PooledForecaster):
    def save(self, directory):
        import joblib
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.normalizer, directory / "normalizer.joblib")
        torch.save({"state_dict": self.network.state_dict(), "active": self.active.tolist(), "columns": self.columns,
            "priors": {k: p.tolist() for k, p in self.priors.items()}, "hidden_size": self.hidden_size,
            "horizons_ms": list(HORIZONS_MS), "semantics": MULTIHORIZON_MODEL_SEMANTICS}, directory / "model.pt")

    @classmethod
    def load(cls, directory):
        import joblib
        import torch

        state = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        if state["semantics"] != MULTIHORIZON_MODEL_SEMANTICS or state["horizons_ms"] != list(HORIZONS_MS):
            raise ValueError("Multi-horizon checkpoint contract changed")
        active = np.asarray(state["active"], dtype=bool)
        network = build_multihorizon_network(int(active.sum()) + 1, hidden_size=state["hidden_size"])
        network.load_state_dict(state["state_dict"])
        network.eval()
        return cls(network, joblib.load(directory / "normalizer.joblib"), active, state["columns"],
            {int(k): np.asarray(p) for k, p in state["priors"].items()}, 1, state["hidden_size"])


def fit_multihorizon_neural(training_features, training_targets, training_available, validation_features, validation_labels, *, auxiliary_weight):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from threadpoolctl import threadpool_limits

    if auxiliary_weight not in (0, .25, .5) or any(set(mapping) != set(SYMBOLS) for mapping in
        (training_features, training_targets, training_available, validation_features, validation_labels)):
        raise ValueError("A registered loss weight and both training/validation assets required")
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    columns = list(training_features[SYMBOLS[0]].columns)
    if any(list(f.columns) != columns for f in [*training_features.values(), *validation_features.values()]):
        raise ValueError("Training and validation observation schemas must match")
    if any(len(training_features[s]) != len(training_targets[s]) or len(validation_features[s]) != len(validation_labels[s]) for s in SYMBOLS):
        raise ValueError("Aligned targets and observation rows required")
    train_x = pd.concat([training_features[s] for s in SYMBOLS], ignore_index=True)
    train_y = np.concatenate([training_targets[s] for s in SYMBOLS])
    valid = np.concatenate([training_available[s] for s in SYMBOLS])
    assets = np.concatenate([np.full(len(training_targets[s]), i) for i, s in enumerate(SYMBOLS)])
    priors, weights = multihorizon_weights(train_y, valid, assets)
    raw = train_x.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Finite training observations required")
    active = np.ptp(raw, axis=0) > 0
    if not active.any():
        raise ValueError("At least one nonconstant training feature required")
    normalizer = QuantileTransformer(n_quantiles=min(1024, len(raw)), output_distribution="normal",
        subsample=min(100000, len(raw)), random_state=CONFIG["seed"])
    normalizer.fit(raw[:, active])
    matrix = np.column_stack([normalizer.transform(raw[:, active]), 2 * assets - 1]).astype(np.float32)
    network = build_multihorizon_network(matrix.shape[1], hidden_size=CONFIG["hidden_size"])
    learner = MultihorizonForecaster(network, normalizer, active, columns, priors[0], 1, CONFIG["hidden_size"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    targets = torch.tensor(np.where(valid, train_y + 1, 0), dtype=torch.long)
    sample_weights = torch.from_numpy(weights)
    best_key, best_state, best_epoch, history = (-np.inf, -np.inf), None, None, []
    with threadpool_limits(limits=CONFIG["threads"]):
        for epoch in range(1, CONFIG["epochs"] + 1):
            network.train()
            order, losses = rng.permutation(len(train_y)), []
            for start in range(0, len(order), CONFIG["batch_size"]):
                rows = order[start:start + CONFIG["batch_size"]]
                values = torch.from_numpy(matrix[rows])
                if auxiliary_weight == 0:
                    loss = multihorizon_loss(network(values), targets[rows, :1], sample_weights[rows, :1], auxiliary_weight=0)
                else:
                    loss = multihorizon_loss(network.all_horizons(values), targets[rows], sample_weights[rows], auxiliary_weight=auxiliary_weight)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite multi-horizon training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), CONFIG["gradient_norm_limit"])
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = {s: classification_metrics(SimpleNamespace(priors=priors[0][i]), learner.predict_proba(validation_features[s], i),
                validation_labels[s]) for i, s in enumerate(SYMBOLS)}
            key = (float(np.mean([m["balanced_accuracy"] for m in validation.values()])), -float(np.mean([m["log_loss"] for m in validation.values()])))
            if key > best_key:
                best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": validation})
    network.load_state_dict(best_state)
    network.eval()
    return learner, {"history": history, "best_epoch": best_epoch, "config": dict(CONFIG), "auxiliary_weight": auxiliary_weight,
        "head_training_priors": [{str(a): pi.tolist() for a, pi in head.items()} for head in priors],
        "head_available_rows": valid.sum(axis=0).tolist(), "head_weight_totals": weights.astype(float).sum(axis=0).tolist(),
        "horizons_ms": list(HORIZONS_MS), "parameter_count": sum(p.numel() for p in network.parameters()),
        "primary_inference_contract": "unchanged five-second path and five-second training class prior"}
