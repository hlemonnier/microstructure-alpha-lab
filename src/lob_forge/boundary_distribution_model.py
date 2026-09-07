"""Class-balanced distribution learning with exact three-class aggregation."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from lob_forge.boundary_distribution_targets import COARSE_CLASSES, coarse_probabilities
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_neural import natural_posteriors
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights


def build_distribution_network(feature_count, hidden_size=64):
    from torch import nn

    return nn.Sequential(nn.Linear(feature_count, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 9))


def distribution_loss(logits, targets, *, fine_weight, ranked_weight):
    """A proper weighted sum of coarse/fine log scores and the ranked score."""
    import torch
    from torch.nn import functional as f

    if not 0 <= fine_weight <= 1 or not np.isfinite(ranked_weight) or ranked_weight < 0:
        raise ValueError("Registered nonnegative loss weights are required")
    if logits.ndim != 2 or logits.shape[1] != 9 or targets.shape != (len(logits),):
        raise ValueError("Aligned nine-bin logits and targets are required")
    logq = f.log_softmax(logits, dim=1)
    coarse_logq = torch.logsumexp(logq.reshape(-1, 3, 3), dim=2)
    losses = (1 - fine_weight) * f.nll_loss(coarse_logq, targets // 3, reduction="none")
    losses += fine_weight * f.nll_loss(logq, targets, reduction="none")
    if ranked_weight:
        cdf = logq.exp().cumsum(dim=1)[:, :-1]
        observed_cdf = (targets[:, None] <= torch.arange(8, device=targets.device)[None, :]).to(logits.dtype)
        losses += ranked_weight * ((cdf - observed_cdf) ** 2).mean(dim=1)
    return losses


class DistributionForecaster(PooledForecaster):
    def predict_proba(self, features, asset_id, *, batch_size=2048):
        import torch

        matrix = self.matrix(features, asset_id)
        self.network.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                results.append(torch.softmax(self.network(torch.from_numpy(matrix[start:start + batch_size])), dim=1).numpy())
        # Training weights depend on the COARSE class, not the nine-bin prior.
        # Aggregate q first, then invert the same three-class weighting as the
        # reference. Multiplying by nine-bin frequencies would be incorrect.
        return natural_posteriors(coarse_probabilities(np.concatenate(results)), self.priors[asset_id])

    def save(self, directory):
        import joblib
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.normalizer, directory / "normalizer.joblib")
        torch.save({"state_dict": self.network.state_dict(), "active": self.active.tolist(), "columns": self.columns,
                    "priors": {key: value.tolist() for key, value in self.priors.items()}, "hidden_size": self.hidden_size,
                    "head_outputs": 9, "posterior_contract": "coarse_class_weight_recovery_v1"}, directory / "distribution_model.pt")

    @classmethod
    def load(cls, directory):
        import joblib
        import torch

        state = torch.load(directory / "distribution_model.pt", weights_only=True, map_location="cpu")
        if state["head_outputs"] != 9 or state["posterior_contract"] != "coarse_class_weight_recovery_v1":
            raise ValueError("Unknown distribution checkpoint contract")
        active = np.array(state["active"], dtype=bool)
        network = build_distribution_network(int(active.sum()) + 1, state["hidden_size"])
        network.load_state_dict(state["state_dict"])
        network.eval()
        return cls(network, joblib.load(directory / "normalizer.joblib"), active, state["columns"],
                   {int(k): np.array(v) for k, v in state["priors"].items()}, 1, state["hidden_size"])


def fit_distribution_neural(training_features, training_bins, validation_features, validation_labels, *, fine_weight, ranked_weight):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from threadpoolctl import threadpool_limits

    if any(set(mapping) != set(SYMBOLS) for mapping in (training_features, training_bins, validation_features, validation_labels)):
        raise ValueError("Both registered assets are required")
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    columns = list(training_features[SYMBOLS[0]].columns)
    if any(list(frame.columns) != columns for frame in [*training_features.values(), *validation_features.values()]):
        raise ValueError("Training and validation observation schemas must match")
    train_x = pd.concat([training_features[s] for s in SYMBOLS], ignore_index=True)
    fine_y = np.concatenate([training_bins[s] for s in SYMBOLS])
    if fine_y.ndim != 1 or not np.isin(fine_y, np.arange(9)).all() or len(train_x) != len(fine_y):
        raise ValueError("Aligned exact nine-bin targets are required")
    fine_y = fine_y.astype(int)
    coarse_y = COARSE_CLASSES[fine_y]
    assets = np.concatenate([np.full(len(training_bins[s]), a, dtype=int) for a, s in enumerate(SYMBOLS)])
    priors, weights = balanced_asset_weights(coarse_y, assets)
    raw_matrix = train_x.to_numpy(dtype=float)
    if not np.isfinite(raw_matrix).all():
        raise ValueError("Finite observation-only training inputs are required")
    active = np.ptp(raw_matrix, axis=0) > 0
    normalizer = QuantileTransformer(n_quantiles=min(1024, len(train_x)), output_distribution="normal",
                                    subsample=min(100000, len(train_x)), random_state=CONFIG["seed"])
    normalizer.fit(raw_matrix[:, active])
    matrix = np.column_stack([normalizer.transform(raw_matrix[:, active]), 2 * assets - 1]).astype(np.float32)
    network = build_distribution_network(matrix.shape[1], CONFIG["hidden_size"])
    learner = DistributionForecaster(network, normalizer, active, columns, priors, 1, CONFIG["hidden_size"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    targets, sample_weights = torch.tensor(fine_y, dtype=torch.long), torch.from_numpy(weights)
    best_key, best_state, best_epoch, history = (-np.inf, -np.inf), None, None, []
    with threadpool_limits(limits=CONFIG["threads"]):
        for epoch in range(1, CONFIG["epochs"] + 1):
            network.train()
            order, losses = rng.permutation(len(fine_y)), []
            for start in range(0, len(order), CONFIG["batch_size"]):
                rows = order[start:start + CONFIG["batch_size"]]
                logits = network(torch.from_numpy(matrix[rows]))
                loss = (distribution_loss(logits, targets[rows], fine_weight=fine_weight, ranked_weight=ranked_weight) * sample_weights[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite distribution training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), CONFIG["gradient_norm_limit"])
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = {s: classification_metrics(SimpleNamespace(priors=priors[a]), learner.predict_proba(validation_features[s], a), validation_labels[s])
                          for a, s in enumerate(SYMBOLS)}
            key = (float(np.mean([m["balanced_accuracy"] for m in validation.values()])), -float(np.mean([m["log_loss"] for m in validation.values()])))
            if key > best_key:
                best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": validation})
    network.load_state_dict(best_state)
    network.eval()
    return learner, {"history": history, "best_epoch": best_epoch, "config": dict(CONFIG), "fine_weight": fine_weight, "ranked_weight": ranked_weight}
