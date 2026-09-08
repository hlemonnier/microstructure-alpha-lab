"""Bounded proper losses with the original neural probability semantics."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_neural import natural_posteriors
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network

VARIANTS = ("log", "brier", "spherical2", "spherical4")


def proper_score_loss(logits, target, variant):
    """Return per-row losses for q=softmax(logits), never escort probabilities."""
    import torch
    from torch.nn import functional as F

    if (variant not in VARIANTS or logits.ndim != 2 or logits.shape[1] != 3 or not len(logits)
        or not logits.is_floating_point() or target.shape != (len(logits),) or target.dtype != torch.long
        or target.device != logits.device or not torch.isfinite(logits).all() or ((target < 0) | (target > 2)).any()):
        raise ValueError("Finite three-class logits and aligned integer class indices required")
    if variant == "log":
        return F.cross_entropy(logits, target, reduction="none")
    if variant == "brier":
        residual = torch.softmax(logits, dim=-1) - F.one_hot(target, 3).to(logits.dtype)
        return 1.5 * residual.square().sum(dim=-1)
    beta = 2 if variant == "spherical2" else 4
    alpha = (beta - 1) / beta
    # softmax(beta*u)_y**alpha equals q_y**(beta-1)/||q||_beta**(beta-1).
    # q=softmax(u) retains the same balanced probability meaning as log loss.
    log_score = F.log_softmax(beta * logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    return -(3 ** alpha / (beta - 1)) * torch.expm1(alpha * log_score)


def predict_matrix(learner, matrix, asset, *, batch_size=2048):
    """Preserve the original MLP's exact softmax, averaging and prior operations."""
    import torch

    x = np.asarray(matrix)
    if (asset not in (0, 1) or x.ndim != 2 or x.shape[1] != int(learner.active.sum()) + 1 or not len(x)
        or x.dtype != np.float32 or not np.isfinite(x).all() or not np.all(x[:, -1] == 2 * asset - 1)
        or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("Finite original normalized query matrix and matching asset required")
    learner.network.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = learner.network(torch.from_numpy(x[start:start + batch_size]))
            result.append(torch.softmax(logits, dim=-1).mean(dim=1).numpy())
    return natural_posteriors(np.concatenate(result), learner.priors[asset])


def fit_proper_score(original, matrix, labels, assets, validation, validation_labels, *, variant):
    """Fit the original architecture; assessment data is absent from the interface."""
    import torch
    from threadpoolctl import threadpool_limits

    x, y, aa = np.asarray(matrix), np.asarray(labels), np.asarray(assets)
    dimensions = int(original.active.sum()) + 1
    if (variant not in VARIANTS or original.members != 1 or original.hidden_size != CONFIG["hidden_size"]
        or x.ndim != 2 or x.shape[1] != dimensions or x.dtype != np.float32 or not np.isfinite(x).all()
        or y.shape != (len(x),) or aa.shape != y.shape or not np.issubdtype(y.dtype, np.integer)
        or not np.issubdtype(aa.dtype, np.integer) or not np.isin(y, [-1, 0, 1]).all() or set(aa) != {0, 1}
        or not np.array_equal(x[:, -1], 2 * aa - 1) or set(validation) != {0, 1} or set(validation_labels) != {0, 1}):
        raise ValueError("Original normalized historical rows, assets, labels and validation partitions required")
    priors, weights = balanced_asset_weights(y, aa)
    for asset in (0, 1):
        np.testing.assert_array_equal(priors[asset], original.priors[asset])
        vx, vy = np.asarray(validation[asset]), np.asarray(validation_labels[asset])
        if (vx.ndim != 2 or vx.shape[1] != dimensions or vx.dtype != np.float32 or not len(vx)
            or not np.isfinite(vx).all() or not np.all(vx[:, -1] == 2 * asset - 1)
            or vy.shape != (len(vx),) or not np.issubdtype(vy.dtype, np.integer) or not np.isin(vy, [-1, 0, 1]).all()):
            raise ValueError("Original normalized validation rows and integer labels required")
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    network = build_member_network(dimensions, members=1, hidden_size=CONFIG["hidden_size"])
    learner = PooledForecaster(network, original.normalizer, original.active.copy(), list(original.columns), priors, 1, CONFIG["hidden_size"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    target, sample_weights = torch.tensor(y + 1, dtype=torch.long), torch.from_numpy(weights)
    best_key, best_state, best_epoch, history, steps = (-np.inf, -np.inf), None, None, [], 0
    with threadpool_limits(limits=CONFIG["threads"]):
        for epoch in range(1, CONFIG["epochs"] + 1):
            network.train()
            order = rng.permutation(len(y))
            losses = []
            for start in range(0, len(order), CONFIG["batch_size"]):
                rows = order[start:start + CONFIG["batch_size"]]
                logits = network(torch.from_numpy(x[rows]))
                repeated_y = target[rows, None].expand(-1, 1).reshape(-1)
                member_losses = proper_score_loss(logits.reshape(-1, 3), repeated_y, variant).reshape(len(rows), -1)
                loss = (member_losses.mean(dim=1) * sample_weights[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite proper-score training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), CONFIG["gradient_norm_limit"])
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
            metrics = {}
            for asset, symbol in enumerate(SYMBOLS):
                p = predict_matrix(learner, validation[asset], asset)
                metrics[symbol] = classification_metrics(SimpleNamespace(priors=priors[asset]), p, validation_labels[asset])
            key = (float(np.mean([m["balanced_accuracy"] for m in metrics.values()])), -float(np.mean([m["log_loss"] for m in metrics.values()])))
            if key > best_key:
                best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": metrics})
    network.load_state_dict(best_state)
    network.eval()
    return learner, {"history": history, "best_epoch": best_epoch, "config": dict(CONFIG), "variant": variant, "optimizer_steps": steps,
        "probabilities": "softmax_original_logits_then_original_asset_prior_recovery"}
