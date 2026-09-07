"""Fine-tune frozen pooled forecasts on released same-day morning outcomes.

Rebase the training logits to the morning class mix while retaining the original
posterior-recovery convention. This changes neither the initial natural forecast
nor the saved forecaster's probability contract.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def morning_masks(decision_times, release_times, day_start_ms, *, window):
    """Purged morning updates and selection, both finished before noon forecasts."""
    clock, release = np.asarray(decision_times), np.asarray(release_times)
    if window not in ("six_hours", "full_morning") or clock.ndim != 1 or release.shape != clock.shape:
        raise ValueError("A registered morning window and aligned clocks are required")
    if any(not np.isfinite(v).all() or not np.equal(v, np.floor(v)).all() for v in (clock, release)) or (release <= clock).any() or (np.diff(clock) <= 0).any():
        raise ValueError("Chronological integer times and strictly delayed label releases are required")
    cutoff = day_start_ms + (11 * 60 + 30) * 60000
    first = day_start_ms + 120000 if window == "full_morning" else cutoff - 6 * 3600000
    training = (clock >= first) & (clock < cutoff) & (release <= cutoff)
    validation_end = day_start_ms + (11 * 60 + 57) * 60000
    validation = (clock >= cutoff + 10000) & (clock < validation_end) & (release <= validation_end + 10000)
    return training, validation


def prior_rebasing_offsets(original_priors, recent_priors):
    """For raw logits z, natural p is softmax(z + log(original_prior)).

Weighted morning CE should instead operate on q=softmax(z+log(original_prior)
    -log(recent_prior)). Multiplying q by recent_prior and normalizing recovers
    the same natural p. Only the loss uses this offset; the existing forecaster
    continues to recover its natural probabilities with original_prior.
    """
    old, new = np.asarray(original_priors, dtype=float), np.asarray(recent_priors, dtype=float)
    if old.ndim != 2 or old.shape[1] != 3 or new.shape != old.shape or any(
        not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1) for p in (old, new)
    ):
        raise ValueError("Matching positive normalized asset-by-class prior matrices are required")
    return np.log(old) - np.log(new)


def fit_morning_neural(checkpoint, train_features, train_labels, validation_features, validation_labels, validation_priors, *, learning_rate, epochs=4):
    """Fit with training/validation inputs only; assessment data is not accepted."""
    import torch
    from threadpoolctl import threadpool_limits

    if learning_rate not in (0.0001, 0.0003) or epochs != 4:
        raise ValueError("Use the fixed four-epoch morning development family")
    if any(set(mapping) != set(SYMBOLS) for mapping in (train_features, train_labels, validation_features, validation_labels, validation_priors)):
        raise ValueError("Both registered assets are required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260907)
    rng = np.random.default_rng(20260907)
    learner = PooledForecaster.load(checkpoint)
    if learner.members != 1 or set(learner.priors) != {0, 1}:
        raise ValueError("The frozen two-asset, single-member pooled checkpoint is required")
    if any(len(train_features[s]) != len(train_labels[s]) or len(validation_features[s]) != len(validation_labels[s]) for s in SYMBOLS):
        raise ValueError("Each asset needs aligned training and validation observations")
    labels = np.concatenate([train_labels[s] for s in SYMBOLS])
    assets = np.concatenate([np.full(len(train_labels[s]), i, dtype=int) for i, s in enumerate(SYMBOLS)])
    recent_priors, weights = balanced_asset_weights(labels, assets)
    old_matrix = np.array([learner.priors[i] for i in range(len(SYMBOLS))])
    new_matrix = np.array([recent_priors[i] for i in range(len(SYMBOLS))])
    offsets = prior_rebasing_offsets(old_matrix, new_matrix)
    # Validate the separately chosen decision prior, which may use a longer
    # available morning history than this gradient update's training window.
    prior_rebasing_offsets(old_matrix, np.array([validation_priors[s] for s in SYMBOLS]))
    matrix = np.concatenate([learner.matrix(train_features[s], i) for i, s in enumerate(SYMBOLS)])
    if len(matrix) != len(labels):
        raise ValueError("Aligned training observations and labels are required")
    target = torch.tensor(labels + 1, dtype=torch.long)
    weight_tensor = torch.from_numpy(weights)
    offset_tensor = torch.tensor(offsets[assets], dtype=torch.float32)
    optimizer = torch.optim.AdamW(learner.network.parameters(), lr=learning_rate, weight_decay=0.01)
    history = []
    best_key, best_state, best_epoch = (-np.inf, -np.inf), None, None
    with threadpool_limits(limits=2):
        for epoch in range(epochs + 1):
            losses = []
            if epoch:
                learner.network.train()
                order = rng.permutation(len(labels))
                for start in range(0, len(order), 1024):
                    rows = order[start:start + 1024]
                    logits = learner.network(torch.from_numpy(matrix[rows])).squeeze(1) + offset_tensor[rows]
                    loss = (torch.nn.functional.cross_entropy(logits, target[rows], reduction="none") * weight_tensor[rows]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite morning adaptation loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            validation = {
                s: classification_metrics(SimpleNamespace(priors=validation_priors[s]), learner.predict_proba(validation_features[s], i), validation_labels[s])
                for i, s in enumerate(SYMBOLS)
            }
            key = (float(np.mean([v["balanced_accuracy"] for v in validation.values()])), -float(np.mean([v["log_loss"] for v in validation.values()])))
            if key > best_key:
                best_key, best_state, best_epoch = key, copy.deepcopy(learner.network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)) if losses else None, "validation": validation})
    learner.network.load_state_dict(best_state)
    learner.network.eval()
    # The probability interface, feature normalization and posterior convention
    # remain those of the original checkpoint throughout all gradient updates.
    np.testing.assert_array_equal(np.array([learner.priors[i] for i in range(len(SYMBOLS))]), old_matrix)
    return learner, {
        "learning_rate": learning_rate, "epochs": epochs, "best_epoch": best_epoch, "history": history,
        "recent_training_priors": new_matrix.tolist(), "posterior_recovery_priors": old_matrix.tolist(),
        "training_logit_offsets": offsets.tolist(), "validation_decision_priors": {s: np.asarray(validation_priors[s]).tolist() for s in SYMBOLS},
        "normalizer_refit": False, "initial_checkpoint_is_validation_candidate": True,
    }
