"""Decompose forecast quality without changing the three-class target or gate."""

from __future__ import annotations

import numpy as np


def forecast_diagnostics(probabilities, labels):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Nonempty, unmodified three-class labels required")
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Finite normalized three-class probabilities required")
    # One shared smoothing step keeps the likelihood decomposition algebraic.
    p = np.maximum(p, np.finfo(float).eps)
    p /= p.sum(axis=1, keepdims=True)
    moving = y != 0
    movement_probability = p[:, 0] + p[:, 2]
    directional_up_probability = p[:, 2] / movement_probability
    movement_loss = -np.log(np.where(moving, movement_probability, p[:, 1]))
    direction_loss = -np.log(np.where(y[moving] == 1, p[moving, 2], p[moving, 0]) / movement_probability[moving])
    full_loss = -np.log(p[np.arange(len(y)), y.astype(int) + 1])
    contribution = float(direction_loss.sum() / len(y))
    np.testing.assert_allclose(full_loss.mean(), movement_loss.mean() + contribution, rtol=1e-13, atol=1e-14)
    natural_matrix = confusion_matrix(y, p.argmax(axis=1) - 1, labels=[-1, 0, 1])
    counts = natural_matrix.sum(axis=1)
    recall = [float(natural_matrix[k, k] / counts[k]) if counts[k] else None for k in range(3)]
    return {
        "rows": len(y),
        "movement_fraction": float(moving.mean()),
        "movement_binary_log_loss": float(movement_loss.mean()),
        "conditional_direction_log_loss": float(direction_loss.mean()) if moving.any() else None,
        "direction_log_loss_contribution_per_decision": contribution,
        "three_class_log_loss_with_shared_smoothing": float(full_loss.mean()),
        "movement_roc_auc": float(roc_auc_score(moving, movement_probability)) if len(np.unique(moving)) == 2 else None,
        "conditional_direction_roc_auc": (
            float(roc_auc_score(y[moving] == 1, directional_up_probability[moving]))
            if len(np.unique(y[moving])) == 2 else None
        ),
        "natural_decision_confusion_matrix": natural_matrix.tolist(),
        "natural_decision_recall_negative_neutral_positive": recall,
        "scope": "Direction is diagnosed conditional on the realized outcome being non-neutral. That future condition is not an available trade-selection rule. These are explanatory metrics, not additional promotion gates.",
    }
