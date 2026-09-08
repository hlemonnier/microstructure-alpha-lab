"""Descriptive error overlap and exact Brier decomposition, never a fitted gate."""

from __future__ import annotations

import numpy as np


def error_geometry(probabilities, decisions, labels, *, reference, ensemble_members):
    """Describe a fixed finite forecast library on one aligned asset/date panel.

    The any-correct statistic uses realized labels to select an expert. It is
    only a hindsight ceiling for choosing among these saved hard decisions.
    Brier calculations use natural row weights; decision overlap uses class-
    balanced weights to reproduce the existing per-panel balanced accuracy.
    """
    p, d, y = np.asarray(probabilities, dtype=float), np.asarray(decisions), np.asarray(labels)
    members = np.asarray(ensemble_members)
    if (p.ndim != 3 or p.shape[-1] != 3 or not p.shape[0] or not p.shape[1]
        or d.shape != p.shape[:2] or y.shape != (p.shape[1],)
        or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, rtol=0, atol=1e-12)
        or not np.issubdtype(d.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer)
        or not np.isin(d, [-1, 0, 1]).all() or set(y) != {-1, 0, 1}
        or not isinstance(reference, int) or not 0 <= reference < p.shape[0]
        or members.ndim != 1 or not len(members) or not np.issubdtype(members.dtype, np.integer)
        or len(set(members)) != len(members) or (members < 0).any() or (members >= p.shape[0]).any()):
        raise ValueError("Aligned finite three-class probabilities, integer decisions and all three label classes required")
    counts = np.bincount(y + 1, minlength=3)
    weights = 1 / (3 * counts[y + 1])
    correct = d == y[None, :]
    baseline = correct[reference]
    any_correct = correct.any(axis=0)
    unanimous = (d == d[0]).all(axis=0)
    ba = correct @ weights
    rescue = (correct & ~baseline) @ weights
    harm = (~correct & baseline) @ weights
    np.testing.assert_allclose(rescue - harm, ba - ba[reference], rtol=0, atol=1e-12)
    # Separate directional-error agreement from posterior residual geometry.
    pair_disagreement = np.array([[(d[i] != d[j]) @ weights for j in range(len(p))] for i in range(len(p))])
    target = np.eye(3)[y + 1]
    residuals = p - target
    gram = np.einsum("inc,jnc->ij", residuals, residuals) / len(y)
    norms = np.sqrt(np.diag(gram))
    products = norms[:, None] * norms[None, :]
    cosine = np.divide(gram, products, out=np.zeros_like(gram), where=products > 0)
    # A zero residual has undefined cosine, not a measured zero correlation.
    cosine_values = [[float(cosine[i, j]) if products[i, j] > 0 else None for j in range(len(p))] for i in range(len(p))]
    selected = p[members]
    mean = selected.mean(axis=0)
    member_error = ((selected - target) ** 2).sum(axis=2).mean(axis=0)
    mixture_error = ((mean - target) ** 2).sum(axis=1)
    ambiguity = ((selected - mean) ** 2).sum(axis=2).mean(axis=0)
    residual = member_error - mixture_error - ambiguity
    if np.max(np.abs(residual)) > 1e-12:
        raise ValueError("The per-row Brier ambiguity identity failed")
    histogram = np.bincount(correct.sum(axis=0), weights=weights, minlength=len(p) + 1)
    by_class = {str(c): {"rows": int(counts[c + 1]), "reference_recall": float(baseline[y == c].mean()),
        "hindsight_any_correct_recall": float(any_correct[y == c].mean()),
        "unanimous_wrong_fraction": float((unanimous & ~any_correct)[y == c].mean())} for c in (-1, 0, 1)}
    return {"rows": len(y), "models": len(p), "reference_index": reference,
        "balanced_accuracy": ba.tolist(), "rescue_balanced_mass": rescue.tolist(), "harm_balanced_mass": harm.tolist(),
        "pairwise_balanced_disagreement": pair_disagreement.tolist(),
        "hindsight_any_correct_balanced_accuracy": float(any_correct @ weights),
        "hindsight_rescuable_reference_error_balanced_mass": float((any_correct & ~baseline) @ weights),
        "all_wrong_balanced_mass": float((~any_correct) @ weights),
        "unanimous_balanced_mass": float(unanimous @ weights),
        "unanimous_wrong_balanced_mass": float((unanimous & ~any_correct) @ weights),
        "correct_expert_count_balanced_mass": histogram.tolist(), "by_class": by_class,
        "natural_brier_residual_gram": gram.tolist(), "natural_brier_residual_cosine": cosine_values,
        "equal_expert_brier": {"members": members.tolist(), "average_individual_brier": float(member_error.mean()),
            "mean_probability_brier": float(mixture_error.mean()), "ambiguity": float(ambiguity.mean()),
            "maximum_per_row_identity_error": float(np.max(np.abs(residual)))},
        "hindsight_ceiling_is_a_prediction": False, "conditional_expert_choice_learnability_established": False}
