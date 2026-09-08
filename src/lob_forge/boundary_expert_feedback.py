"""Small delayed full-information mixtures of nine unchanged forecast experts."""

from __future__ import annotations

import numpy as np

CASES = {"frozen_reference": None, "uniform_eight": None, "fixed_reference_prior": None}
CASES.update({f"{mode}_eta{int(rate * 100):02}_h{half}": (mode, rate, half)
    for mode in ("global", "confidence_context") for rate in (.03, .1) for half in (300, 1800)})


def confidence_context(probabilities, decision_priors):
    score = probabilities / decision_priors
    score /= score.sum(axis=1, keepdims=True)
    ordered = np.sort(score, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return 3 * np.argmax(score, axis=1) + np.searchsorted([.1, .25], margin, side="right")


def mix_experts(probabilities, decision_priors, labels, decision_times, release_times, training_prior, *, case):
    """Replay a registered coefficient rule; future labels never enter updates.

    The final expert is the retained reference. All arrays are aligned by the
    original decision clock; expert priors describe their already-frozen causal
    decision policy. Labels and exact release clocks simulate delayed feedback.
    """
    p, pi = np.asarray(probabilities, dtype=float), np.asarray(decision_priors, dtype=float)
    y, times, releases, prior = map(np.asarray, (labels, decision_times, release_times, training_prior))
    if (case not in CASES or p.ndim != 3 or p.shape[1:] != (9, 3) or not len(p) or pi.shape != p.shape
        or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, rtol=0, atol=1e-12)
        or not np.isfinite(pi).all() or (pi <= 0).any() or not np.allclose(pi.sum(-1), 1, rtol=0, atol=1e-12)
        or any(a.shape != (len(p),) or not np.issubdtype(a.dtype, np.integer) for a in (y, times, releases))
        or not np.isin(y, [-1, 0, 1]).all() or (np.abs(times.astype(float)) >= 2 ** 53).any()
        or (np.abs(releases.astype(float)) >= 2 ** 53).any() or (np.diff(times) <= 0).any() or (releases <= times).any()
        or prior.shape != (3,) or not np.isfinite(prior).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1)):
        raise ValueError("Nine aligned natural experts, exact increasing clocks and strictly delayed valid labels required")
    initial = np.r_[np.full(8, 1 / 16), .5]
    weights = np.empty((len(p), 9), dtype=float)
    forecasts = np.empty((len(p), 3), dtype=float)
    updates = np.zeros(len(p), dtype=np.int64)
    contexts = confidence_context(p[:, -1], pi[:, -1])
    if CASES[case] is None:
        fixed = initial if case == "fixed_reference_prior" else np.r_[np.full(8, 1 / 8), 0.]
        if case == "frozen_reference":
            fixed = np.r_[np.zeros(8), 1.]
        weights[:] = fixed
        forecasts = p[:, -1].copy() if case == "frozen_reference" else np.einsum("imc,m->ic", p, fixed)
        return {"probabilities": forecasts, "weights": weights, "updates": updates, "contexts": contexts,
            "final_global_loss": np.zeros(9), "final_context_loss": np.zeros((9, 9))}
    mode, rate, half_life = CASES[case]
    log_initial = np.log(initial)
    global_loss, context_loss = np.zeros(9), np.zeros((9, 9))
    choices = np.argmax(p / pi, axis=2) - 1
    release_order = np.argsort(releases, kind="stable")
    cursor, previous = 0, int(times[0])

    def floored_softmax(loss):
        score = log_initial - loss
        mass = np.exp(score - score.max())
        return .95 * (mass / mass.sum()) + .05 * initial

    for row, clock in enumerate(times):
        clock = int(clock)
        decay = np.exp2(-(clock - previous) / (1000 * half_life))
        global_loss *= decay
        context_loss *= decay
        previous = clock
        while cursor < len(p) and releases[release_order[cursor]] <= clock:
            released = release_order[cursor]
            # The realized class is read only inside this availability guard.
            cls = int(y[released])
            loss = (choices[released] != cls) * (prior.min() / prior[cls + 1])
            aged = rate * np.exp2(-(clock - int(times[released])) / (1000 * half_life)) * loss
            global_loss += aged
            context_loss[contexts[released]] += aged
            cursor += 1
            updates[row] += 1
        w = floored_softmax(global_loss)
        if mode == "confidence_context":
            w = .5 * (w + floored_softmax(context_loss[contexts[row]]))
        weights[row] = w
        forecasts[row] = w @ p[row]
    if (not np.isfinite(forecasts).all() or (forecasts < 0).any()
        or not np.allclose(forecasts.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Finite normalized natural probability mixtures required")
    return {"probabilities": forecasts, "weights": weights, "updates": updates, "contexts": contexts,
        "final_global_loss": global_loss, "final_context_loss": context_loss}
