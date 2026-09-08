import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_expert_feedback import CASES, confidence_context, mix_experts  # noqa: E402


def inputs(rows=30):
    rng = np.random.default_rng(417)
    p = rng.dirichlet([1, 2, 1], size=(rows, 9))
    pi = np.tile([.2, .5, .3], (rows, 9, 1))
    y = rng.integers(-1, 2, rows)
    times = 1_600_000_000_000 + 1000 * np.arange(rows)
    # Out-of-order arrival and exact equality with a later decision are included.
    release = times + np.tile([5100, 12000, 6000], int(np.ceil(rows / 3)))[:rows]
    return p, pi, y, times, release, np.array([.2, .5, .3])


def explicit_weights(args, mode, rate, half):
    p, pi, y, times, released, prior = args
    reference = p[:, -1] / pi[:, -1]
    reference /= reference.sum(1, keepdims=True)
    gap = np.sort(reference, axis=1)[:, -1] - np.sort(reference, axis=1)[:, -2]
    contexts = 3 * reference.argmax(1) + (gap >= .1) + (gap >= .25)
    decisions = (p / pi).argmax(2) - 1
    initial = np.r_[np.full(8, 1 / 16), .5]
    output = []
    for i, t in enumerate(times):
        chosen = np.flatnonzero(released <= t)
        losses = [(decisions[j] != y[j]) * prior.min() / prior[y[j] + 1]
            * rate * 2 ** (-(int(t) - int(times[j])) / (1000 * half)) for j in chosen]
        global_loss = np.sum(losses, axis=0) if losses else np.zeros(9)
        local_loss = sum((loss for j, loss in zip(chosen, losses, strict=True) if contexts[j] == contexts[i]), start=np.zeros(9))
        a = initial * np.exp(-global_loss)
        a = .95 * a / a.sum() + .05 * initial
        b = initial * np.exp(-local_loss)
        b = .95 * b / b.sum() + .05 * initial
        output.append(a if mode == "global" else (a + b) / 2)
    return np.asarray(output)


def test_delayed_discounted_updates_match_independent_direct_sum_for_every_case():
    args = inputs()
    for case, settings in CASES.items():
        result = mix_experts(*args, case=case)
        if settings:
            expected = explicit_weights(args, *settings)
            np.testing.assert_allclose(result["weights"], expected, rtol=0, atol=5e-15)
            np.testing.assert_allclose(result["probabilities"], np.einsum("im,imc->ic", expected, args[0]), rtol=0, atol=5e-15)
            np.testing.assert_array_equal(np.cumsum(result["updates"]), [(args[4] <= t).sum() for t in args[3]])
        elif case == "frozen_reference":
            np.testing.assert_array_equal(result["probabilities"], args[0][:, -1])
        elif case == "uniform_eight":
            np.testing.assert_allclose(result["probabilities"], args[0][:, :8].mean(1), rtol=0, atol=3e-16)


def test_unreleased_labels_future_forecasts_and_prefix_truncation_cannot_change_past():
    args = inputs()
    limit = 18
    for case in CASES:
        before = mix_experts(*args, case=case)
        changed = [a.copy() for a in args]
        unavailable = args[4] > args[3][limit - 1]
        changed[2][unavailable] = (changed[2][unavailable] + 2) % 3 - 1
        changed[0][limit:] = np.roll(changed[0][limit:], 1, axis=2)
        changed[1][limit:] = np.roll(changed[1][limit:], 1, axis=2)
        after = mix_experts(*changed, case=case)
        short = mix_experts(*(a[:limit] for a in args[:5]), args[5], case=case)
        for key in ("probabilities", "weights", "updates", "contexts"):
            np.testing.assert_array_equal(before[key][:limit], after[key][:limit])
            np.testing.assert_array_equal(before[key][:limit], short[key])


def test_confidence_context_class_and_margin_semantics_and_invalid_contracts():
    p = np.array([[.5, .4, .1], [.5, .25, .25], [.1, .2, .7], [.2, .5, .3]])
    prior = np.full_like(p, 1 / 3)
    actual = confidence_context(p, prior)
    scores = p / prior
    scores /= scores.sum(1, keepdims=True)
    expected = []
    for row in scores:
        gap = sorted(row)[-1] - sorted(row)[-2]
        expected.append(3 * int(row.argmax()) + (1 if gap >= .1 else 0) + (1 if gap >= .25 else 0))
    np.testing.assert_array_equal(actual, expected)
    args = inputs()
    for index, replacement in [(3, args[3].astype(float)), (4, args[3]), (0, args[0] * 2), (5, np.zeros(3))]:
        bad = list(args)
        bad[index] = replacement
        with pytest.raises(ValueError):
            mix_experts(*bad, case="global_eta03_h300")
