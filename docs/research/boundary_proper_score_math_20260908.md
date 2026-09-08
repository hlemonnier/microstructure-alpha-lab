# Proper scoring losses for the fixed forecast target

This is a mathematical definition for a prospective experiment, not a market result. The network's balanced probability is always `q = softmax(u)`. Natural probabilities retain the existing conversion `p_c ∝ q_c π_c`. Four losses share this probability target: log loss, Brier loss, and pseudospherical losses with powers β = 2 and 4.

## Definition and population target

For K classes, β > 1, α = (β−1)/β and A = K^α/(β−1), define

    Lβ(q,y) = A [1 − q_y^(β−1) / ||q||β^(β−1)].

These are scaled, negatively oriented [pseudospherical scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). β = 2 is the spherical score. The scale A is positive, so it does not change the probability minimizing population risk.

For true class probabilities η, Hölder's inequality gives

    Σ_c η_c q_c^(β−1) ≤ ||η||β ||q||β^(β−1).

Equality requires η and q to be proportional on their support. On the probability simplex this means q = η; mass assigned outside η's support cannot improve the score. Consequently the risk is uniquely minimized at q = η, with Bayes risk A(1−||η||β). Finite logits describe the simplex interior; distributions on its boundary are reached as limits.

For the normalized Brier loss

    LB(q,y) = (K/2) Σ_c (q_c − 1[c=y])²,

the excess population risk is exactly `(K/2)||q−η||₂²`. Log loss has KL excess risk. All four losses therefore elicit the same population probabilities. [Composite-loss theory](https://www.jmlr.org/papers/v17/14-294.html) distinguishes that property from convexity in a chosen parameterization. None of these facts proves that a regularized finite network reaches its population optimum or is calibrated on market data.

## Stable logits and matched local scale

Let `s = softmax(βu)`. Then

    s_y^α = q_y^(β−1) / ||q||β^(β−1),
    Lβ(u,y) = −A expm1(α log_softmax(βu)_y),
    ∇u Lβ = K^α s_y^α (s − e_y).

Using `log_softmax` and `expm1` avoids explicit small probability powers and cancellation near a correct confident prediction. At uniform predictions, the gradient is exactly `1/K − e_y`, matching log loss. The K/2 Brier scaling gives the same gradient there because the uniform softmax Jacobian acts as 1/K on zero-sum vectors.

This is one local gradient match. It does not equate gradients elsewhere, loss curvature, optimization paths or the effect of weight decay. Pseudospherical and Brier logit gradients can become small for confidently wrong predictions. That can reduce their influence, but it can also prevent recovery from a bad solution. No corrupted-label model is assumed for stochastic market outcomes.

## Relation to generalized cross entropy

[Generalized cross entropy](https://papers.nips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html) in probabilities s has loss `(1−s_y^α)/α`. Its population optimizer for 0 < α < 1 satisfies `s_c ∝ η_c^(1/(1−α))`. Reporting s itself as η would change probability semantics.

Here β = 1/(1−α), and the model reports `q = softmax(u)` while the training score evaluates `s = softmax(βu)`. At the population optimum q = η. This is the known pseudospherical rule, not a new probability calibration method. No temperature or inverse link is fitted on assessment outcomes.

The endpoint α = 1 is deliberately excluded: its linear objective selects a class mode and ceases to identify the full probability vector. Increasing β indefinitely is not used as a shortcut to a claimed probability forecast.

## Existing class weights

A row in asset/class cell (a,c) keeps weight `N/(6 n[a,c])`. Conditional on x and a, its ideal weighted posterior is proportional to the natural posterior divided by that asset's historical class prior. Strict propriety applies to that weighted distribution. Multiplying the learned balanced probabilities by the original prior therefore retains the same ideal natural-posterior convention as the log-loss model. Finite-sample and temporal distribution changes still need empirical assessment.

The [prospective definition](boundary_proper_score_definition_20260908.json) fixes the architecture, historical rows, optimizer, seeds, checkpoint selection and both decision policies. Its six freshly trained log-loss controls must reproduce the existing original neural state dictionaries and forecasts exactly before any assessment result is opened.
