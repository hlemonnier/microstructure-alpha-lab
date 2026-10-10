# Quote-block renewal hypothesis and failure contract

This is a new, unvalidated modeling hypothesis prepared while the fixed midpoint experiment runs. No predictive result is implied. Any market experiment requires its own frozen inputs, procedures and controls; this note does not authorize independent confirmation.

## Observation and target

Use the existing publisher-time bookTicker observations, grouped by equal timestamp in update-ID order. Keep the first and last BBO in each group. The feature quote is the last quote strictly before decision d; the entry/exit quotes are the first quotes at or after d+100 ms and d+5100 ms. Grouping must preserve those three different selections. Published updates are not individual orders, cancellations or executions.

The prototype requires strictly positive spreads, matching the existing `features.iter_quote_events` raw iterator. A locked quote (ask equals bid), crossed quote or invalid quantity rejects the whole input attempt; never delete that row or replace its forecast. The canonical label function has a broader domain that accepts locked quotes, and remains unchanged. Market feasibility must verify this narrower source domain before any screen can retain the required full row universe.

For a block at t, let S_last be bid+ask for its last quote. The next observed block provides the joint mark (S_first_next-S_last, spread_first_next, S_last_next-S_last, spread_last_next, next_state) and positive integer-millisecond duration G. Preserve the joint mark instead of independently sampling its coordinates. State is at most four training-derived spread bins times eight imbalance categories (seven equal-width bins on [-1,1], plus a separate zero-total-depth category). Fit each asset separately. Sparse or absent states use a disclosed pooled kernel; never drop query rows.

The forecast integrates over future blocks, signed jumps and reversals. Label each path using the canonical exact decimal comparison D=S_exit-S_entry against T=max(spread_entry, 2*minimum_tick), with strict inequalities and neutral ties. Entry spread is sampled, and the two nominal clocks stay fixed. A single block crossing both clocks supplies the same first quote twice and gives a neutral outcome.

## Minimal parametric law

For state z, use two geometric duration components with component-specific empirical joint mark probabilities:

Q_z(g,m) = sum_r w_zr * p_zr * (1-p_zr)^(g-1) * F_zr(m), for g >= 1.

Both duration and mark enter complete-transition responsibilities. A right-censored interval ending at exclusive boundary E, following last observed block t, supplies P(G >= E-t), with exponent E-t-1. Do not join file/coverage boundaries as observed transitions. Each first block initializes a state without inventing a preceding duration.

Use an explicitly regularized EM objective: Beta(2,2) priors on each p, Dirichlet(2,2) on weights, and an emission prior with coefficients equal to one unit of the state's empirical mark distribution. If A_r is summed completed responsibility and B_r is summed responsibility-weighted (g-1), including censored (c-1), then p_r=(A_r+1)/(A_r+B_r+2). Weight updates add one to each component count. Emission updates add the empirical base mass once, then normalize. This is penalized likelihood, not unregularized MLE. Implement stable log probabilities and verify the penalized objective does not decrease beyond a stated numerical tolerance. Fix initialization and iterations before market fitting. Preserve degenerate/sparse behavior explicitly.

For initial age a=d-t_last >=1, lack of any strictly earlier new block conditions on G>=a. Initial component weights are proportional to w_zr*(1-p_zr)^(a-1). Given the component, the residual duration measured from d is Geom(p_zr)-1: a block exactly at d is possible but unobserved. After each block, draw a new component from the next state's weights and use a positive full duration. Cache query-independent paths per initial transition kernel/component; age changes mixture weights only. Two initial states can share a cache only when their complete transition kernels coincide. Later state transitions always follow the sampled mark. No future quote or realized future regime enters inference.

The target and stopping rule are exact, but the conditional law is approximate: coarse-state Markov renewal, additive innovations, the explicit boundary rule below, stationary dynamics over each training window and a two-component duration family are assumptions requiring empirical comparison. No jump, dwell, path count or probability tail may be clipped silently.

## Positive-price support amendment, before implementation

Unconstrained additive paths can generate nonpositive quotes. Multiplicative ratios preserve positivity but change additive threshold point masses when transported to a different price; exact rational products also create a numerical-cost problem. The selected prototype instead defines an explicit reflected additive law. The floor is a modeling assumption, with upward pressure near zero, not an empirically established exchange mechanism. It does not reject or renormalize simulated paths.

Let Q be a common integer denominator of the training primitive quote prices and minimum tick, interpreted as exact shortest-decimal fractions. Set the strictly positive bid floor u=1/(2Q). Every observed positive training price is then strictly above u. This is a computational quantum, not an assumed exchange tick grid. Preserve query prices as exact fractions; an initial bid below u fails the whole attempt without deleting the query. Do not round a query to the training quantum.

Process each sampled block as two ordered substeps. Its first displacement is delta_first from the preceding last quote; its second displacement is delta_last-delta_first. Thus its unreflected first and last sum displacements are C_previous_last+delta_first and C_previous_last+delta_last. Capture the first endpoint before processing the same block's last quote. The two substeps share publisher time, but their order is retained.

For substep k, let C_k be cumulative unreflected sum displacement, s_k its strictly positive sampled spread, and c_k=s_k+2u. With initial sum S0, define

    R_k = max_{j<=k}(c_j-C_j)
    L_k = max(0, R_k-S0)
    S_k = S0+C_k+L_k.

Then S_k>=s_k+2u, so bid=(S_k-s_k)/2>=u and ask=bid+s_k>bid. The nondecreasing push L persists after a boundary contact. It is the minimal push keeping every generated substep inside this domain. Absolute spread and imbalance states are invariant to the common price translation, so the original joint-mark transition rule remains well defined. This is a discrete application of the one-sided reflection map described by [Kruk et al., equation 1.1](https://www.math.cmu.edu/users/shreve/DoubleReflection.pdf); applying it to these BBO marks is our modeling choice.

For each sampled path retain C_entry, C_exit, R_entry, R_exit and s_entry. The exact target numerator is

    D = C_exit-C_entry + max(0,R_exit-S0) - max(0,R_entry-S0),
    T = max(s_entry, 2*minimum_tick).

All comparisons are exact rational/integer comparisons. When S0>=R_exit, the additive path's original threshold ties are unchanged. Otherwise evaluate the reflected expression without dropping its mass. Every component uses its full simulation count as denominator, including ties and paths where both clocks select the same first quote. Record the query-specific reflected fraction as well as zero physical-support violations. This establishes support for the defined proxy law; it does not establish realistic behavior near its floor or a fully exchange-grid-aligned process.

Training uses observed first and last quotes strictly above u. A path that receives a push at either observed substep has bid=u at that substep and therefore cannot produce that interior observation. Consequently the complete observed transition likelihood retains the stated joint-mark form. No hidden intrablock path beyond the retained first/last substeps is claimed. Censoring changes duration exposure only, with no emission contribution.

There are at most 5,101 sampled blocks through the first block at/after d+5,100 ms: the initial residual may be zero, later gaps are positive integer milliseconds. This is an exact stopping bound, not a truncation rule. Validate integer bounds before vectorized arithmetic; use exact arithmetic or abort with a preserved receipt if a finite resource ceiling is exceeded. Never wrap, round, discard paths or keep a partial cache. The small synthetic implementation must pass before a separate training-only feasibility protocol is considered.

## Failure cases before implementation

Wrong first/last quote for tied timestamps; features at d instead of strictly before it; an exit shifted by entry overshoot; ignoring initial age; equality errors in duration survival; retaining the initial component after subsequent blocks; directional outcomes when one block crosses both endpoints; omitted reversals; current spread substituted for random entry spread; rounded prices changing ties; duplicate/out-of-order IDs; crossed/nonfinite prices or invalid quantities; right-censor likelihood omitted or off by one; fabricated cross-file transitions; impossible/sparse states without explicit backoff; decreasing EM objective; invalid or lost probability mass; forgetting pre-entry reflection, failing to carry its push, or letting a same-block last quote alter a first endpoint; finite simulation bias hidden by clipping; query-dependent randomness, future perturbation, serialization or batch partition changing forecasts; overflow; and silent resource-limit violations.

## E2E required before market assessment

Write the source-shaped synthetic ZIP workflow and independent expected endpoint ledger before the implementation. Cover same-timestamp first/last differences, exact thresholds, a multi-step reversal, delayed first quotes crossing both clocks, file censoring, malformed input and variable entry spread. Independently compare direct sequential reflection with the cached C/R formula, including pre-entry and post-entry contacts, changing spreads, near-floor and off-grid queries, ties, and a single block serving both stops. Verify fit/save/load/forecast through the CLI, finite normalized natural probabilities, deterministic prefix and partition equivalence, and a small kernel with independently calculated probabilities. Record source hashes, synthetic tapes, exact endpoint IDs/labels, fitted parameters, objective trajectory, censored/backoff counts, forecasts, resource measurements and a manifest. No unit tests.

Measure simulation convergence using independent deterministic seeds and doubled path counts; compare distributions and decisions before looking at market labels. Freeze a numerical tolerance and finite resource ceiling in a separate preflight before market fitting. Failure to meet that ceiling is a computational limitation, not permission to truncate tails. Keep only one substantial market learner active.

## Research basis

[Cont and de Larrard](https://arxiv.org/abs/1104.4596) derive state-dependent move and duration quantities in a stylized queueing model. [Huang, Lehalle and Rosenbaum](https://arxiv.org/abs/1312.0563) model order-flow rates conditional on book state. The [Swishchuk–Vadori abstract](https://epubs.siam.org/doi/abs/10.1137/15M1015406) extends durations and dependence using Markov renewal processes, while retaining restrictive order-book assumptions. These papers motivate a mechanism; none establishes gains for this observed-BBO, five-second crypto target. The proposed dense transition supervision and exact-clock integration are our hypothesis, not a reported replication.
