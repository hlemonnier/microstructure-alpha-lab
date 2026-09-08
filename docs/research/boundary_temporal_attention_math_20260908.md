# Causal finite-window attention: implementation and mathematical checks

The [prospective definition](boundary_temporal_attention_definition_20260908.json) fixes four temporal aggregation branches before implementation and before the ongoing CatBoost scores were opened. The implementation passes eight targeted tests and 671 dependency-complete direct test cases. These are correctness results. No market attention model, production-size preflight or MPS case has run, and no predictive gain is established.

## Forecast definition

Let x[t] be the original normalized feature vector, including the asset coordinate. Define e[t] = GELU(Ws x[t] + bs) and v[t] = Wv e[t] + bv. The allowed set J[t] contains t and at most 127 preceding one-second observations from the same contiguous asset/day segment. A missing observation or date boundary cuts off the history. Each x[t] retains the original engineered feature lookbacks; 128 limits the explicitly stored rows, not the total raw-data lookback.

The instantaneous control uses v[t]. The uniform control averages v[j] over J[t]. The age and content branches use four heads of sixteen coordinates:

    score[h,t,j] = age_bias[h,t-j]                                  (age)
    score[h,t,j] = query[h,t] · key[h,j] / sqrt(16) + age_bias[h,t-j] (content)
    weight[h,t,j] = exp(score[h,t,j]) / sum(k in J[t]) exp(score[h,t,k])
    aggregate[h,t] = sum(j in J[t]) weight[h,t,j] value[h,j].

The content query and key maps are separate affine projections of e. Age biases start at zero, with lag zero referring to the current row. Consequently the age branch initially agrees exactly with the uniform control when their common weights match. This uses the [scaled dot-product and multiple-head construction](https://arxiv.org/html/1706.03762v7), restricted to a finite residual readout; it does not implement the full Transformer encoder/decoder.

Concatenate the current e[t] and the head aggregates, then apply the fixed 128 → 128 → 64 → 3 GELU readout to obtain delta[t]. The final affine head starts at zero. Common layers initialize before architecture-specific layers so their initial values match under the fixed seed.

For the original balanced logits z and asset prior pi, report

    p[c] = exp(z[c] + delta[c]) pi[c] / sum(k) exp(z[k] + delta[k]) pi[k].

The implementation evaluates this normalization stably. For an exactly zero correction, it returns the original saved probabilities bit for bit rather than recomputing a numerically different representation. Original network parameters, quantile transformation, active columns and priors remain frozen. Historical original logits are in-sample, not out-of-fold predictions; all branches share that limitation.

## Training objective and reuse of historical projections

In one 512-second block, let S be the original stride-four supervised asset/time rows and N the total original historical sample size. A row in historical asset/class cell (a,c), containing n[a,c] rows, keeps weight N/(6 n[a,c]). The block objective is

    loss = (1 / |S|) sum((a,t) in S) N/(6 n[a,y[a,t]]) CE(z[a,t] + delta[a,t], y[a,t]).

The denominator is the supervised row count across all eight streams, not the sum of weights or the number of streams. Empty blocks cause no optimizer update. Historical class counts remain global; they are not recomputed within each block.

The at-most-127-row prefix is recomputed using current weights before each update. An input projection used by several query windows contributes to each query's gradient. The independent test explicitly constructs one window per supervised query and compares every parameter gradient against the shared batched calculation, including unequal weights and gaps. Agreement is checked in float64 with absolute tolerance 2e-12 and relative tolerance 1e-10. No recurrent state, transformed-key cache or gradient is carried across optimizer updates.

Six chronological epochs use AdamW with learning rate 0.0003, weight decay 0.01 and gradient norm limit 5. The preceding complete noon selects by equal-asset balanced accuracy using original priors, then natural log loss, then earliest epoch. Current assessment outcomes are absent from the fitting API. The small pipeline test uses two epochs on 96 synthetic supervised rows and verifies that every original target contributes once per epoch and the original network stays unchanged.

## Causality, numerical masking and persistence

Only current and earlier keys enter a query. Segment identifiers must increase at each new observed segment, preventing a reused identifier from reconnecting history across a gap. Query chunking includes the required prefix and therefore must preserve predictions within the registered floating-point tolerance.

An entirely missing query temporarily receives one masked-safe key to avoid an all-negative-infinity softmax; its final correction and parameter-gradient contribution are zero. An extra stress probe exposed an implementation defect: finite values of 1e25 in excluded cells could overflow before score masking and yield nonfinite gradients despite a finite loss. Missing inputs are now cleared **before** projection. A regression checks identical finite losses and gradients with ordinary versus extreme excluded values in all four branches. Observed nonfinite inputs and nonfinite corrections fail explicitly. This synthetic failure did not involve normalized market inputs or a market fit.

Tests also cover independent-window gradients, future-prefix invariance, chunk sizes 1/7/13, sparse queries, gap resets, cross-stream isolation, lag orientation, all-missing queries, exact zero corrections, unchanged original parameters, and bitwise save/load replay after nonzero updates. Production checkpoints use float32 parameters and weights-only loading. The targeted suite passes eight tests in 5.64 seconds; the final full suite passes 671 cases. The isolated minimal environment passes 488 cases and explicitly skips six optional tests and 58 optional modules.

The initial full-suite attempt failed because its direct test launcher does not supply pytest's monkeypatch fixture. Using a scoped unittest.mock.patch.dict in the tiny fit test resolved this runner incompatibility. That failure, the subsequent 670-case pre-fix run and the final 671-case run are separately logged and hashed in the [implementation evidence](boundary_temporal_attention_implementation_evidence_20260908.json).

## Capacity and scope of the evidence

The common parameter count is 64D + 29,187. Age adds 512 parameters; content adds those 512 plus 8,320 query/key parameters. These are nominal counts, not equal functional capacity or runtime.

| Normalized width D | Instant | Uniform | Age | Content |
| --- | ---: | ---: | ---: | ---: |
| 215, actual observation schema | 42,947 | 42,947 | 43,459 | 51,779 |
| 221, first preflight upper bound | 43,331 | 43,331 | 43,843 | 52,163 |
| 400, second preflight upper bound | 54,787 | 54,787 | 55,299 | 63,619 |

A small associative task gives a current query, four historical keys and their independent three-class values. The content branch learns to retrieve the matching value and exceeds 90% accuracy on 256 new synthetic examples after training on 384 examples. This confirms that the query-dependent mechanism can learn under that synthetic construction. It supplies no market effect size, matched-control superiority, backend agreement or production runtime evidence.

[TLOB](https://arxiv.org/abs/2502.15757) motivates comparing attention with a strong MLP, but uses other data, labeling and architecture. Our construction is not a TLOB replication. [Bilokon and Qiu](https://arxiv.org/abs/2309.11400) report mixed Transformer/LSTM comparisons; architectural novelty alone does not establish a forecasting advantage.

The registered execution order remains CatBoost, proper-scoring losses, then the twenty-date anchor audit. A separately frozen full-size CPU/MPS preflight is still required before attention market fits. A candidate supported by that broader audit should reach fresh confirmation before adding this family. The twenty reserved successor dates remain opaque and require their own frozen model/evaluation protocol. Substantive accuracy, per-asset, log-loss and nominal uncertainty gates remain unchanged, followed by the separate economic and execution requirements.
