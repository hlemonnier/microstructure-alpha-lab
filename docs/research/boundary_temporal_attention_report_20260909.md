# Bounded temporal attention: no predictive promotion

The best new procedure reached **57.246038% balanced accuracy**, below the retained complete procedure's **57.340789%** by **0.094750 percentage points**. The separately evaluated expert-feedback procedure remains higher at **57.364662%**. These are development results on the repeatedly exposed June 3, 7 and 11, 2023 dates, with 42,420 assessment rows per forecast source. They do not establish independent performance.

| Procedure, common `forecast_3600` decision rule | Balanced accuracy | Natural log loss |
| --- | ---: | ---: |
| Retained combined instantaneous HGB / original neural blend | 57.340789% | 0.762205735 |
| Best new: observations uniform-history / deep HGB blend | 57.246038% | 0.761148722 |
| Observations content-attention / deep HGB blend | 57.233958% | 0.761149538 |
| Matched observations instantaneous / deep HGB blend | 57.233644% | 0.761243047 |
| Separate retained expert-feedback procedure | 57.364662% | 0.760554967 |

The best new model improves BTC by 0.024250 points but loses 0.213750 points on ETH against the retained complete procedure. Its date-level differences are -0.088852, +0.079996 and -0.275394 points. Its log-loss ratio is 0.998613. Against the separate expert procedure, balanced accuracy falls 0.118623 points and the log-loss ratio is 1.000781. No new model is promoted.

The mechanism test fitted 24 models: two input representations, four residual branches and three chronological folds. Each branch conditions a frozen original model on the current observation and up to 127 preceding eligible one-second observations. The branches were instantaneous, uniform-history, learned age weights, and content-dependent attention with four heads. Training, validation, resource limits, fixed blends and the common primary decision rule were registered before market execution. This is a bounded residual-attention experiment, not a reproduction of a full Transformer or TLOB.

The temporal mechanism itself provides little support: the observations content-attention model loses 0.070314 points against its matched instantaneous model before blending. Uniform history loses 0.097046 points. The small advantage of the best uniform-history blend over the matched instantaneous blend is only 0.012394 points. These results do not justify expanding this particular family to fresh confirmation dates.

All 24 workers completed, all three fold inventories and all 1,119 registered input hashes verified, all original parameters and priors remained exact, and all saved-checkpoint forecasts reproduced exactly. All 3,072 retained metrics reproduced exactly. Assessment labels were not decoded by training workers. The largest query-partition probability difference was 4.307084e-8 against the frozen 2e-5 tolerance; future-prefix and shorter-prefix differences were zero. There were 24,336 optimizer steps, a maximum sampled worker RSS of 1,956,937,728 bytes, and 1,350.999 seconds summed across supervised worker processes. The CPU preflight passed; MPS was unavailable, so CPU/GPU agreement was not measured.

Legacy registered decision priors remain method-specific. New branches use historical training priors for their registered rule; primary comparisons above use the common causal forecast-prior rule. The other 17 exposed development dates and the 20 reserved successor dates remain unmeasured for these new attention models. No execution or economic gain is inferred.

The next hypothesis concerns additional information: official OKX historical depth contains price-level order counts as well as quantities. Its predictive contribution must be tested against quantity-only features from the same venue, with explicit time alignment and missing-book coverage. Source availability alone is not evidence of accuracy improvement.

Reproduction records:

- Protocol: `docs/research/boundary_temporal_attention_screen_20260908.json`, SHA256 `e5aea285da9643b20020284106f038ed31ad7c75a3c113246e2dd07f1883c33c`.
- Full result: `results/boundary_temporal_attention_screen_20260908/summary.json`, SHA256 `37074823e91c6dc89627fe53673fb1e9a9c0790fdc85e5f9046cc4f6a5481214`.
- Tracked evidence: `docs/research/boundary_temporal_attention_evidence_20260909.json`, SHA256 `071910e94556153a6b6e8206a42e24fdea11655726ef1352f2dcfa5f5acaafce`.

Architecture references: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [TLOB](https://arxiv.org/abs/2502.15757). They motivate the mechanism comparison; their published task results are not transferred to this benchmark.
