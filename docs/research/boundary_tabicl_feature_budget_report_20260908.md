# Historical feature budget and larger transformer context

The completed experiment does **not** improve the strongest retained development procedure. The best new blend reaches **56.9629%** balanced accuracy, compared with **57.3408%** for `combined_instant_hgb_old_neural`: **−0.3779 percentage points**, with losses on both assets and all three dates. No independent confirmation is triggered.

The useful component result is narrower: increasing the selected transformer context from 3,072 to 12,288 rows gains **0.3979 points**, positive on both assets and all three dates. The equally enlarged HGB context gains **2.0852 points**. More context helps, but this intervention does not establish a transformer advantage over the complete existing procedure.

## Frozen comparison

The [official TabICL documentation](https://github.com/soda-inria/tabicl#faq) reports pretraining with 2–100 columns. The previous local study supplied 220 or 399 market features. This motivated a [prospective definition](boundary_tabicl_feature_budget_definition_20260908.json) with 96 original observation features plus an asset indicator. The [model paper](https://arxiv.org/abs/2602.11139) supplies the pretrained method, not evidence of a market gain.

One subset is random; the other uses balanced historical feature/class information followed by a redundancy filter. The supervised selector consumes **all historical labels**, even with a small downstream context. Both sampled transformer/HGB members receive the same selector and rows. Selected-versus-original transformer comparisons therefore do not hold total label use fixed.

The 12,288-row context contains the exact original 3,072 rows and adds a uniform sample within each asset/class cell. Every model uses historical dates d−5 through d−2. The full neural model chooses its epoch on previous-day noon. No assessment or same-day label enters fitting or selection. Both original decision policies are retained; `forecast_3600` remains primary.

All three exposed dates—June 3, 7 and 11, 2023—completed before scores were opened. The family contains nine transformer context procedures, nine sampled HGB fits, three full-data HGB fits and three full-data neural fits. There are 184 retained/new sources, 2,208 source/policy/asset/date panels, and **42,420 unchanged observations per source**.

## Primary results

| Historical inputs | Transformer balanced accuracy | Matched HGB balanced accuracy |
|---|---:|---:|
| Random 96 features; 3,072 rows | 55.5610% | 53.2962% |
| Selected 96 features; 3,072 rows | 55.9257% | 54.0167% |
| Selected 96 features; 12,288 rows | 56.3236% | 56.1019% |

The full-data selected HGB scores 56.5195%; its selected-feature neural counterpart scores 56.0621%. Both fall below their corresponding original full-feature models. The best new transformer blend, with `deep500_hgb`, scores **56.9557%**, still below the retained control. Its natural log loss is 0.761659; a lower log loss by itself does not establish calibration or a substantial accuracy gain.

The transformer context expansion gains 0.5894 points on BTC and 0.2065 on ETH. Date-average gains are 0.4414, 0.4059 and 0.3465 points. The best new complete blend loses 0.3076 points on BTC and 0.4481 on ETH; its date losses are 0.3632, 0.2668 and 0.5036 points. These are descriptive development comparisons, not independent inferential results.

## Verification and artifacts

All original context rows, labels, timestamps, priors and retained forecasts agree exactly. Every saved model reproduces its full forecasts exactly. Maximum future-prefix error is zero; the largest alternate-query-partition error is 3.162e−6. Maximum sampled worker RSS is 8,708,669,440 bytes. The longest fit/save/reload is 262.11 seconds and the longest complete worker is 511.66 seconds. These research-worker timings include caching and verification; they are not live trading latency measurements.

The [preflight evidence](boundary_tabicl_feature_budget_preflight_evidence_20260908.json) records both failed GPU memory attempts and the accepted CPU backend. The [final protocol](boundary_tabicl_feature_budget_screen_20260908.json) pins 752 inputs, with SHA256 `c210db0b2ca0207c6d0a12ed31329f33961f70dfa3b65cf6e5f0af2a2c51c259`. The local summary is `results/boundary_tabicl_feature_budget_screen_20260908/summary.json`, SHA256 `17bdeed29787c3f242b80d59e5eb2d14612a10cc19e34779fb87f61d025de27f`. The [tracked evidence](boundary_tabicl_feature_budget_evidence_20260908.json) contains all comparisons and worker records. Large data/checkpoints remain local and ignored by Git.

The subsequent retrieval family was defined and implemented before this score reveal. Its separate temporal exclusion and weighted-neighborhood objective address a different hypothesis; this negative result is not used to retune the finished transformer experiment.
