# Pretrained tabular transformer: completed development result

The corrected TabICLv2 experiment improves on the fixed tree trained with the same small context, but it does not beat the strongest existing full-data control. No new overall accuracy winner or independent gain is established.

All three exposed dates, June 3, 7 and 11, completed before score inspection. The experiment preserves 42,420 original decisions across BTCUSDT and ETHUSDT, the exact five-second three-class target, and both registered decision policies. The primary comparison below uses the existing prediction-only one-hour prior adaptation.

| Context / feature representation | Same-context HGB balanced accuracy | TabICLv2 balanced accuracy | Difference |
| --- | ---: | ---: | ---: |
| Four past days / observations | 53.6349% | 56.1056% | +2.4706 points |
| Four past days / combined venues | 53.8266% | 55.7567% | +1.9301 points |
| Same morning / observations | 53.9501% | 55.7530% | +1.8028 points |
| Same morning / combined venues | 54.2518% | 55.6268% | +1.3750 points |

Each contrast is positive on both asset averages and every date average. The HGB is the unchanged registered seven-leaf procedure, not a separately optimized small-data comparator. Both algorithms receive the same 3,072 uniquely sampled rows, balanced within six asset/class cells. Posterior recovery uses the full pre-sampling pool prior; assessment frequencies never enter it.

The best new fixed blend combines historical observations-only TabICLv2 with `deep500_hgb`: **56.8757%** balanced accuracy and **0.765379** natural log loss. The existing `deep500_hgb_old_neural` control remains stronger at **57.3074%** and **0.761756**, respectively. The new blend loses **0.4317 percentage points**. Its component advantage at a restricted context size must not be presented as an overall model improvement. Larger contexts are a plausible next experiment, not an established gain.

The [pinned official model and source](boundary_tabicl_definition_20260908.json) use two ensemble members on CPU. [TabICLv2's paper](https://arxiv.org/abs/2602.11139) motivates inference from tabular training examples, but published benchmark gains do not establish transfer to this noisy market target.

The initial attempt exposed batch-dependent power-transform fallback in upstream preprocessing. Its eight attempted procedures remain preserved: seven completed, and one failed before full forecasting. The [correction and diagnostic](boundary_tabicl_causality_fix_20260908.md) isolate fallback to each failing query row while preserving fitted transforms and weights. The complete restart adds twelve transformer context procedures and twelve matched HGB fits, giving **32 total market procedure attempts** including the first run. No first-run market forecasts are reused.

In the corrected run, changing later queries and shortening the query prefix produce zero earlier-probability differences. The largest single-query numerical difference is 0.000001968, below the registered tolerance. Eight actual assessment-query rows require the individual fallback. The largest measured worker memory is 8,034,910,208 bytes; maximum transformer fit/save/reload readiness is 111.876 seconds. These measurements do not certify live trading latency.

Reproduce with the [corrected frozen protocol](boundary_tabicl_screen_row_local_20260908.json) and its matching source hashes. The [machine-readable evidence](boundary_tabicl_evidence_20260908.json) binds the complete results to summary SHA256 `ab585c9f819184c9b83f8a8f9e6c30a60fe9f444dbc00fd25ed188f11d13b9de`. Three repeatedly exposed dates support development comparisons, not independent significance. Delayed supervised adaptation and continuous memory have separate frozen definitions and remain subsequent experiments.
