# Delayed expert combinations: no substantial gain

The best registered adaptive combination reaches **57.3647% balanced accuracy**, only **+0.0239 percentage points** above the retained reference. BTC worsens and June 11 loses 0.3901 points. This is a small, selected development difference, not a substantial discovery or independent confirmation.

All 42,420 original rows are retained. Forty-eight adaptive coefficient trajectories and eighteen static controls produce 132 case/policy/asset/date panels on the already exposed June 3, 7 and 11 dates. No base model is refitted. All six asset/date panels were saved and verified before any family score was computed.

## Fixed comparison

These figures use the existing causal one-hour forecast-prior policy. The complete evidence also includes every registered-prior result.

| Case | Learning rate | Loss half-life | Balanced accuracy | Natural log loss |
|---|---:|---:|---:|---:|
| Frozen reference | — | — | 57.3408% | 0.762206 |
| Uniform eight individuals | — | — | 57.3477% | 0.760213 |
| Fixed reference-centered prior | — | — | 57.3429% | 0.760668 |
| Global weights | 0.03 | 300 s | 57.3331% | 0.760602 |
| Global weights | 0.03 | 1,800 s | 57.3455% | 0.760465 |
| Global weights | 0.10 | 300 s | 57.3312% | 0.760545 |
| Global weights | 0.10 | 1,800 s | 57.1792% | 0.760565 |
| Confidence contexts | 0.03 | 300 s | 57.3326% | 0.760625 |
| Confidence contexts | 0.03 | 1,800 s | 57.3547% | 0.760524 |
| Confidence contexts | 0.10 | 300 s | **57.3647%** | 0.760555 |
| Confidence contexts | 0.10 | 1,800 s | 57.2951% | 0.760413 |

The best case gains 0.0170 points over uniform averaging and 0.0217 points over its fixed initial mixture. It gains 0.0335 points over the matched global rule, with both assets positive in that comparison but June 3 negative. This small context benefit does not establish a robust gain over the retained reference.

| Best adaptive case versus frozen reference | Balanced-accuracy difference |
|---|---:|
| BTC | −0.0801 pp |
| ETH | +0.1278 pp |
| June 3 | +0.2005 pp |
| June 7 | +0.2612 pp |
| June 11 | −0.3901 pp |

Its natural log loss is 0.2166% lower than the reference, but slightly worse than the uniform mixture. The substantial independent-gain gate remains unmet. The ordered/Langevin study is still running with its scores sealed; these results do not claim superiority to that unfinished family.

## What was tested

The nine experts are exactly the eight individual forecasters and retained blend in the [error-overlap diagnostic](boundary_error_geometry_report_20260908.md). The initial mixture gives half the weight to the retained blend and one-sixteenth to each individual. A five-percent mixture with that initial distribution remains in every adaptive weight vector.

For each expert, a released classification error receives weight `min(training_prior) / training_prior[realized_class]`, keeping the loss in [0,1]. Losses decay from their original decision time, but are added only when the exact future quote has actually released their label. The learner processes out-of-order releases exactly once. This is a historical-prior proxy for class-balanced error, not an exact current balanced-risk objective under class-prior shift.

Global exponential weights use those accumulated losses. Context variants also maintain nine sets of losses, indexed by the reference's predicted class and its normalized decision-score margin in three fixed bins. Their weights average the global and matching-context distributions. The final forecast is an arithmetic mixture of unchanged natural posterior vectors; the ordinary causal forecast-prior policy is then recomputed for that mixture.

[Freund and Schapire](https://www.sciencedirect.com/science/article/pii/S002200009791504X) motivate exponential expert weighting, and [Joulani et al.](https://proceedings.mlr.press/v28/joulani13.html) motivate explicit treatment of delayed feedback. This procedure uses full-information supervised outcomes. It is neither AdaBoost retraining nor a bandit/RL policy. Classical regret claims do not automatically apply to these discounted contexts, proxy class weights, dependent data and deterministic argmax-mixture decisions.

The experiment tests a narrow, fixed family of coefficient rules. It does not rule out a learned feature-dependent gate, historical cross-fitted stacking, different observations or other online losses. None of those alternatives is demonstrated by this result.

## Verification and cost

The full-length synthetic preflight passed all eleven cases. Its longest checked trajectory took 0.4432 seconds, with 126,238,720 bytes maximum sampled process RSS. The market study took 27.2913 seconds in total; its longest trajectory took 0.4611 seconds and peak sampled process RSS was 192,053,248 bytes. It used one CPU thread and stayed below the prospectively fixed 60-second trajectory deadline and 1 GiB RSS limit. The separate large boosting fits remained unchanged.

There were 339,072 released loss-vector updates across the adaptive trajectories. Unreleased labels and future expert forecasts cannot alter the saved prefix. Full repeated trajectories and saved/reloaded arrays are exact, as are the frozen-reference probabilities and both original decision policies. All row, label, timestamp, historical-prior and actual-release checks pass; every actual release is at least 5,100 ms after its decision. All source and completed artifact hashes were reverified for the report.

Three targeted mathematical tests pass, including independent explicit discounted sums, out-of-order feedback, exact availability boundaries, prefix causality, context semantics and invalid inputs. The dependency-complete suite passes 663 direct cases with no optional skips. The minimal environment passes 488, skipping six optional tests and 57 optional modules. Ruff and compilation pass. The [definition](boundary_expert_feedback_definition_20260908.json), [synthetic evidence](boundary_expert_feedback_preflight_evidence_20260908.json), [market protocol](boundary_expert_feedback_screen_20260908.json) and [complete results](boundary_expert_feedback_evidence_20260908.json) preserve the full experiment.
