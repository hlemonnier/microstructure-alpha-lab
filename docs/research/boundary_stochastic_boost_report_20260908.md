# Ordered and Langevin boosting: completed development comparison

All 24 registered CatBoost fits completed. The best new source, combined-feature plain depth-six boosting, reaches **57.2061% balanced accuracy**, below the retained blend's **57.3408%** and the separately completed expert-combination rule's **57.3647%**. None of the new sources or fixed blends improves the accuracy leader. No substantial predictive or economic gain is confirmed.

The [evidence record](boundary_stochastic_boost_evidence_20260908.json) preserves every worker, all 2,784 panels and the paired asset/date comparisons. The [protocol](boundary_stochastic_boost_screen_20260908.json) was fixed before fitting; all three exposed dates completed before assessment scores were opened. There are 232 forecast sources and 42,420 unchanged assessment rows per source. The primary comparison uses the prospectively fixed forecast-prior rule with a 3,600-second half-life.

| Individual CatBoost branch | Observation features: balanced accuracy | Combined features: balanced accuracy |
| --- | ---: | ---: |
| Plain, depth six | 56.8331% | **57.2061%** |
| Ordered, depth six | 56.8765% | 57.1239% |
| Plain, depth eight | 56.9591% | 57.1825% |
| Langevin/shrinkage, depth six | 56.9338% | 57.1736% |

Every branch also has two fixed 50/50 blends, with its original neural model and with the existing deep500 HGB. All sixteen blends remain below the retained accuracy leader. The best new result happens to be an individual model, not one of those blends.

| Complete procedure | Balanced accuracy | Natural log loss |
| --- | ---: | ---: |
| Best new CatBoost: combined plain6 | 57.2061% | 0.7604951 |
| Retained instantaneous-feature HGB/neural blend | 57.3408% | 0.7622057 |
| Separate delayed expert-combination rule | 57.3647% | 0.7605550 |

The best new CatBoost's difference from the retained blend is **−0.1347 percentage points**: BTC −0.3476 and ETH +0.0782. Its date effects are −0.0347 on June 3, +0.0680 on June 7 and −0.4373 on June 11. Against the separate expert rule it loses 0.1586 points, with degradation on both assets and all three dates. These are descriptive comparisons on repeatedly exposed dates, not independent inference.

Within the observation representation, Ordered gains 0.0434 points, depth eight gains 0.1260, and Langevin/shrinkage gains 0.1007 against plain depth six. Those component gains do not survive as an overall lead. Within combined features, all three interventions reduce individual accuracy relative to plain depth six. That plain combined model gains 0.1606 points over its original combined HGB component, again without beating the stronger complete procedure.

Natural log loss presents a modest separate improvement: combined plain depth eight reaches 0.7594142, the best new value. It is not the primary-accuracy winner, and a small log-loss reduction does not satisfy the substantial predictive gate.

## Mathematical and execution scope

All variants use the same historical dates, original stride-four rows and historical-only quantile clipping. Native class indices 0/1/2 represent negative/neutral/positive outcomes. Each historical asset/class cell keeps the original N/(6n) row weights, and natural probabilities are recovered proportionally to native probabilities times the exact original asset prior.

The final tree count is exactly 1,000 for every fit. No validation or assessment outcomes select a tree count or a prefix checkpoint. Timestamp metadata and serialized native settings verify chronological ordering, CPU execution and two threads. Ordered boosting changes historical residual estimation; the [ordered-boosting paper](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) does not establish an unbiased or independent-data result for these overlapping market labels.

The Langevin branch uses the [documented native posterior-sampling settings](https://catboost.ai/docs/en/references/training-parameters/common): noise together with shrinkage, diffusion temperature equal to nominal row count and shrink rate 1/(2N). This is a composite intervention. Nominal rows are not independent market observations, and this finite multiclass experiment does not establish the posterior, mixing or global-optimum properties discussed by [Ustimenko and Prokhorenkova](https://proceedings.mlr.press/v139/ustimenko21a.html).

All 24 native checkpoints reproduce forecasts exactly. Future-prefix, shorter-prefix and single-query probability errors are zero. Every saved original control, label, decision clock and prior is retained. Fitting workers do not decode assessment labels or use the previous noon for training or selection.

Worker time totals 17,122.42 seconds, or 4.76 hours, excluding some parent preparation and supervision work. The longest worker takes 1,893.11 seconds. Maximum sampled worker RSS is 2,569,633,792 bytes, approximately 2.39 GiB. All fits remain within their fixed operational limits. The more expensive mechanisms have not earned an accuracy promotion here.

The registered proper-scoring experiment follows, then the twenty-date anchor audit. The fresh successor dates remain reserved and opaque. Any candidate still needs a separately frozen fresh-date evaluation and the unchanged accuracy, per-asset, log-loss and nominal uncertainty gates, followed by execution and economic validation.

Protocol SHA256: `1ece130d6f36def969cafaa5012f0d9fd111656eca1a96a05c9d797edc641c16`.

Summary SHA256: `d78961603a4765170c831d2a458b32f6f4982dca8ec90f3a2e2ce248f64d20b8`.

Evidence SHA256: `2a93e9893b02a13dc658b2daa7de721334657d8f99927afb756949c49c199e60`.
