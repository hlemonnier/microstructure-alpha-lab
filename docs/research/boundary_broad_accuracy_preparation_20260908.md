# Twenty-date observation-model audit: data ready

The data preparation is complete, and this audit's forty model fits remain pending. The diagnostic below regroups previously revealed scores. The audit will test the existing 220-feature neural/tree procedure across all twenty previously exposed dates, extending the three-date comparison on June 3, 7 and 11.

The observation-only blend scores 57.1421% on those three dates, just 0.1987 percentage points below the strongest retained 57.3408% procedure. It is a useful fixed baseline for checking transfer because it requires no additional venue feed. Those figures are development results, not a projection for the other dates.

## Why broader date coverage matters

The [descriptive comparison of already revealed results](boundary_broad_accuracy_transfer_context_20260908.json) uses the three recent dates and the other seventeen dates already fixed by this audit. The original reference is unchanged; `forecast_3600` denotes the existing causal forecast-prior decision rule.

| Existing forecast | Three recent dates | Other seventeen dates | All twenty dates |
| --- | ---: | ---: | ---: |
| Original reference, registered rule | 50.7469% | 50.5840% | 50.6085% |
| Original reference, forecast_3600 | 53.1375% | 50.4132% | 50.8218% |
| Earlier original blend, forecast_3600 | 56.3124% | 53.0854% | 53.5694% |
| Current observation blend, forecast_3600 | 57.1421% | Pending | Pending |

The current observation blend gains 6.3951 percentage points over the original registered reference on the recent three-date sample. Its gain is 4.0046 points against that reference using the same decision rule, and 0.8297 points against the earlier blend using that rule. The earlier blend's stronger score on the recent subgroup shows why the three-date difference cannot establish transfer. These are exposed-data descriptions; the full twenty-date retraining and later fresh confirmation remain necessary. No schedule, fitting rule or success threshold changes.

## Verified data

- Fifty feature sessions cover BTCUSDT and ETHUSDT from May 19 through June 12, 2023. Forty-eight existing sessions are reused unchanged; only the two missing June 12 event-clock sessions were prepared.
- All raw files were already local. Their official source checksums and exact-label records were verified. The extension preserves every original column, raw quote resolution and stored outcome exactly.
- All sessions have the same 220 observation columns and lossless storage round trips. The cache contains 4,299,594 eligible feature rows and 1,074,884 rows at the original calendar stride of four seconds.
- Every original assessment label and timestamp agrees exactly across all forty asset/date panels: 7,070 rows each, or **282,800 rows per forecast source**.
- Actual target-release clocks are stored as exact integers. The minimum delay is 5,100 milliseconds, matching the fixed entry-delay and forecast-horizon contract.

The cache keeps original feature dtypes in feature-only Parquet files. Labels, decision times and release times are stored separately. There is no downcasting, new target, feature selection, model fitting or score calculation. Derived cache files occupy 6,792,486,915 bytes and remain local, outside Git.

## Fixed audit and current status

The [prospective definition](boundary_broad_accuracy_definition_20260908.json) retains the original neural and HGB configurations, four historical training dates, previous-noon neural epoch selection and a fixed 50/50 posterior blend. Four existing forecast procedures remain unchanged as references. The eventual audit contains forty fits, seven forecast sources and 560 source/policy/asset/date panels.

On the three recent development dates, the runner requires exact reproduction of neural parameters, fitted quantiles, priors and complete forecasts, and the HGB clipping bounds, priors and forecasts. It will report all twenty dates and the other seventeen separately. **The other seventeen dates were also exposed in earlier research; they are not an independent holdout.**

A separate [cache replay](boundary_broad_accuracy_replay_evidence_20260908.json) is now complete. The actual audit preparation function reproduces 515,922 historical normalized rows, 42,420 validation rows and all associated labels, clocks and priors exactly on the three shared dates. The original normalizer retains 214 active columns plus the asset indicator. Applying the saved neural and HGB checkpoints to the cached queries reproduces every original forecast exactly, 42,420 rows per model. This took 31.3 seconds and fitted no model or normalizer. It verifies data/preprocessing/inference compatibility; exact parameter reproduction after retraining remains a separate required check in the full audit.

The runner is prepared, but its final execution protocol is not yet frozen and its forty fits have not run. Its definition places fitting after the current ordered/Langevin and proper-scoring families. Any transfer test of a newly selected challenger requires a separate prospective protocol.

The [preparation evidence](boundary_broad_accuracy_preparation_evidence_20260908.json) records all session checks and artifact hashes. The context protocol SHA256 is `e6b937f39e1017032ed82318b090881b00caadae2e8aa5738ef6b41d383472a5`. The local context inventory, `data/research/boundary_broad_accuracy_context_20260908/context_manifest.json`, has SHA256 `f5192092d7bef7636e14d6eef7cc87e7aca9f75f3943494d56f20b33f86bbeed`. This establishes data readiness only; the substantial independent-gain gate remains unmet.
