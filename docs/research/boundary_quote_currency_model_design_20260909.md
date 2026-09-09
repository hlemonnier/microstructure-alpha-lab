# Alternative quote-currency predictive screen

The hypothesis is that the original BTC/USDT spot input omitted useful price discovery and signed trade flow in BTC/TUSD. Four training-day source inventories found more BTC base volume and more aggregate trade activity in BTC/TUSD. This is a reason to test the source, not evidence of predictive accuracy. ETH/BTC is a separate relative-price and trade-flow hypothesis.

This study uses June 3, 7 and 11, 2023, which are already exposed development dates. It fits 36 models before revealing any assessment metrics. The twenty reserved independent confirmation dates remain unopened for market analysis.

## Fixed source ablations and model family

For each of the original 220-field observation and 399-field combined representations, fit the following three input variants with the unchanged seven-leaf histogram gradient boosting model and unchanged 64-unit neural model:

- `fx100`: both observed conversion streams, including rate, age and availability; no new native trade fields.
- `btc100`: the same two conversion streams plus the 75 BTC/TUSD native trade and converted-basis fields.
- `both100`: the same conversion streams plus both the BTC/TUSD and ETH/BTC native fields.

All source and conversion observations use a strict 100 ms publication-time delay. The already prepared 500 ms caches remain available for a later separately specified sensitivity study; this model family does not fit them. Six FX-only fields are added to each base representation; the BTC and both-native variants add 81 and 156 fields in total. Cross-asset fields retain explicit market prefixes and their own base-asset quantity units.

For each representation and variant, declare three fixed equal-weight posterior blends: new tree/new neural, new tree/original neural of that representation, and new neural/original deep-500-ms tree. There are 12 fitted models and 18 fixed blends per date, plus eight retained controls. The complete screen therefore contains 36 fits and 456 reporting panels: 38 sources, two decision policies, two assets, three dates. No blend weight is optimized on the assessment labels.

The original eight retained controls are `original_reference`, `observations_tree`, `observations_neural`, `observations_blend`, `combined_hgb`, `combined_neural`, `deep500_hgb`, and `combined_instant_hgb_old_neural`. Their saved predictions must reproduce exactly. The strongest completed counted-depth procedure and the previously measured feedback-expert procedure are also compared in the final report, without refitting them.

## Data and mathematical contracts

Reuse the exact original-column subsets, selected row clocks, historical labels and sealed assessment arrays from the completed counted-depth preparation. Verify every original preparation artifact before reuse. The source augmentation may neither change an original value nor drop, forward-fill or add a decision row. The original tree/neural forecasts and training priors were replayed exactly during the source preparation; this screen preserves those exact original matrices and performs exact retained-prediction replay again after fitting.

Training remains days d-5 through d-2 at a four-second calendar stride. The preceding noon session is validation; the unchanged 7,070 noon decisions per asset on d are assessment. Historical label-release clocks must remain strictly before the next partition. Query metadata contains no assessment-label path. Every model worker only decodes training/validation labels; the parent preserves the sealed query labels until all registered model fits finish.

BTCTUSD prices are converted to USDT using observed TUSDUSDT trades; ETHBTC uses observed BTCUSDT spot trades. The conversion rate has units of target quote per native quote. A rate is eligible only if publisher time plus 100 ms is strictly before the decision and its resulting observed age is at most 60 seconds. Converted basis fields compare native last/VWAP prices valued at the current observed rate with the current canonical futures quote. Native returns remain native-quote returns. Missing conversion observations mask basis values and cannot enter basis EWM state. Native trade availability does not certify a fresh executable quote.

The tree retains 200 boosting iterations, learning rate 0.05, seven leaves, minimum leaf size 100, L2 10, train-only 0.001/0.999 clipping, no early stopping, and seed 20260907. Training weights are N/(6 n_asset,class). The neural model retains train-only quantile normalization, hidden dimension 64, twelve epochs, AdamW with learning rate 0.001 and weight decay 0.01, batch size 1024, gradient clipping 5, and seed 20260907. Choose its epoch using preceding-noon balanced accuracy, with natural log loss and then earlier epoch as exact ties.

Recover natural posterior probabilities from balanced training by multiplying by each asset's training class prior and renormalizing before blending or log-loss evaluation. Primary decisions use the existing causal forecast-prior policy with half-life 3,600 seconds; `registered` is secondary. Preserve legacy method-specific registered priors and the exact frozen initialization of the causal prior. Evaluation uses balanced accuracy, natural accuracy, macro F1 and natural posterior log loss.

## Execution and advancement

Prepare all three folds before the first learner. Run one learner at a time on two CPU threads with the existing 8 GiB sampled RSS ceiling, 0.2 second sampling and 900 second per-fit wall limit. Before metrics are revealed, require exact checkpoint probability and training-prior replay, zero future-prefix error and at most 2e-6 absolute probability difference from query partitioning. Check all pinned inputs before preparation, before fitting and after all fits. Preserve every failed attempt; no implicit retries or resumes.

A native-source procedure can justify broader development evaluation only if it gains at least 0.5 percentage points over the retained complete procedure, improves both assets and has a log-loss ratio at most 1.01. It must additionally gain at least 0.25 points over its exact FX-only counterpart, improve both assets against that counterpart and keep the same log-loss bound. Report every native-minus-FX and both-minus-BTC comparison, including losses. Compare the best candidate with the strongest completed counted-depth procedure and expert control; a candidate does not advance if it fails the corresponding 0.5-point, both-asset and log-loss gates against an already stronger measured procedure.

These development rules do not replace the unchanged substantial-gain requirement: at least five percentage points over the original frozen registered reference, improvement on both assets, the original log-loss restriction, at least twenty independent dates and the registered uncertainty and sequential research-budget gates. This three-date screen consumes no independent confirmation round. Repeatedly exposed development data cannot establish a generalization breakthrough.
