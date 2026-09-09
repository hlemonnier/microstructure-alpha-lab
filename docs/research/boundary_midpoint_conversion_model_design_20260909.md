# Matched trade-side conversion representation screen

The research question is whether explicitly removing a fixed bid-ask-bounce component from the TUSD/USDT conversion price improves the unchanged prediction problem. Training-only diagnostics motivated the hypothesis and fixed the half-spacing at 0.00005 USDT/TUSD. They did not measure predictive gains. The full raw-plus-side and midpoint-plus-side source windows are prepared and independently verified before any model in this study is fitted.

## Frozen family

On each of June 3, 7 and 11, 2023, fit all combinations of the original observation/combined representations, conversion-only/BTC-native sources, raw-plus-side/midpoint-plus-side conversion, and the unchanged seven-leaf HGB/64-unit neural learners. These are 16 fits per date and 48 fits in total. The native ETH/BTC source is omitted from this mechanistic family; the already available BTC/USDT conversion stream remains in both source controls.

For every representation/source/conversion group, define three fixed 50/50 natural-posterior blends: new tree/new neural, new tree/original same-representation neural, and new neural/original deep-500-ms tree. There are 24 such blends per date. With the eight original retained controls, the study reports 48 sources times two policies times two assets times three dates: 576 panels. No assessment-tuned blend weights or unregistered refits are permitted.

Each conversion-only input receives the observed TUSD/USDT aggressor side. Each native input additionally receives BTC/TUSD trade fields and converted-basis information. Raw and midpoint variants have identical column counts, observed native trade information, FX-side information, row clocks and availability masks. Only the conversion log rate and derived basis representation can differ. Observation inputs have 227 or 302 fields; combined inputs have 406 or 481. The shared input cache preserves every original 555-field source value and adds the side and separately named midpoint representation; models select exact declared subsets.

All source observations use the same strict 100 ms publication delay and 60-second conversion publication-age bound. The half-spacing is fixed across all dates. No additional 500 ms models are fitted in this family. The midpoint is a modeling proxy, not a certified executable quote or an observed latent efficient price.

## Unchanged training and evaluation contracts

Reuse the exact selected matrices, clocks, training/validation labels and sealed assessment arrays from the completed original quote-currency screen. Verify all original artifacts before reuse, and require the new raw-plus-side features to reproduce every original raw feature exactly. An augmentation may neither change an original predictor nor drop, duplicate, forward-fill or add a row.

Training remains d-5 through d-2 with the original four-second calendar stride; validation is the preceding noon session. Assessment retains all original 7,070 noon decisions per asset/date. Historical labels must be released before the next partition. Preparation copies query-label bytes without decoding them; worker query metadata contains no label path. The parent opens query labels only after all 48 fits finish. All three input folds must finish before the first fit.

Use the original HGB and neural implementations, parameters, class/asset weighting, training-only transforms, random seed, validation epoch selection and natural-posterior recovery. Primary decisions use the causal forecast prior with a 3,600-second half-life and its original frozen initialization. The registered prior policy is secondary. Recompute balanced accuracy, natural accuracy, macro F1 and natural posterior log loss from every saved forecast. Retained controls must reproduce exactly.

Run one learner at a time with two CPU threads, an 8 GiB sampled RSS limit, 0.2-second sampling and a 900-second per-fit wall limit. Require exact saved/restored probabilities and training priors, zero future-prefix error and at most 2e-6 absolute query-partition difference. Check pinned inputs before preparation, before fitting and after fitting. Preserve every failed attempt; no implicit retries or resumes.

## Decisions and limits

Compare every new procedure with the retained complete procedure and the strongest completed quote-currency, counted-depth and expert procedures. A performance candidate can justify broader development evaluation only if it improves mean balanced accuracy by at least 0.5 percentage points, improves both assets and has a natural log-loss ratio at most 1.01 against the complete procedure and every already stronger measured control.

Report two additional sets of attribution comparisons in full: all 20 native-minus-conversion-only pairs and all 20 midpoint-minus-raw-side pairs. A native-information-specific conclusion additionally requires at least 0.25 points over its exact FX-only counterpart, both assets improving and log-loss ratio at most 1.01. A midpoint-representation-specific conclusion applies the same additional rule against its exact raw-plus-side counterpart. A good FX-only procedure can justify a performance follow-up but cannot establish a BTC/TUSD native-information gain. A raw-side gain cannot establish a midpoint-representation gain. These distinctions are fixed before this study's model fits or assessment scores.

All dates are repeatedly exposed development data. Neither this screen nor the source-noise diagnostics consume an independent confirmation round. The unchanged substantial-gain requirement remains at least five points over the original frozen registered reference, improvement on both assets, its natural log-loss bound, twenty independent dates, the original nominal uncertainty requirements and the sequential research-budget correction. No model or trading procedure is promoted merely because its code or feature checks pass.
