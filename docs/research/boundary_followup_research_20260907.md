# Follow-up accuracy research after the first confirmation

The first independent round did not meet the substantial-gain gate. All follow-up experiments use **already exposed development dates**. The event-clock and morning studies use May 26, May 30, June 3, June 7 and June 11, 2023: the same 70,700 distinct noon decisions, across five dates and two assets. The direction-transfer study reuses all twenty first-round dates and their 282,800 decisions. Repeated models and policies do not create additional independent observations.

The objective remains at least five percentage points of balanced-accuracy improvement, with independent date evidence, improvements in both assets and no material natural-log-loss regression. No result in this document alone can satisfy that objective because these dates are exposed. The preceding independent evidence and its failed gates remain in [the confirmation report](boundary_confirmation_report_20260907.md).

## Event-time observations

The [feature protocol](boundary_event_clock_data_20260907.json) adds observations of recent aggregate-trade sequences and BBO queue changes. It measures the last 8, 32 and 128 aggregate trades, including signed quantities, trade-side persistence, volume concentration, trade prices relative to the known quote, and elapsed time. Separate same-price bid/ask quantity increases and decreases retain information lost in net flow. Quote-update windows of 16, 64 and 256 events retain changes at different activity rates.

The [preparation evidence](boundary_event_clock_preparation_evidence_20260907.json) covers 44 sessions and 3,801,359 existing rows. Every original field and label is preserved exactly. Raw current-quote alignment and save/reload equality pass for every session. Sixty-seven own-asset observations and six peer observations expand the fitted schema from 95 to 168 columns. A full-session loader comparison preserved all 85,966 eligible rows per asset and every original feature exactly.

The [development screen](boundary_event_clock_screen_20260907.json) fits four models per date: pooled trees with 7 and 31 leaves on the new schema, a 31-leaf tree on the original schema, and the same pooled neural architecture on the new schema. The old 7-leaf tree and neural checkpoint predictions must reproduce exactly. The study includes fixed equal-probability blends and applies the original and one-hour causal forecast-prior policies to each procedure. It comprises 20 new fits and 320 model/policy/asset/date evaluation panels.

[Binance's current aggregate-trade contract](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#compressed-aggregate-trades-list) groups executions over 100 ms by price and taking side. The new trade observations therefore impose an explicit additional 100 ms lag on the recorded transaction timestamp. This is a modeling assumption, not an observed historical delivery time or network-latency guarantee. The original features retain their existing timestamp convention for the feature ablation.

[Queue-reactive modeling](https://arxiv.org/abs/1312.0563) motivates measuring how order-flow activity depends on queue state. [Event-sequence classification](https://arxiv.org/abs/1707.05642) provides a related predictive approach. Neither source establishes a gain in this dataset. Here, BBO quantity decreases do not identify cancellations versus executions, aggregate records are not individual fills, and the observations do not reconstruct full L2.

## Same-day supervised adaptation

The original procedure stops gradient training roughly 36 hours before the noon assessment. Its preceding-day validation chooses a checkpoint, but does not update the weights with that day's observations. The [morning study](boundary_morning_screen_20260907.json) tests whether released same-day outcomes can improve a frozen representation.

Training uses either 05:30–11:30 or 00:02–11:30 UTC, retaining a row only when its actual future quote arrived by 11:30. Validation begins at 11:30:10, ends before 11:57, and requires the label to be released by 11:57:10. The unchanged assessment starts at 12:02. Fitting and checkpoint time are recorded against that 290-second interval; this is a computation check with cached observations, not a live-ingestion benchmark.

The neural model retains the original 95-column schema, fitted quantiles and posterior-recovery convention. Four update epochs at either 0.0001 or 0.0003 learning rate are compared with epoch zero on morning validation. Separate pooled 7-leaf trees train on each morning window. Six frozen sources, six new fitted procedures and eight fixed probability blends are evaluated under three common decision policies. Across the five dates this means 30 new fits and 600 evaluation panels.

Changing training class weights requires care. For original recovery prior pi_old and raw network logits z, the natural posterior is:

    p_k = exp(z_k) * pi_old,k / sum_j(exp(z_j) * pi_old,j)

Morning class-balanced training uses the adjusted logits:

    z_recent,k = z_k + log(pi_old,k) - log(pi_recent,k)

If q is their softmax, normalizing q * pi_recent recovers exactly the same natural p. Only the training loss uses the adjustment. The existing forecaster still recovers natural probabilities with pi_old, before and after gradient updates. This avoids an artificial probability shift merely from changing the class-frequency convention. Tests verify the identity, a zero expected weighted-loss gradient at the correct natural posterior, preserved normalization, and checkpoint parity.

A separately [registered matched-control extension](boundary_morning_controls_20260907.json), fixed before morning fitting and outcome inspection, adds the original neural model blended with each same-day tree. Its 60 additional evaluation panels distinguish a fine-tuning benefit from an improvement already supplied by the fresher tree. It adds no model fits and changes no fitting or selection rule.

## Validation and evidence status

The implementation suite passes 531 direct tests with the research dependencies. Minimal-dependency CI passes 476 direct cases, skipping 6 optional tests and 19 optional modules. The new research files pass Ruff. The five CI mypy targets were unchanged from the verified first-round push.

Both fixed screens are complete. Their outcomes remain developmental, with no substantial-gain or economic-performance claim.

The [event-clock results](boundary_event_clock_evidence_20260907.json) give the new neural/tree blend 55.5100% balanced accuracy, versus 55.0515% for the original blend under the same one-hour forecast-prior rule: **+0.4586 points**. Both assets improve (+0.4239 BTC, +0.4932 ETH points), and the paired gain is positive on all five dates. The 7-leaf tree gains +0.4847 points over its original-feature counterpart; at 31 leaves the feature gain is +0.3571 points. Larger tree capacity contributes little. Relative to the original registered reference on these five dates, the best blend gains +4.7959 points, but the subset gives the known June 3 quiet-day failure more weight than the full twenty-date sample. This is not a near-confirmed discovery.

The [morning results](boundary_morning_evidence_20260907.json) show a smaller improvement. Full-morning fine-tuning at 0.0003, blended with the original tree and using the same forecast-prior rule, reaches 55.2072% balanced accuracy: **+0.1557 points** over the matched original blend. Natural log loss improves from 0.838498 to 0.834422. Both assets improve in that combination. The [additional matched controls](boundary_morning_controls_evidence_20260907.json) show that fine-tuning does not improve balanced accuracy when combined with the corresponding newly trained morning tree. The longest fit/checkpoint operation is 8.54 seconds, within the declared 290-second interval under the recorded cached-data setup.

## Cross-asset direction transfer: rejected

The [registered direction-transfer rule](boundary_direction_transfer_20260907.json) tests whether simultaneous peer forecast odds add information beyond the raw peer features already supplied to each model. For own movement probability m, it shifts the conditional direction log odds by beta times the peer direction log odds, then reallocates the same m between up and down. Neutral and movement probability remain exactly unchanged. The zero-strength control returns the original probabilities exactly; peer clocks must match; no realized peer outcome enters the rule.

The fixed grid of nine strengths, four sources and two policies on forty asset/date panels comprises **2,880 post-fit evaluations and no new fits**. In the [complete results](boundary_direction_transfer_evidence_20260907.json), beta zero is best for both strong blends. For the original blend under the one-hour prior rule, balanced accuracy is 53.5694% at zero, 53.5029% at +0.125 and 52.9874% at +0.5. At -0.125 it is 53.5391%. The frozen transformation is rejected. This result does not reject every possible cross-asset architecture.

[Cont, Cucuringu and Zhang](https://arxiv.org/abs/2112.13213) report that richer own-asset order-flow measurements can subsume contemporaneous cross-impact, while lagged cross-asset flows can help short-horizon equity forecasts. This motivates distinguishing additional information from redundant peer summaries; it does not establish a result for crypto probability transfer. Three unit tests cover exact controls, movement-mass preservation, price-direction reflection, prefixes and zero-probability stability.

## Broader historical coverage: registered and running

The [history-diversity protocol](boundary_regime_screen_20260907.json) compares fourteen past days sampled every four seconds with four full recent days; a fourteen-second wider-history sample is also compared with four days sampled every four seconds. These are approximate training-row-budget controls. All sampling uses a fixed UTC clock phase independent of labels or volatility. Each cohort fits the same pooled seven-leaf tree and 64-wide MLP, with a fixed equal-probability blend. Prior-day noon remains validation; assessment uses June 3, June 7 and June 11, all exposed dates. The study plans 18 fits and 168 model/policy/asset/date panels over 42,420 distinct assessment decisions.

Four extra May 19/20 sessions extend the event-clock preparation to 48 sessions. The [preparation record](boundary_regime_preparation_evidence_20260907.json) preserves both parent manifests and confirms original-field, current-quote and save/reload parity. An initial preparation failed on relative CLI path bookkeeping after one parquet write, before model fitting. Its output and protocol are preserved; the documented amendment only resolves CLI paths. Feature mathematics and labels are unchanged.

Any selected combination still requires a fresh, separately registered confirmation round and the continuing nominal research error budget.
