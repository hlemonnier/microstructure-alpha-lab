# Boundary-model research

## Objective and current evidence

The user requested ambitious research and then explicitly prioritized predictive accuracy. The active objective is not complete: no substantial gain has yet been independently confirmed. Trading costs, fills, inventory and net PnL remain a separate later evaluation.

The primary forecast is the unchanged three-class five-second movement target. Balanced accuracy receives equal weight across the three classes, assets and assessment dates. Ordinary argmax accuracy and natural posterior log loss are reported alongside it. A rise in ordinary accuracy can reflect better neutral predictions, so it cannot substitute for class-balanced discrimination.

The confirmation gate is fixed at at least 5 percentage points more mean balanced accuracy than the original validation-selected reference, positive paired date-cluster 95% lower bound, improvements in both assets, at least 20 distinct dates, and no material log-loss regression. An additional nominal continuing-research error budget, registered before score inspection, requires a positive 98.75% date-interval lower bound for each of the two first-round procedures. A second reference has the same full-day training volume as the candidate, to distinguish model improvements from additional observations.

## Completed development research

All results below use the previously exposed May 16–23, 2023 data and legacy floating-point labels. Assessment dates are May 21–23: six asset/date panels and 42,420 decisions. These repeated observations do not become independent evidence through additional models or experiments. The machine-readable record, source hashes and exact means are in [boundary_development_evidence_20260907.json](boundary_development_evidence_20260907.json).

| Experiment | Model fits / post-fit trials | Validation-selected balanced accuracy | Ordinary accuracy | Log loss |
|---|---:|---:|---:|---:|
| Original base-feature reference | Included in tabular screen | 52.6712% | 53.1400% | 0.967143 |
| Temporal, cross-asset and path-signature features | 108 fits | 53.4091% | 57.5625% | 0.909991 |
| Movement/direction factorization and reflection symmetry | 30 fits | 53.2730% | 57.0863% | 0.908754 |
| Causal temporal convolutions and sequence MLP | 18 fits | 53.8390% | 56.8883% | 0.915229 |
| Full-day training and raw event features | 72 fits | 53.7321% | 58.2862% | 0.888779 |
| Delayed-label probability recalibration | 144 trials | 53.3506% | 52.5224% | 0.947177 |
| Pooled cross-asset MLP and shared-parameter ensembles | 12 fits | 54.2498% | 59.4389% | 0.870718 |
| Available class-prior decision rules | 288 trials | 54.6375% | 59.6157% | 0.869082 |
| Fixed-weight neural/tree blends | 30 trials | See fixed candidate below | | |

There are 240 exploratory model fits and 462 post-fit trials. Different studies use different stronger controls; the table describes their selected challengers and is not a claim that each study incrementally improves its predecessor.

The strongest fixed development candidate is a 50/50 average of pooled-MLP and full-day event-feature boosted-tree natural posteriors. For balanced decisions, divide each posterior by the mean of training priors and the preceding noon's class frequencies. Its development balanced accuracy is 55.8520%, ordinary accuracy 60.0825%, and log loss 0.863360. Relative to the original reference these are +3.1808 points, +6.9425 points, and 10.73% lower log loss. Its weight was selected on exposed development outcomes. These figures are a hypothesis, below the 5-point balanced-accuracy gate, and use legacy labels.

Shared training gave a larger improvement than the tested sequence complexity. The four-member shared-parameter network did not beat the ordinary pooled MLP. The implementation is inspired by BatchEnsemble/TabM, not an exact reproduction of TabM. Delayed adaptation uses only labels whose actual future quote has arrived; it is online supervised learning, not reinforcement learning.

## Exact target arithmetic amendment

Before any independent model fitting, a development-only audit found 1,818 incorrect labels among 115,584 original noon observations (1.5729%). Every disagreement was an exactly-at-threshold move incorrectly classified as directional by floating-point subtraction. The exact diagnostic and immutable input hashes are in [boundary_exact_label_diagnostic_20260907.json](boundary_exact_label_diagnostic_20260907.json).

The corrected label compares twice the mid-price change to the exact decimal threshold:

    D = future_bid + future_ask - entry_bid - entry_ask
    T = max(entry_ask - entry_bid, 2 * minimum_tick)
    label = +1 if D > T, -1 if D < -T, otherwise 0

Prices and minimum tick use their shortest decimal representations. There is no epsilon and no rounding to an assumed tick grid. Canonical scalar and vectorized builders share this contract. Semantic versions invalidate stale caches. Exact ties, tiny real moves beyond the threshold, half-spread and one-tick modes, nonfinite values, and large integer scales are covered by tests.

Original data and results are preserved. Copies of all 56 full-day sessions resolve the same raw entry/future quotes, require identical timestamps and midpoint values, and preserve every column except the label and semantic version. Label corrections alone do not count as model performance gains.

The dated [confirmation amendment](boundary_confirmation_20260907_exact_labels_revision.json) was registered after raw acquisition but before any independent fit or outcome inspection. Both references and the candidate receive corrected labels. All model settings, partitions, selection rules, mixture weight and metric gates remain unchanged. The earlier bitwise reproduction checks concern legacy labels; they do not imply corrected-label scores.

## Independent confirmation

The initial [confirmation protocol](boundary_confirmation_20260907.json) was registered before acquisition. It fixes BTCUSDT and ETHUSDT, May 24–June 12 inclusive, four past training days, the preceding day for validation, and 7,070 noon decisions per asset/date. Training may subsequently use older assessment dates only after their outcomes would be known. The procedure and model choices remain frozen throughout.

All 80 public bookTicker/aggTrades ZIPs have been acquired and verified against official SHA256 values: 4,822,880,970 compressed bytes. Every registered noon window passes coverage checks. The feature dataset is forecast-only and carries no invented maker-fill or L2 evidence. Prior source coverage problems on May 16/17 were amended and documented before fitting development event models; those original failed attempts are preserved.

The confirmation runs 17 fits per assessment date, 340 total. Every date's predictions and model selections must be saved before any assessment score is inspected. Artifact inventories and hashes are rechecked immediately before unsealing. Uncertainty uses 10,000 paired date-cluster bootstrap draws, with three-day circular blocks as a dependence sensitivity check. These intervals do not remove longer regime dependence.

Corrected preparation is complete: all 56 sessions and 4,672,228 rows passed raw-resolution and unchanged-field checks. All 340 primary fits and 20 secondary pooled-tree fits are complete. Both procedures were frozen before either score was inspected. One failed secondary fit is counted separately; its compatibility fix is documented below.

The [independent confirmation report](boundary_confirmation_report_20260907.md) records the complete outcome. The pooled-tree blend reaches 52.5565% balanced accuracy versus 50.6085% for the original reference and 51.2411% for the matched-data control. Its +1.9481-point gain is below the 5-point gate, and its nominal research-budget 98.75% date interval is [-0.2226, +3.4175] points. Both candidates fail the discovery gate. Ordinary accuracy and log loss improve, but no substantial predictive discovery or economic gain is confirmed.

An [exposure ledger](boundary_confirmation_exposure_20260907.json) permanently marks all twenty dates as now available for development. The subsequent [same-day prior study](boundary_same_day_priors_evidence_20260907.json) adds 1,760 decision-only trials and no model fits. Its best fixed development combination reaches 53.5694% balanced accuracy, +2.9610 points over the original registered reference and +1.9602 points over the same-policy matched-data control. It preserves natural probabilities, ordinary accuracy and log loss exactly. This is exploratory evidence on exposed data, below the 5-point target, and requires fresh confirmation before any promotion.

The [follow-up research report](boundary_followup_research_20260907.md) records the next exposed-date experiments. Event-clock trade and queue observations add +0.4586 balanced-accuracy points to the matched blend across five dates, with positive gains on both assets and every date. Morning fine-tuning adds only +0.1557 points in its best matched blend, and its morning-tree controls do not support a consistent fine-tuning benefit. A fixed cross-asset direction-transfer family is rejected on all twenty exposed dates: zero transfer is best for both strong blends. Later historical-coverage and long-context studies add small matched gains on three exposed dates, while exact move-size distribution losses do not beat their coarse-target control. A hindsight diagnostic finds limited additional room in the tested score-threshold family. New spot trade observations improve the tree by +0.3367 points over an identical perpetual-trade transformation, but only +0.1385 points over the stronger existing blend; the improvement is sensitive to added delay. None of these observations is fresh independent confirmation.

## Research basis and limits

- [Sirignano and Cont](https://arxiv.org/abs/1803.06917) found benefits from pooling financial observations across equities. That motivates shared training; transfer to these two crypto assets must be measured locally.
- [Chevyrev and Kormilitzin](https://arxiv.org/abs/1603.03788) describe path signatures. The implemented level-two antisymmetric areas encode temporal ordering and pass orientation tests, but their observed predictive gain was modest.
- [Gould and Bonart](https://arxiv.org/abs/1512.03492) study next-move queue imbalance in equities. Their horizon and markets differ from this five-second target.
- [Tsantekidis et al.](https://arxiv.org/abs/1810.09965) motivate stationary order-book features and sequence architectures. We have BBO and trades, not the full depth required for a faithful DeepLOB experiment.
- [TabM](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf) motivates efficient parameter-sharing ensembles. Our smaller custom implementation is explicitly an inspired experiment.
- [Nagy, Calliess and Zohren](https://arxiv.org/abs/2301.08688) study RL execution using synthetic noisy forward-return alpha. This is not evidence that RL discovers forecast information in these observations. [Nevmyvaka, Feng and Kearns](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf) similarly address execution decisions. The user's accuracy priority therefore favors supervised forecasting research first.
- [de Lataillade et al.](https://arxiv.org/abs/1203.5957) derive trading thresholds under specified signal dynamics, costs and position limits. That supports a later policy stage, not a forecast-performance claim.

A possible further route is conditional distribution modeling, motivated by [NGBoost](https://proceedings.mlr.press/v119/duan20a.html), or event intensities informed by [queue-reactive Hawkes models](https://arxiv.org/abs/1901.08938). These are untested here. They would need their own fixed experiment and independent confirmation; no theoretical model supplies an empirical gain by itself.

The later raw quote-sequence screen completed nine fits and 192 panels; none beats the strongest spot tree's 57.2806% balanced accuracy on the three exposed dates. Aligning neural checkpoint selection with the causal decision policy improves that neural component but fails to improve its strongest fixed blend. These negative results are retained in the follow-up report. The Bybit depth experiment reconstructed 28 public sessions and completed 24 fits with matched best-quote/deeper-book controls at 100ms and 500ms publisher-time delays. Its strongest blend reaches 57.3006%, only +0.0201 points above the spot tree. The completed XGBoost/source-union experiment and subsequent causal-basis screen also fail to beat that control. Native queue-flow preparation and a target-venue Binance full-depth preflight now address missing information. The public raw source includes capture timestamps, with exact economic quote-content parity but documented publisher-clock differences from the original archive. All remain development research; no substantial predictive gain or execution gain is confirmed.

## Reproduction and validation

The branch preserves each research family in a separate commit. Legacy protocols pin source hashes and must be replayed at their matching commit. Re-running legacy scripts against corrected label code is intentionally rejected by provenance checks.

For the corrected procedure:

    PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/relabel_boundary_exact.py
    PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/run_boundary_confirmation.py --development-check --run
    PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/run_boundary_confirmation.py --run

Raw sources and large model artifacts remain local and ignored by Git; tracked protocols and manifests record their hashes. After both confirmations and the same-day prior study, the full suite passed 515 direct tests with all research dependencies; the isolated minimal-dependency CI run passed 476 direct cases and correctly skipped 6 optional tests and 14 optional modules. This includes five same-day prior tests covering explicit delayed-observation weighting, prefix causality, unchanged inputs, invalid data, and empty history. Ruff and the five CI-targeted mypy modules pass. No live trading or paid compute has been used.

The additional pooled-tree procedure is frozen in [boundary_pooled_tree_confirmation_20260907.json](boundary_pooled_tree_confirmation_20260907.json). Mathematical tests for sample weighting, posterior inversion, checkpoint-prior compatibility and multiplicity intervals pass. Its twenty independent fits are complete and their results are included in the report.


## Operational and statistical amendments before score inspection

The secondary run stopped after its first training fit at a compatibility assertion: the tree stores priors as a two-dimensional array, while the neural checkpoint stores a dictionary indexed by asset. The corrected check compares the corresponding numerical arrays. Exact parity was verified on the actual saved checkpoints, and a regression test covers the container difference. No secondary prediction or assessment score had been produced at failure; neither procedure's independent scores were inspected. The failed artifact and traceback are preserved under `results/boundary_pooled_tree_confirmation_20260907_initial_priors_check_failure`. The [compatibility amendment](boundary_pooled_tree_confirmation_20260907_priors_check_revision.json) repeats the identical mathematical procedure and counts the failed fit separately.

Because the requested research is open-ended, an additional [nominal error budget](boundary_research_error_budget_20260907.json) was registered before the first independent outcomes were inspected. Round j receives 0.05 / 2**j, split across its registered candidates. This first round therefore uses 98.75% marginal intervals for its two candidates, with 20,000 date resamples. The original 95% and two-procedure 97.5% analyses remain preserved. The allocation assumes valid bootstrap coverage; it does not establish an always-valid guarantee under arbitrary market dependence. The effect-size, per-asset, date-count and log-loss gates remain in force.

The [diagnostic definitions](boundary_confirmation_diagnostics_protocol_20260907.json) were also fixed before outcome inspection. They separate movement ranking from direction ranking and exactly decompose three-class log loss. Directional metrics conditioned on a realized move are explanatory; the condition is unavailable when placing a trade. [Lee (2024)](https://arxiv.org/html/2409.14157v1) highlights how target smoothing and the difference between movement and direction can complicate published order-book accuracy claims. Our target uses two future raw quote observations and no backward-smoothed target price. [Briola et al. (2024)](https://arxiv.org/abs/2403.09267) likewise distinguish forecasting metrics from actionable transactions. Neither paper establishes a gain for this dataset.

The final evidence pipeline applied the registered research gate, computed the diagnostics, reran both dependency-complete and minimal-dependency tests, and rendered and visually checked the comparison figure after both procedures completed. Plotting uses an isolated temporary environment with NumPy 2.0.2 and Matplotlib 3.10.1; the model environment is unchanged.
