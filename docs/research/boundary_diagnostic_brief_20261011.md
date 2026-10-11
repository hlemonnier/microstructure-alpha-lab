# Diagnostic brief: Microstructure Alpha Lab

Prepared October 11, 2026, after completion of the fixed 48-fit midpoint study. The user requests independent advice because many experiments have produced only modest gains. The priority is predictive accuracy. No substantial independently confirmed improvement or executable alpha has been established.

## Read these first

1. This brief and `boundary_diagnostic_review_request_20261011.md`.
2. `boundary_confirmation_report_20260907.md`: the original independent result and subsequent conversion of those dates into development data.
3. `boundary_broad_accuracy_report_20260909.md`: the strongest measured broad development anchor and the difference between three-date and twenty-date scores.
4. `boundary_midpoint_continuation_report_20261011.md`, its matching evidence JSON, and `boundary_midpoint_completion_accounting_20261011.json`: the newly completed result.
5. `boundary_research_log.md` and the dated experiment reports: what was already tried.
6. `boundary_quote_renewal_design_20261011.md`, `boundary_quote_renewal_synthetic_report_20261011.md`, and `boundary_latent_renewal_research_20261011.md`: the current, unvalidated research frontier.

Historical documents are dated records. Their old next-step paragraphs and pause statuses are not current instructions. In particular, the September and October midpoint pause records have been superseded by the completed continuation, while their original contents remain intact for provenance.

## The exact prediction problem

The main task is three-class direction prediction for BTCUSDT and ETHUSDT. At decision time d, causal features use observations strictly before d, with any additional source delay declared by the experiment. Let entry be the first raw BBO at or after d+100 ms, and exit the first raw BBO at or after d+5100 ms. The latter time does not move with a late entry.

Define D = exit_bid + exit_ask − entry_bid − entry_ask and T = max(entry_ask − entry_bid, 2 × minimum_tick). The label is +1 if D>T, −1 if D<−T, and zero otherwise. Boundary ties are neutral. BTC uses tick 0.1 and ETH tick 0.01. Primitive prices are converted to exact fractions of their shortest parsed-float decimal representation; the canonical implementation is `src/lob_forge/label_math.py`. Do not substitute next-event labels, smoothed prices, or a different horizon when comparing scores.

Each standard assessment date contributes 7,070 decisions per asset, one per second from 12:02:00 to before 13:59:50 UTC. This narrow intraday coverage is a limitation. Adjacent five-second outcomes overlap; the row count is not an independent sample count. Main summaries average asset/date panels, and uncertainty uses paired dates rather than treating every row as independent.

Primary performance is balanced accuracy: mean recall of the three classes. Natural accuracy and natural log loss are separate metrics. The forecasting models produce natural class probabilities in down/neutral/up order. Balanced decisions divide by a declared class-prior estimate before argmax. The common `forecast_3600` policy uses a causal one-hour forecast-marginal update, initialized from available historical label information. A change in decision priors changes balanced decisions, not the underlying natural probabilities or their log loss. Consult the policy code and each frozen protocol for initialization and release timing.

The resumed midpoint screen uses three repeatedly exposed dates: June 3, June 7 and June 11, 2023. Each fold uses four earlier training days and a separate preceding validation day. Its training dates are d−5 through d−2; validation is d−1. No assessment labels are decoded by its fitting workers.

## Data and information boundary

Binance official quote and trade archives supply the target-venue BBO and flow. Successive studies add observation histories, peer-asset features, spot/perpetual relationships, several quote-currency representations, and recorded depth/flow from other venues or native capture sources. Their source clocks, availability flags, delayed access and replay contracts differ; they should not be conflated into one perfectly synchronized feed.

Binance `bookDepth` percentage bands alone are not genuine price-level L2. Some later studies do contain separately sourced depth/capture information. Review the relevant source and replay reports before making either a blanket L2-availability or L2-absence claim. Publisher timestamps plus assumed delay also do not establish a measured end-to-end trading latency.

Twenty dates, May 24–June 12, were first used for independent confirmation and then explicitly exposed for development. They cannot now be called unseen. A successor twenty-date reservation, June 13–July 2, is still opaque for market analysis. The diagnostic package intentionally does not include its raw contents. Do not acquire or open those outcomes merely to diagnose the current ceiling.

## What the evidence actually shows

| Comparison | Balanced accuracy / result | Interpretation |
|---|---|---|
| First independent confirmation, 20 dates | Best candidate +1.9481 pp versus original reference; 98.75% paired-date interval [−0.2226,+3.4175]pp | Failed the 5 pp effect gate and stricter interval gate |
| Best broad development observation blend, 20 exposed dates | 54.0516%; +3.4431 pp versus original registered reference, +0.4821 pp versus established blend under common policy | Modest development gain; no independent substantial gain |
| Same observation blend, 3 recent exposed dates | 57.1421%, versus 53.5062% on the other 17 exposed dates | Absolute performance depends strongly on cohort |
| Complete retained control for midpoint study, 3 dates | 57.340789%; natural log loss 0.762205735 | Current matched complete reference for this screen |
| Best new midpoint-family candidate | 57.312987%; difference−0.027802 pp; log-loss ratio 1.000810139 | No improvement over complete control; BTC+0.098180 pp, ETH−0.153784 pp |
| Quote-renewal prototype | 44 synthetic E2 E steps passed; no market fit or score | Software evidence only |

The twenty-date original reference is 50.6085% under its originally registered decision rule and 50.8218% under the common `forecast_3600` rule. The observation blend gains 3.4431 pp versus the former and 3.2298 pp versus the latter. Keep these comparisons distinct. The additional matched-data control and established blend are documented in the broad report.

The completed midpoint study tested 48 distinct fits and 576 reporting panels. Every source retained 42,420 assessment rows. All 576 panels were recomputed, all 96 retained panels reproduced exactly, and the eight repeated procedures matched all 32 probability/prior arrays. The best new candidate was `observations_fx 100_raw_side_hgb_old_neural`, so even the family winner did not require the midpoint transformation. No procedure passed expansion.

Its 48 procedures required 56 completed fit executions across preserved attempts:8 in September,11 in the October restart,37 fresh in the continuation, plus 2 historical interrupted executions. The continuation copied 11 successful workers. These repeats add no independent observations, and the final assessor’s 48-resource list does not total all historical compute costs.

The existing development expansion gate requires at least 0.5 pp over the complete control and every already stronger measured reference, gains in both assets, and natural log-loss ratios at most 1.01. Separate mechanism attribution requires 0.25 pp over the matched counterpart, both assets improving, and the log-loss restriction. The original substantial-gain requirement needs 5 pp over its original frozen reference, twenty independent dates, uncertainty and research-budget gates. A reviewer may critique these thresholds prospectively; changing them cannot retroactively turn failure into the original success claim.

## What has already been explored

The dated reports cover linear baselines; pooled balanced trees; boosted-tree variants; tabular and numerical-embedding MLPs; raw quote sequences; recurrent and temporal-attention models; pooled and residual models; TabICL; retrieval; learned or fading memory; multihorizon/distributional/proper-score variants; same-day and forecast-based class priors; delayed calibration and expert feedback; event clocks; richer quote/trade observation histories; cross-venue depth and flow; native captured depth; quote-currency conversion; and the latest fixed midpoint representation.

Many variants improve a weak component or a narrow metric without beating the strongest complete procedure. For example, the observation representation transfers modestly across twenty dates, while its neural/tree averaging contributes only 0.0423 pp over the new tree alone. The counted-depth study, raw quote-currency study, temporal-attention study and completed midpoint family supplied no advancement candidate. Refer to their reports for exact cohorts and comparators; do not rank scores from different date sets directly.

This research history itself is a concern: extensive adaptation to three repeatedly exposed dates can produce a deceptively precise local leaderboard. A rigorous diagnosis should examine experiment selection, cohort representativeness, signal headroom and the value of additional data, not just nominate another architecture.

## Current unvalidated renewal hypothesis

The new prototype groups quote rows by exact publisher millisecond, retaining first and last BBO in each block. A coarse state combines training spread bins with queue-imbalance bins. A two-component geometric mixture jointly models the next block’s delay and its first/last price/spread/next-state mark. Window endings contribute right-censor likelihoods. Regularized EM uses only completed and censored historical transitions. Survival conditioning accounts for the age of the last observed block at a query.

Simulation preserves both future quote stopping times and the random entry spread. An explicit persistent reflected additive law keeps bid prices positive without discarding paths, and exact comparisons preserve threshold ties. The near-zero reflection and coarse stationary state are modeling assumptions. The existing mixture resets its latent component after each block; it does not model a persistent hidden regime.

All 44 synthetic checks and 190 artifact hashes passed. The 256-path fixture still moved query probabilities by about 0.04628 across seeds and 0.04520 after doubling paths. It is not a convergence certificate. A proposed training-only market preflight has not been registered or run. No market adapter exists, and the CLI accepts synthetic manifests only.

Independent review identified practical preflight caveats:30 EM steps enforce monotonicity, not convergence; current intrinsic limits are 250,000 total cache paths and 90 seconds; query archives need explicit hash/registration verification for market use; an unreflected query still scans all path frontiers; fixed source windows may exceed row/block ceilings. A contemplated 60-query/1% decision-stability threshold allows zero flips, and tight Monte Carlo tolerances may legitimately fail even at 16,384 paths. These are unresolved design issues, not empirical failures or approved settings.

The persistent-HMM research note has further unresolved points: a shared transition matrix does not nest state-specific reset mixtures; exact empirical marks assign zero likelihood to unseen marks; observed reflected boundary points require a pushforward likelihood; and component labels must align across states. No HMM implementation or predictive result exists.

## Package scope and useful review questions

The diagnostic ZIP supplies the current tracked source, protocols and reports; actual saved predictions/labels/priors for the completed three-date and twenty-date studies; representative real feature matrices and training samples; a small source-shaped raw quote/trade excerpt with provenance; result summaries from earlier studies; and the complete final synthetic E2 E artifact. Its inventory states exactly which files are included, derived or omitted.

This is a diagnosis and score-recalculation package. It is not a complete tens-of-gigabytes training-data mirror, nor a promise that all historic training commands can run offline. Raw source inventories and frozen hashes make missing inputs explicit. Environments, credentials, bulky duplicate checkpoints and untouched confirmation contents are omitted. Exported samples are selected by fixed row/time positions for inspection, not by outcomes, and must not be treated as a new validation set.

Please challenge the statistical estimand, target/metric alignment, label and clock contracts, information content, sampling design, nuisance-prior adjustment, model diversity, and research process. Determine whether the next high-value action is better conditional modeling, a different representation, broader/cleaner observation coverage, or a principled stopping decision. Do not promise a gain that the evidence cannot support.
