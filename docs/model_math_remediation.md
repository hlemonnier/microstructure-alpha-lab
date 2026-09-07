# Model and mathematics remediation — 7 September 2026

This document supersedes the implementation assessment and empirical interpretations from before the September audit. The audit inspected commit `5cafab88b54c4ddb3fe5c630acdcc96f3a42ee75`. The remediation changes source and contracts; it does not reproduce historical market results. Existing historical CSVs, intervals and model candidates require regeneration. Synthetic regression success is evidence of implementation behavior only.

## Findings and changes

| Audit finding | Implemented correction | Regression evidence |
|---|---|---|
| 1. Retrospective last-event decision schedule | Decide at the close of a completed `[start, close)` bucket. Trades/depth must precede close; trailing partial buckets are omitted. Unsupported bucket-retained execution is rejected. | `test_math_remediation.py`, `test_features.py` |
| 2. Same-millisecond entry precedes predictive quote | Preserve update IDs; raw execution/path traversal uses `(timestamp, update_id)`. Generic causal samples cannot enter before their feature event index. | `test_math_remediation.py` |
| 3. Incorrect bootstrap resampling law | Seeded `random.Random` with unbiased index draws; circular equal-length blocks, proper stationary restarts, grouped draws; invalid/degenerate ratio intervals fail explicitly. | `test_statistics.py`, exact iid-limit and nonzero-variance tests |
| 4. NaN can pass acceptance and be formatted as zero | Reject non-finite inputs/calculations and invalid inference outputs. Canonical JSON refuses NaN. A separate explicit policy handles valid infinite no-loss ratios. | `test_alpha_factory.py`, `test_statistics.py`, `test_math_remediation.py` |
| 5. Repeated OOS rows in overlapping folds | Every baseline, conditional, calendar, logistic, ridge and protocol schedule requires step at least test length. Explicit zero is rejected. | `test_math_remediation.py`, walk-forward tests |
| 6. `include_flat` reactivates rejected ridge | Export preserves `always_flat`, including side zero and edge zero. | `test_math_remediation.py` |
| 7. Invalid/non-atomic L2 replay | One atomic-message replay contract backs readiness and tensor loading. Validate complete snapshots, sequence/time/identity, duplicates, finite levels and exchange resets; row limits never consume partial messages. | `test_model_audit_regressions.py`, L2 tests and C++ parity |
| 8. Cumulative and incremental fills double counted | Preserve quantity kind, deduplicate executions, and reconcile cumulative per-order state. Ambiguous mixed normalized records fail closed. | `test_live_validation.py` |
| 9. Invented one-sided aggregate flow | Recover both aggregate volumes using `Q(1±I)/2`; mark aggregate data explicitly. Passive eligibility requires suitable raw, price-aware trades. | `test_live_validation.py`, `test_execution_sim.py` |
| 10. Expired reservations, blocked closing/reversals, pre-cost risk checks | Expire before reservation/target sizing, split risk-reducing child orders within venue limits, use post-cost equity for exposure checks, and attempt market deleveraging after a risk stop. | `test_execution_sim.py` |
| 11. Class-balanced scores mislabeled as original probabilities | Store class weights and invert the posterior tilt before probability/alpha reporting; preserve an explicit raw balanced-score API. Torch weighted CE uses a fixed denominator. | `test_math_remediation.py`, `test_model_audit_regressions.py` |
| 12. OFI loses within-bucket events | Sum every raw quote transition before decimation, including the transition into a new bucket. | `test_math_remediation.py`, `test_features.py` |
| 13. Predicted and evaluated fees differ | Exact two-leg cost equations are shared by thresholding and edge export; non-finite/negative taker costs are rejected. | `test_math_remediation.py` |
| 14. Taker fills ignore liquidity | Finite top/depth quantities bound fills; missing/zero quantities imply no available liquidity. Feature CSVs now carry actual entry/future quantities. | `test_execution_sim.py`, `test_live_validation.py` |
| 15. Unfilled remainder mislabeled inventory; fake passive exit | Carry is filled minus exited inventory. Maker exit requires a subsequent eligible fill; otherwise inventory remains open. | `test_execution_sim.py` |
| 16. Calmar ratio has inconsistent units | Normalize drawdown and return consistently. Report total-period return/drawdown, without silently annualizing it. | `test_portfolio.py` |
| 17. Binary-float lot flooring | Decimal lot arithmetic implements venue flooring, including `.29/.01`. | `test_execution_sim.py` |
| 18. Self-reported holdout completion accepted | Verify source, manifest, candidate file/content, result hash and canonical ledger. Reserve before reading held-out observations. Overlapping dataset ranges remain consumed across candidates/manifests/alternate lock directories. | `test_holdout.py`, `test_evidence_gates.py` |

Additional mathematical/model findings are addressed as follows:

- **Feature identity:** weighted midpoint is `m + spread*imbalance/2`; normalized deviation equals imbalance/2. The default model uses imbalance once. `weighted_midprice` is the descriptive name; `microprice` remains a compatibility field. This is not an estimated Stoikov microprice.
- **Ridge numerics:** solve the augmented least-squares problem with Householder QR instead of forming `X'X`. The intercept is unpenalized; singular unregularized designs are rejected without hidden jitter. The documented objective remains **summed** squared loss plus `lambda*||beta_without_intercept||²`, so mean-loss regularization is `lambda/n`. A predeclared grid must account for training-size changes.
- **Posterior uncertainty:** the mean helper now uses a Normal–Inverse-Gamma prior and Student-t marginal posterior. Prior scale is in squared observation units. A single positive observation retains uncertainty. Zero prior mean precision uses the limiting improper NIG prior (not an independently flat mean prior). This model assumes independent Gaussian observations; dependent/overlapping trades must first be grouped appropriately. Its descriptive Sharpe ratio is not a posterior expectation of Sharpe.
- **Classification reporting:** pooled macro-F1/balanced accuracy are recalculated from summed confusion counts. They are not weighted averages of fold ratios. Direction/half-spread labels remain classification conventions, not break-even targets.
- **Sequence information:** causal return channels retain price movements lost by per-snapshot centering. TCN dilation depth covers the requested window. XGBoost encoded labels are decoded to `-1,0,+1`.
- **Sequence decisions/economics:** validation data fit a conditional payoff table and abstention policy. Frozen candidates bind this policy, priors, costs, latency, checkpoint, normalization, data and load limits. Evaluation uses actual timestamps/depth and one persistent account. A class probability alone does not determine expected executable payoff.
- **Pretraining claim:** masked mean imputation is explicitly an untrained reconstruction baseline. It cannot certify learned self-supervised pretraining. Building and assessing a learned representation remains an open research experiment.
- **Inference calibration:** retain HAC as a diagnostic; official inference also requires the separated-batch Student-t guard, at least 20 folds and a minimum AR(1)-approximate information diagnostic. Degenerate/single-observation evidence cannot yield a zero p-value. `scripts/calibrate_mean_inference.py` exposes the declared null domain and Monte Carlo uncertainty. These conservative screens are not a theorem for arbitrary dependence, heavy tails or adaptive model search.
- **Portfolio/Kelly:** exact turnover fees include exit notional. Calendar-day reporting includes inactive days, defaults to 365 periods/year, resets daily stops, and enforces margin/bounds for supplied notionals. Kelly requires decimal returns and an explicit matching horizon; raw dollar-fold PnL cannot be silently used as return variance.
- **Capacity/fill validation:** identifiable historical liquidity budgets are shared across overlapping orders. New observed-capacity and joint fill/markout scenario estimators report sampling uncertainty. Shadow validation includes observation coverage, decision-level mismatch and relative size error; the evidence gate requires complete observed coverage. Assumed impact curves and deterministic queue scenarios remain assumptions.
- **Holdout/data provenance:** development labels overlapping held-out intervals are purged. Feature build markers and combined manifests are version 2 and bind `bucket_close_raw_ofi_v2`; neural/economic artifacts use version 3 contracts. Recognizable generated feature CSVs must also prove current daily/combined builder provenance before reservation and final verification; arbitrary external feature schemas remain supported as a separate trust boundary. A new hash on an old result does not recreate the experiment.

## Mathematical contracts

For one unit, entry ask `A`, exit bid `B`, and equal per-side fee-plus-slippage `c` bps:

`g_long = 10000*(B/A - 1)`

`net_long = (1-c/10000)*g_long - 2*c`.

For entry bid `B`, exit ask `A`, `g_short = 10000*(1-A/B)` and

`net_short = (1+c/10000)*g_short - 2*c`.

These are fixed-quantity round-trip diagnostics. Realized account returns additionally depend on partial fills, timestamps, shared capital, inventory and subsequent liquidity. A sum of diagnostic one-unit row payoffs is not an executable portfolio return. Deployment comparisons must use the same chronological decision stream and stateful execution contract. Rejecting repeated OOS blocks does not, by itself, remove overlap between the holding intervals of distinct decisions.

For passive orders, expected payoff requires the joint fill/markout law `E[F*payoff|X]`; it cannot be replaced by `E[F|X]*E[payoff|X]` without a justified conditional-independence assumption. Aggregate trade totals cannot reconstruct queue identity, intra-bucket prices or local feed latency. Measured shadow fills are required to estimate those relationships.

The fixed-trade portfolio helper has entry/exit observations, not an intratrade marked equity path. Its risk checks cannot certify intratrade liquidation or margin survival. Use the event-driven simulator for those checks. Even that simulator can only attempt liquidation against supplied liquidity: no fill is invented when the book is empty, and a price gap can make a hard mark-to-market bound impossible to maintain.

## Verification and regeneration

Run the dependency-light checks and the pinned CPU research checks separately:

```bash
bash scripts/run_tests.sh direct
PYTHONPATH=src .venv/bin/python -m pytest -q --ignore="tests/test_experiments 2.py" --ignore="tests/test_study_provenance 2.py"
.venv/bin/ruff check src tests scripts/run_reduced_e2e.py scripts/calibrate_mean_inference.py
.venv/bin/mypy src/lob_forge/protocol.py src/lob_forge/holdout.py src/lob_forge/execution_sim.py src/lob_forge/l2_replay.py src/lob_forge/statistics.py
PYTHONPATH=src .venv/bin/python scripts/run_reduced_e2e.py
PYTHONPATH=src .venv/bin/python scripts/calibrate_mean_inference.py --samples 10000
```

The new neural regressions execute real small CPU training for both supported sequence architectures, checkpoint freezing, frozen-limit validation, final holdout evaluation, and XGBoost label decoding when the pinned research dependencies are present. Synthetic fixtures remain explicitly ineligible as empirical evidence. The added CI CPU job installs pinned Torch/NumPy/scikit-learn/XGBoost and executes these regressions so absent optional dependencies cannot conceal model-path failures.

Regenerate in dependency order: raw-data checks → causal features/labels → frozen development/holdout partitions → validation-selected OOS decisions → stateful ledgers → grouped inference and attempt registry → frozen candidate → one reserved final holdout → matching shadow observations. Do not overwrite historical records as though they had used the corrected semantics.

Raw market data and real observed order fills are not present in this checkout. No repaired-market profitability, deployment readiness, impact curve, learned pretraining result or Kelly allocation is claimed by this remediation.

## Seeded inference calibration result

The remediation run used seed `20260907` and 10,000 Gaussian AR(1) null replications per scenario. This measures only the named p-value procedures, not the whole acceptance gate or a real strategy false-discovery rate. Monte Carlo standard errors of the guarded rates were 0.10–0.16 percentage points.

| Folds | AR coefficient | Original HAC z rejection at .05 | Guarded rejection at .05 |
|---:|---:|---:|---:|
| 20 | 0 | 7.72% | 1.26% |
| 20 | .8 | 25.47% | 1.94% |
| 60 | 0 | 6.50% | 1.07% |
| 60 | .8 | 20.03% | 2.59% |
| 20 | .95 | 37.76% | 2.70% |
| 60 | .95 | 34.45% | 1.54% |

The conservative result comes with reduced power. The sample-based persistence screen rejected roughly 80–90% of small, strongly persistent samples; it does not manufacture independent information. Other dependence and tail regimes need predeclared calibration before inferential claims.

## Completed local verification

- Canonical repository suite: **442 passed**, Python 3.12. This excludes the two pre-existing `* 2.py` test backup files (seven duplicate cases), which were preserved. The audit baseline had 368 canonical tests, including one failing readiness fixture.
- Dependency-light runner on Python 3.14: **438 passed, 4 explicitly skipped** because Torch/XGBoost are unavailable there. The pinned environment executes all four.
- Ruff: clean. Mypy: clean across **22 source modules**. `git diff --check`: clean.
- Compiled C++ fixture replay and Python/C++ equivalence: passed.
- Reduced end-to-end pipeline: completed with actual CPU TCN/Transformer artifacts. Both small fixture policies abstained; zero PnL is not evidence of a profitable strategy. Undefined tiny-sample intervals are reported as unavailable.
- Six seeded null scenarios, **60,000 total replications**: completed with results above.
- Current empirical evidence gates: **0/7 passed**. Missing market studies, immutable empirical holdout, observed fills and learned pretraining remain unfilled evidence requirements.
- The added CPU CI job was not run on GitHub as part of the local verification above; remote CI results are separate from these checks.

Local logs, environment versions, calibration output, evidence-gate output and content hashes are preserved in `artifacts/model_math_remediation/verification_manifest.json`. That directory is generated/ignored; this document and all regression tests are source changes.
