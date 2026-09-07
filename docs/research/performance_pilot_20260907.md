# Performance pilot: market state and nonlinear expected payoff

**Result: neither challenger improves the registered primary objective. Retain `ridge_default` as the research comparator; do not promote a new trading model.** At the primary assumption of 1 bp per side and 100 ms latency, all three model families generate zero OOS entries and zero PnL. Validation selects the flat control in 16 of the 18 model/session cases; the two BTC May 23 ridge cases select a 5 bps threshold that triggers no OOS entries. This is a result about this candidate family and window, not proof that executable crypto alpha does not exist.

The completed experiment contains 18 fitted models, 864 logged validation threshold trials, and 216 OOS model/session/fee/latency evaluations. The full machine-readable comparison is in [the portable results snapshot](performance_pilot_20260907_results.json). Raw data, checkpoints, predictions and ledgers remain local under the paths described in [reproduction instructions](performance_pilot_reproduction.md).

## What was tested

The unchanged expected-payoff ridge is compared with:

- `ridge_state`: the same regularized linear learner plus 11 causal spread, depth, activity, age and OFI-interaction features.
- `hgb_state`: the same enriched information with two fixed shallow histogram gradient-boosted regressors, one for each side's gross return. It uses 150 iterations, 7 leaves, learning rate 0.05, L2 regularization 10, and no shuffled early-stopping split.

The hypothesis is that conditioning flow on available liquidity and activity improves forward executable payoff. The OFI/depth mechanism is motivated by Cont, Kukanov and Stoikov's stock-market study, whose contemporaneous price-impact result does not itself demonstrate forward crypto prediction. [Primary paper](https://arxiv.org/abs/1011.6402)

Each asset uses rolling four-session training, one-session validation, and one-session testing. Test dates are May 21, 22 and 23, 2023: **three distinct dates**, with two correlated assets on each date. All transformations fit training rows only. Label endpoints are purged before the next partition. A previously tested date can subsequently enter validation or training under this preregistered rolling procedure; it is never counted as a second test date.

The data comprises 32 official Binance USD-M `bookTicker` and `aggTrades` archives for BTCUSDT/ETHUSDT, May 16–23: 1,655,629,727 compressed bytes, 13,538,664 BBO updates, 1,334,162 aggregate trades and 115,584 feature rows in the extracted windows. Every source hash was verified against its provider checksum. Binance documents its checksum files and possible archive revisions. [Provider documentation](https://github.com/binance/binance-public-data#checksum)

The [original midnight protocol](performance_pilot_20260907.json) was frozen before acquisition. Four required BBO prefixes were empty because both assets' archives begin at about 11:49 UTC on May 16 and 08:46 UTC on May 17. The [recorded coverage amendment](performance_pilot_20260907_window_revision.json) uses the first common whole-hour two-hour window after those starts, 12:00–14:00 UTC. Only source integrity and first timestamps were inspected before the amendment; no model had been fitted and no outcomes were inspected. [Coverage evidence](performance_pilot_20260907_source_coverage.json)

## Economic contract

Each asset/session starts with 10,000 USDT, targets 100 USDT per entry, and closes inside the session. Decisions occur every 10 seconds, starting after a one-minute feature warm-up; the fixed exit instruction is five seconds after the entry decision. Primary latency is 100 ms on both instructions; stress latency is 250 ms. The funding schedule assumed in the protocol has no settlement inside the studied window. No exposure carries between sessions.

Features use completed one-second buckets. Training targets use the first quote at or after decision +100 ms and decision +5,100 ms. With entry bid/ask \(b_e,a_e\) and future bid/ask \(b_f,a_f\), the gross targets are

\[
g_L=10^4(b_f-a_e)/a_e,\qquad g_S=10^4(b_e-a_f)/b_e.
\]

For a per-side proportional cost \(c\) in bps, prediction-time net edges are

\[
n_L=(1-c/10^4)g_L-2c,\qquad n_S=(1+c/10^4)g_S-2c.
\]

Actual PnL comes from the persistent cash/inventory replay, using observed bid/ask quantities, partial fills, expiry and residual flatten retries. It is not the sum of target labels. Eligibility uses the clock and the observed decision quote age; it never filters test decisions on future quote lag or realized labels. Maximum observed OOS entry/future quote lag beyond each target timestamp was 461 ms. All 216 evaluation cases finish flat within the 1e-6 USDT comparison tolerance; maximum residual marked inventory is 1.27e-14 USDT from floating-point arithmetic.

For each fee scenario, validation chooses among thresholds 0, 0.1, 0.25, 0.5, 1, 2 and 5 bps plus flat. Ties prefer fewer entries and then higher thresholds. The selected threshold is carried unchanged into its test session and latency stress. These fee rows therefore describe separately selected policies, not repricing one universal policy.

## Out-of-sample results

Values below are **summed USDT PnL across six separately funded asset/sessions** at 100 ms latency, after spread and the indicated fee. They are neither daily returns nor returns on a single continuously traded 10,000 USDT account. Extra slippage is assumed zero. Fees are research scenarios, not a verified account tier.

| Fee per side, bps | Original ridge | Enriched ridge | Enriched boosting |
| ---: | ---: | ---: | ---: |
| 0 | 6.905844 | 7.063354 | 6.603272 |
| 0.1 | 1.453599 | 1.338345 | 1.426535 |
| 0.5 | 0.041465 | 0 | 0.073095 |
| **1, primary** | **0, no trades** | **0, no trades** | **0, no trades** |
| 2 | 0, no trades | 0, no trades | 0, no trades |
| 5 | 0, no trades | 0, no trades | 0, no trades |

The 0.5 bp observations comprise just **one ridge entry and eleven boosted entries**. They cannot support a meaningful high-fee edge claim. Boosting also underperforms ridge on BTC within that row; its total advantage is not cross-asset improvement.

At 0.1 bps per side, the comparison is:

| Model | BTC net USDT | ETH net USDT | Filled entries | Positive asset/sessions | Worst session drawdown, USDT | Net at 250 ms, USDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original ridge | 0.776428 | 0.677171 | 1,394 | 4/6 | 0.351477 | 1.197840 |
| Enriched ridge | 0.483042 | 0.855303 | 1,266 | 5/6 | 0.392082 | 1.006464 |
| Enriched boosting | 0.688778 | 0.737757 | 1,226 | 6/6 | 0.475985 | 1.005630 |

Boosting produces a useful development clue: fewer entries and more uniformly positive sessions at this very low assumed fee. Nevertheless, total net PnL is below the original ridge, its worst session drawdown is larger, and its BTC performance is weaker. At 250 ms both challengers remain below ridge. Those observations do not satisfy the predeclared promotion rule.

The enriched ridge's zero-fee gain is only 0.157510 USDT and does not transfer to ETH or to the pooled 250 ms comparison. This is insufficient support for the added static state features.

Gross forecast error pools the two side-specific squared errors over the same 42,810 OOS feature decisions. It does not treat overlapping labels as independent statistical evidence.

| Model | Pooled gross RMSE, bps | BTC RMSE, bps | ETH RMSE, bps |
| --- | ---: | ---: | ---: |
| Original ridge | 1.370547 | 1.428227 | 1.310332 |
| Enriched ridge | 1.385247 | 1.455054 | 1.311729 |
| Enriched boosting | 1.373798 | 1.423252 | 1.322495 |

The original ridge also has the lowest pooled forecast error. Larger static representation or nonlinear capacity has not improved the main forecast metric in this pilot.

## Research decision

Keep the original ridge and the flat control. The next priority is **causal flow history and signal horizon**, holding model complexity fixed: test multiscale OFI/activity histories first, then independently compare how much executable markout accumulates over longer holding horizons relative to trading costs. The full decision branches and rejection conditions are in [the research path](performance_research_path.md).

This priority is an inference from the results: the enriched models fail to improve pooled forecasts, while the tested active policies have little room for transaction costs. For the policies selected at 0.1 bps, fixed-fill break-even fees are only approximately 0.153–0.159 bps per side. These are conditional repricing calculations, not estimates of achievable live fees or results of retuning at those prices. A larger neural architecture is not the next justified experiment.

May 24–26 remain unacquired and uninspected. Before using them, freeze the next candidate/procedure and execution/data manifests. If they are used for development comparisons, reserve later untouched dates for formal final evaluation. Do not tune on this pilot's results and present a rerun as fresh OOS evidence.

Three dates and two historical intraday hours per asset cannot establish statistical significance, regime robustness, present-day performance, annualized returns or deployment readiness. The observations are more than three years old at the time of this run. Order sizes use fractional research constraints; venue lot rules, market impact, receive-time delays and real fill calibration remain outside this pilot. None of the repository's formal empirical promotion gates is claimed closed by it.

## Verification and identity

- 474 canonical direct tests pass, with zero optional-dependency skips; full filtered pytest also passes 474 tests.
- Ruff passes; mypy passes the checked seven modules and two scripts; the C++ replay smoke passes.
- All 954 recorded job artifacts were rehashed successfully. All 864 validation candidate records and 216 OOS records are present, with no missing or duplicate model/date/scenario combinations.
- Every saved model was reloaded before evaluation and reproduced its validation-prefix predictions exactly.
- Independent QA covered BTC ridge, BTC boosting and ETH boosting: 36 scenarios, 2,454,496 equity rows, 144 threshold candidates, 384 checkpoint forecasts and 16 fills matched directly to raw quotes. Maximum reconstructed PnL discrepancy was 4.0e-11 USDT. This validates accounting and provenance in that sample, not actual live-fill accuracy.
- Preparation peak RSS was 130,531,328 bytes. The 18 fitted/replayed/exported model jobs totalled 485.98 seconds, excluding acquisition, tape loading between folds and verification.

| Artifact | SHA-256 |
| --- | --- |
| Original protocol | `cdf2725bdb9635f66cd55ce06697280826e32d51513d1b7e0694847532a78429` |
| Coverage-amended protocol | `c4cd117451fd5e8bd251bfd1b6a0a6cc10c3a0804611e03f4cfae066325e17c9` |
| Local dataset manifest | `d2b702f25f5bb5ac84da48913927061c6cb113764633f34a75c9dcf06a8b4831` |
| Full local result summary | `e5b28fc577260c079dba8eadced2744243c23c5f9cd9bd62c3bfcc79894bbe1a` |

The [portable source snapshot](performance_pilot_20260907_source_integrity.json) preserves all archive and feature hashes. The result snapshot preserves the code fingerprint and runtime dependency versions. Local engineering and independent accounting evidence is under `artifacts/performance_pilot_20260907/`; detailed model evidence is under `results/performance_pilot_20260907_window_revision/`.
