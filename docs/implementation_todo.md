# Full Implementation TODO

This checklist tracks every missing item found during the project audit. Check an item only when code, docs, and tests or a reproducible script exist.

## Data Sources And L2

- [x] Verify and document free L2 source priority with primary-source links: OKX, Bybit, Binance, Coinbase, Tardis.dev, Crypto Lake, FI-2010.
- [x] Add machine-readable local free/freemium API source catalog for paper fills, true-L2 smoke imports, and public quote/trade archives.
- [x] Add historical L2 source adapters for OKX.
- [x] Add historical L2 source adapters for Bybit.
- [x] Add sample/validation adapters for Tardis.dev CSV formats.
- [x] Add sample/validation adapters for Crypto Lake formats.
- [x] Add FI-2010 benchmark loader for neural LOB sanity checks.
- [x] Add schema validation for every L2 file: snapshot/delta flag, timestamps, side, price, size, and sequence/update IDs when available.
- [x] Add Parquet storage path for normalized L2 events and snapshots.
- [x] Add and verify laptop-safe tiny/quick expected-edge study profiles.
- [x] Add bounded-memory streaming expected-edge walk-forward so large combined CSVs do not have to be fully loaded into RAM.
- [x] Add feature-build memory preflights and make depth data optional for laptop profiles.
- [x] Add and verify an existing-features-only partial smoke artifact for already-built BTC/ETH days.
- [x] Add executable OKX/Bybit historical L2 manifest/download/import workflow with manual `direct_url` selection.
- [x] Add explicit run-plan dry-run and 64 GB cloud RAM gate for full expected-edge studies.
- [x] Add expected-edge study completion verifier driven by `run_plan.json`.
- [x] Add source-only cloud handoff package and cloud bootstrap/run/verify script.
- [x] Add executable Modal batch runner with persistent cloud volume storage for online full-study execution.
- [x] Add external-gate readiness checker for Modal/package/full-study/paper-fill prerequisites without starting paid cloud compute.
- [x] Make the Modal runner prefer `.venv/bin/modal` and distinguish Modal CLI installation from Modal authentication.
- [x] Add feature-coverage verifier for expected daily markers and combined feature CSVs.
- [x] Add dry-run-first runner for result evaluation on feature-complete jobs only.
- [x] Add existing-feature local16 result-matrix smoke runner with a 20-fold completion guard.
- [x] Make existing-feature result runners rerun artifacts whose audit fold count is below the configured evidence threshold.
- [x] Add runtime process-memory caps for laptop expected-edge profiles so a bad local run fails before exhausting swap.
- [x] Block accidental `local16_60day` execution, lower its laptop RAM budgets, disable depth by default, and expose a one-command Modal cloud path for the full study.
- [x] Add OKX public historical-data download-link resolver for L2 acquisition manifests.
- [x] Add Bybit public history-data orderBook resolver and `.data.zip` importer for L2 acquisition manifests.
- [x] Add row-capped historical L2 import smoke path for 16 GB laptop runs and allow evidence gates to use OKX or Bybit normalized L2 candidates.
- [x] Re-run the capped 60-day BTC/ETH local/cloud profile with current holdout-manifest and result-verifier metadata.
- [ ] Run the full uncapped 60-90 day multi-symbol expected-edge study on a high-RAM/cloud machine.
- [x] Expand coverage beyond first-hour slices to full-day or sampled multi-session data.

## L2 Replay And Live Collection

- [x] Implement normalized L2 event model.
- [x] Implement deterministic order book replay from snapshots and deltas.
- [x] Add sequence-gap detection and replay reset logic.
- [x] Add crossed-book and monotonic-side validation.
- [x] Add top-N tensor/snapshot extraction for LOB tensor models.
- [x] Add live collector specs and URL/subscription builders for Binance REST snapshot plus diff-depth WebSocket.
- [x] Add live collector specs and subscription builders for OKX snapshot/delta WebSocket.
- [x] Add live collector specs and subscription builders for Bybit snapshot/delta WebSocket.
- [x] Add live collector specs and subscription builders for Coinbase level2 WebSocket.
- [x] Add venue-specific live L2 message normalization and optional WebSocket capture that persists normalized rows and fails fast on sequence gaps.
- [x] Implement runnable live collector loops that connect, persist normalized rows, and resnapshot/reset on sequence gaps for Binance/OKX/Bybit/Coinbase.
- [x] Add paper/shadow decision logger for live validation.

## Execution And Passive Fill

- [x] Implement event-level taker latency replay beyond fixed bucket delay.
- [x] Implement passive queue-position assumptions.
- [x] Support cancellation-ahead assumptions for our order.
- [x] Support partial fills.
- [x] Support maker exit and inventory carry instead of forced taker exit.
- [x] Support cancel/replace when signal decays.
- [x] Add fill-probability calibration utility by spread, volatility, top imbalance, and trade intensity.
- [x] Add order constraints: tick size, lot size, min quantity, min notional, order rejection.
- [x] Add rate-limit and websocket-delay assumptions.
- [x] Add simulated-vs-live fill validation metric utility and synthetic unit test.
- [x] Add CSV-driven shadow/paper fill validation CLI that joins simulated and observed fills by `decision_id`.
- [x] Add OOS edge-model shadow-decision export and feature-to-market-event conversion for offline fill-validation dry runs.
- [x] Add observed paper/live fill import CLI that aggregates partial fills by `decision_id` or client order ID.
- [x] Add free/freemium paper-demo API source priority and raw-fill normalization for Bybit, OKX, Binance Spot Testnet, and Alpaca Paper exports.
- [x] Add observed-fill template generation, ignore blank templates during import, and enforce default shadow-fill error thresholds in evidence gates.
- [ ] Run simulated-vs-paper/live fill validation on real shadow or paper observations.

## Validation And Statistics

- [x] Add immutable final holdout workflow with manifest verification, required candidate hashing, and duplicate-evaluation locks.
- [x] Make holdout manifest verification resolve repository-relative source paths from the manifest location, not only from the current working directory.
- [x] Require final holdout manifests to pre-register `candidate_sha256` before `final-holdout-rule` evaluates the held-out data.
- [ ] Run and check in the declared immutable final holdout result artifact for the full selected candidate.
- [x] Add dependence-aware intervals with explicit inference grain; reduced E2E intervals are ledger-derived, while current walk-forward CSV audits remain fold-summary audits.
- [x] Add Bayesian posterior scoring: `P(mu > 0)`, `P(mu > cost_margin)`, and posterior Sharpe checks.
- [x] Add Brier score and ECE calibration metrics.
- [x] Add reliability tables/curves by predicted edge decile.
- [x] Run latency grid: `0ms`, `250ms`, `500ms`, `1000ms`, `2000ms`.
- [x] Run fee-near-actual sensitivity grid as a promotion gate.
- [x] Add `make verify-results` or an equivalent single verifier script for checked-in artifacts.
- [x] Add machine-readable evidence-gate checker for remaining cloud/live/L2 execution TODOs.
- [x] Add regime splits by time of day, weekday/weekend, funding windows, trend/chop, and stress/high-volume days.

## Risk, Portfolio, And Capacity

- [x] Add fixed-notional portfolio simulation with explicit capital base.
- [x] Report returns on capital, leverage, margin, exposure, trade concurrency, and max inventory.
- [x] Add capped expected-edge sizing.
- [x] Add capped fractional-Kelly sizing helper with explicit variance input.
- [x] Add OOS variance-stability gate and gated fractional-Kelly helper that returns zero unless variance evidence passes.
- [x] Enable fractional Kelly on real strategy artifacts only after the OOS variance-stability gate passes.
- [x] Add daily PnL, daily Sharpe, autocorrelation, and Calmar-like ratios.
- [x] Add daily loss / rolling PnL kill-switch.
- [x] Add inventory penalty and max inventory constraints.
- [x] Add PnL-by-notional capacity curves and edge-decay curves.
- [x] Add passive capacity estimates once queue/fill model exists.

## Modeling

- [x] Add optional sklearn/PyTorch model spec and builder hooks for serious experiments.
- [x] Add gradient boosting / random forest / XGBoost-style baselines where dependencies are available.
- [x] Add nonlinear expected-net-PnL models.
- [x] Add top-of-book sequence MLP/TCN alternative before true L2.
- [x] Add FI-2010 loader, top-N tensors, and gated LOB-CNN builder hook.
- [x] Add optional Transformer/TCN builder hooks.
- [x] Add model-readiness gate that requires accepted baseline evidence plus verified normalized L2 data before sequence/deep experiments.
- [x] Add executable Torch-backed L2 Transformer/TCN experiment runner and artifact validation gate.
- [x] Add dry-run-first cloud/Modal runner for Transformer/TCN L2 experiment artifacts.
- [x] Run Transformer/TCN experiments only after baseline and L2 paths are verified.
- [x] Add masked sequence pretraining batch utility.
- [x] Run self-supervised order-book pretraining smoke only after true L2 tensor data exists.

## Documentation And Claims

- [x] Keep all full-depth neural LOB claims gated on true L2/FI-2010 evidence.
- [x] Keep 5 bps taker research as a negative control, not a target strategy.
- [x] Remove stale Markdown that describes already implemented items as missing.
- [x] Keep all results labeled as sanity/pipeline evidence until 20+ OOS folds and cost gates pass.
