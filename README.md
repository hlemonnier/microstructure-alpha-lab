# Microstructure Alpha Lab

Research-grade Binance Vision microstructure project for a quant research portfolio.

## Thesis

The serious question is not "can we build a crypto trading bot?" It is:

Can Binance Vision quote, trade, and depth archives expose short-horizon microstructure signals that survive time-based validation, fees, spread, slippage, and latency assumptions?

## Data Reality

Binance Vision is the source of truth for v1: <https://data.binance.vision/>.

The archive does not expose historical full level-by-level Spot order books. For USD-M futures, it exposes:

- `bookTicker`: best bid/ask quote updates. This is the main source for mid-price labels and spread-aware execution math.
- `trades` / `aggTrades`: trade flow. This is the main source for signed flow, trade intensity, and volume pressure.
- `bookDepth`: aggregated depth bands, with columns like `timestamp, percentage, depth, notional`. This is not DeepLOB-style L2 data, but it is useful for regime and liquidity context.
- `klines`, mark/index/premium klines, funding, and metrics as slower context features.

So v1 is a Binance Vision quote/trade/depth-band microstructure lab. A true DeepLOB project still needs live depth stream capture or another historical full L2 source.

## V1 Scope

Primary market:

- USD-M futures: `data/futures/um`
- Symbols: `BTCUSDT`, `ETHUSDT`
- Datasets: `bookTicker`, `trades`, `bookDepth`

Initial research loop:

1. Download exact archive files from Binance Vision with checksum verification.
2. Inspect archive schemas without extracting large files manually.
3. Build quote/trade/depth-band features: spread, mid, microprice proxy, quote imbalance, trade imbalance, intensity, and aggregate liquidity asymmetry.
4. Label future mid-price moves from `bookTicker` using tick/spread-aware neutral bands.
5. Compare simple statistical baselines before any neural model.
6. Evaluate with purged walk-forward splits and cost/latency-aware PnL simulation.

## Quick Start

For the microstructure case study, start with [docs/microstructure_case_study.md](docs/microstructure_case_study.md).

For the implementation notes and runbooks:

- [docs/alpha_factory.md](docs/alpha_factory.md)
- [docs/historical_l2_sources.md](docs/historical_l2_sources.md)
- [docs/l2_data_sources.md](docs/l2_data_sources.md)
- [docs/l2_replay.md](docs/l2_replay.md)
- [docs/implementation_todo.md](docs/implementation_todo.md)
- [docs/full_study_cloud_run.md](docs/full_study_cloud_run.md)

For compact auditable result artifacts, see [results/current/](results/current/). To regenerate them from local processed data:

```bash
bash scripts/reproduce_result_artifacts.sh
```

To regenerate compact latency/fee/regime sensitivity artifacts from the current processed BTC/ETH samples:

```bash
bash scripts/run_current_sensitivity_grids.sh
```

To verify checked-in result artifacts:

```bash
make verify-results
```

To check the remaining execution-backed TODO gates and write a machine-readable status artifact:

```bash
make verify-evidence-gates
```

This gate intentionally exits non-zero until the full expected-edge study, real shadow-fill validation, and nonzero-cost Kelly promotion gates have live artifacts. The L2 gate checks the OKX BTC swap path first and the Bybit BTCUSDT path second by default; pass repeatable `--l2` values to override the candidate list. The current JSON status is written to `results/current/remaining_evidence_gates.json`.

To check operational readiness for the two remaining external gates without starting paid cloud compute or needing exchange credentials:

```bash
make external-readiness
```

This writes `results/current/external_gate_readiness.json` and reports Modal CLI availability, cloud handoff package hygiene, full-study completion state, and paper/live fill-observation readiness.

The larger review-recommended expected-edge study is scripted but intentionally not run by default. Do not run the uncapped profile on a 16 GB laptop; use the local smoke profiles or move the full job to cloud:

```bash
bash scripts/run_60day_expected_edge_study.sh
```

On a 16 GB laptop this command now uses the drastically scaled `laptop_tiny` profile: BTCUSDT only, two days, capped quote buckets, depth disabled, and a 4 GB feature-build memory guard. A larger but still capped laptop smoke run is available with:

```bash
STUDY_PROFILE=laptop_quick bash scripts/run_60day_expected_edge_study.sh
```

The laptop profiles also export `LOB_FORGE_MAX_PROCESS_MEMORY_GB` by default (`6` GB for `laptop_tiny`, `10` GB for `laptop_quick`, `8` GB for `local16_60day`). Heavy CLI children try to apply that address-space cap at startup, but macOS may reject some process limits, so the real safety comes from small row caps, depth disabled on the heavy local profile, and explicit profile confirmation. Set `LOB_FORGE_MAX_PROCESS_MEMORY_GB=0` only on a deliberate high-RAM run.

The shortest safe local command is:

```bash
make laptop-smoke
```

A heavier capped local run is available, but it is now blocked unless you explicitly confirm it. First inspect the plan:

```bash
PLAN_ONLY=1 STUDY_PROFILE=local16_60day bash scripts/run_60day_expected_edge_study.sh
```

Then run it only if you accept the local RAM/disk/network cost:

```bash
CONFIRM_LOCAL16=1 STUDY_PROFILE=local16_60day bash scripts/run_60day_expected_edge_study.sh
```

This profile is deliberately scaled down for a 16 GB laptop: BTC/ETH only, 5s horizon only, `MAX_QUOTE_BUCKETS=3600`, depth disabled by default, 8 GB process cap, 8 GB CSV-load guard, and 6 GB feature-build guard.

If 60-day features already exist but a full local scan is too slow, use the existing-feature runner. It defaults to dry-run and can produce bounded smoke artifacts with `MAX_FOLDS=1`:

```bash
bash scripts/run_local16_existing_feature_edge_jobs.sh
DRY_RUN=0 MAX_FOLDS=1 bash scripts/run_local16_existing_feature_edge_jobs.sh
```

The real evidence gate still requires at least 20 audit folds per artifact. Bounded smoke files are useful for wiring checks, but they do not satisfy the capped 60-day study gate.

The existing-feature runners rerun stale smoke artifacts by default when their audit `fold_count` is below `MIN_AUDIT_FOLD_COUNT`, so a 1-fold file matrix does not trap the study in a false-resumed state.

Current capped local16 evidence status: `results/expected_edge_local16_20230516_20230714/` verifies as complete with 10/10 result files and 10/10 audit files at the 20-fold audit threshold. That is the upper bound for laptop-backed evidence; the uncapped multi-symbol study still belongs on cloud.

The uncapped five-symbol/two-horizon study requires a high-RAM machine:

```bash
CONFIRM_HEAVY=1 STUDY_PROFILE=cloud_full bash scripts/run_60day_expected_edge_study.sh
```

The script uses bounded-memory streaming evaluation by default, RAM preflights, CSV load-memory guards for non-streaming runs, and feature-build memory guards before parsing daily ZIP files. See [docs/full_study_cloud_run.md](docs/full_study_cloud_run.md) for sizing, commands, and cleanup notes. The easiest managed cloud path is Modal:

```bash
python3 -m pip install ".[cloud]"
modal setup
make modal-study
MODE=run make modal-study
```

The Modal wrapper prefers `.venv/bin/modal` when available, otherwise it falls back to `modal` on `PATH`. The Modal job defaults to 8 CPU cores, 64 GiB requested RAM, a 128 GiB hard memory limit, and a persistent volume for `data/` and `results/`.

Direct `edge-walk-forward` CLI calls also stream by default now. Use `--no-stream` only for tiny debug CSVs or on a high-RAM machine.

The true-L2 lane now includes a runnable live capture MVP. It normalizes Binance/OKX/Bybit/Coinbase snapshot/delta payloads into the project L2 schema and fails fast on detected sequence gaps:

```bash
PYTHONPATH=src python3 -m lob_forge.cli live-l2-capture \
  --venue binance \
  --symbol BTCUSDT \
  --output data/live_l2/binance/BTCUSDT/session.csv \
  --seconds 60 \
  --depth 100 \
  --reset-on-gap
```

Install `.[research]` first for the optional WebSocket dependency.

For OKX historical L2, create a manifest and resolve public download links before download/import:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-manifest \
  --source okx \
  --symbols BTC-USDT-SWAP,ETH-USDT-SWAP \
  --start 2023-05-16 \
  --end 2023-07-14 \
  --output data/manifests/okx_l2_20230516_20230714.csv

PYTHONPATH=src python3 -m lob_forge.cli l2-resolve-okx \
  data/manifests/okx_l2_20230516_20230714.csv \
  --depth 400 \
  --throttle-seconds 2
```

For Bybit historical orderBook files, resolve through Bybit's public history-data file endpoint and import the returned `.data.zip` archives:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-manifest \
  --source bybit \
  --symbols BTCUSDT,ETHUSDT \
  --start 2023-05-16 \
  --end 2023-07-14 \
  --output data/manifests/bybit_l2_20230516_20230714.csv

PYTHONPATH=src python3 -m lob_forge.cli l2-resolve-bybit \
  data/manifests/bybit_l2_20230516_20230714.csv \
  --throttle-seconds 2

PYTHONPATH=src python3 -m lob_forge.cli l2-download-manifest \
  data/manifests/bybit_l2_20230516_20230714.csv \
  --raw-root data/raw

PYTHONPATH=src python3 -m lob_forge.cli l2-import-manifest \
  data/manifests/bybit_l2_20230516_20230714.csv \
  --output-root data \
  --format csv \
  --max-import-rows 5000 \
  --require-sequence
```

Drop `--max-import-rows` for the full normalized dataset. Keep it for laptop smoke tests; a Bybit daily orderBook ZIP can expand to a much larger normalized CSV even though the importer streams rows instead of holding the full day in RAM. The cap is event-preserving, so the importer may stop below the requested row count rather than split one exchange update in half.

For the bounded one-day Bybit L2 smoke path, run:

```bash
bash scripts/run_bybit_l2_smoke.sh
```

Override `MAX_IMPORT_ROWS`, `SYMBOL`, `START_DATE`, or `DOWNLOAD=0 IMPORT=0` when you only want to resolve the manifest without downloading/importing.

After a shadow or paper session, compare simulated fills against observed fills by `decision_id`. The dry-run chain below exports OOS edge-model decisions and simulated fills; for the real gate, import paper/live fills into the shadow file before running `validate-shadow-fills`:

```bash
PYTHONPATH=src python3 -m lob_forge.cli edge-shadow-decisions \
  data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --output results/shadow_validation/shadow_decisions.csv \
  --venue binance \
  --symbol BTCUSDT \
  --intended-notional 100 \
  --train-size 7200 \
  --validation-size 3600 \
  --test-size 3600 \
  --step-size 3600 \
  --taker-fee-bps 0

PYTHONPATH=src python3 -m lob_forge.cli features-to-market-events \
  data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --output results/shadow_validation/market_events.csv

PYTHONPATH=src python3 -m lob_forge.cli simulate-shadow-fills \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --market-events results/shadow_validation/market_events.csv \
  --output results/shadow_validation/simulated_fills.csv \
  --mode taker \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --min-quantity 0.001 \
  --min-notional 5

PYTHONPATH=src python3 -m lob_forge.cli observed-fill-template \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --output results/shadow_validation/observed_fills_template.csv \
  --limit 50

PYTHONPATH=src python3 -m lob_forge.cli import-observed-fills \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --observed results/shadow_validation/observed_fills.csv \
  --output results/shadow_validation/shadow_decisions_observed.csv

PYTHONPATH=src python3 -m lob_forge.cli validate-shadow-fills \
  --simulated results/shadow_validation/simulated_fills.csv \
  --shadow results/shadow_validation/shadow_decisions_observed.csv \
  --max-price-error 0.5 \
  --max-size-error 0.01 \
  --max-fill-rate-error 0.05
```

`observed-fill-template` creates the client-order/decision ID mapping to fill from a real paper/live export. A blank template is ignored by `import-observed-fills`; explicit `cumExecQty=0` rows count as real no-fill observations, while positive fill sizes require a fill price. `import-observed-fills` accepts common paper/exchange CSV aliases such as `decision_id`, `client_order_id`, `orderLinkId`, `clOrdId`, `avgPrice`, `fill_price`, `cumExecQty`, `executedQty`, `fill_size`, and `realizedPnl`. Multiple rows with the same client order ID are aggregated as partial fills.

The dry-run-first helper is:

```bash
bash scripts/prepare_shadow_fill_validation_session.sh
DRY_RUN=0 bash scripts/prepare_shadow_fill_validation_session.sh
```

Before using fractional Kelly sizing, require stable OOS fold variance:

```bash
PYTHONPATH=src python3 -m lob_forge.cli kelly-variance-gate \
  results/current/btc_full_day_edge_zero_fee.csv \
  --column test_net_pnl \
  --min-observations 20 \
  --window-size 5 \
  --max-variance-cv 0.5
```

The aggregate evidence gate is stricter than this standalone diagnostic: it scans the promoted local16 strategy artifacts and requires the same nonzero-cost candidate to pass both audit acceptance and variance stability. The default cost floor is `--kelly-min-cost-bps 0.05`; zero-fee artifacts cannot enable Kelly.

The current Kelly-eligible local candidate is reproducible with:

```bash
bash scripts/run_kelly_candidate_search.sh
DRY_RUN=0 bash scripts/run_kelly_candidate_search.sh
```

It writes `results/kelly_candidate_search/BTCUSDT_5000ms_fee_0p05_balanced_edge.csv` plus its audit. This candidate passes the local Kelly gate, but it is still capped local16 research evidence. Keep real sizing conservative until the full cloud study and paper/live fill validation pass.

Before running Transformer/TCN/DeepLOB experiments, require accepted baseline evidence and verified normalized L2 rows:

```bash
PYTHONPATH=src python3 -m lob_forge.cli model-readiness-gate \
  --model sequence_transformer \
  --baseline-audit results/current/btc_full_day_edge_zero_fee_audit.csv \
  --l2 data/normalized_l2/okx/BTC-USDT-SWAP/2023-05-16.csv \
  --min-fold-count 20 \
  --min-l2-rows 1000
```

If the Bybit resolver/import path is used first, point the gate at `data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv` instead.

Once the gate passes on a Torch-capable machine, write the required experiment artifacts with:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-sequence-experiment \
  --model sequence_transformer \
  --baseline-audit results/current/btc_full_day_edge_zero_fee_audit.csv \
  --l2 data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/model_experiments/sequence_transformer_results.csv \
  --min-fold-count 20 \
  --min-l2-rows 1000

PYTHONPATH=src python3 -m lob_forge.cli l2-sequence-experiment \
  --model sequence_tcn \
  --baseline-audit results/current/btc_full_day_edge_zero_fee_audit.csv \
  --l2 data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/model_experiments/sequence_tcn_results.csv \
  --min-fold-count 20 \
  --min-l2-rows 1000
```

`evidence-gates` requires those artifacts to match the selected L2 path, model name, dependency state, readiness state, and `passed=1`; nonempty placeholder files do not pass.

For a dry-run-first wrapper over both required artifacts:

```bash
bash scripts/run_l2_sequence_experiments.sh
DRY_RUN=0 bash scripts/run_l2_sequence_experiments.sh
```

The wrapper and evidence-gate verifier prefer `.venv/bin/python` when it exists, otherwise they fall back to `python3`. A local Torch-capable run has produced the current smoke artifacts in `results/model_experiments/`: `sequence_transformer` passed the artifact gate with `test_macro_f1=0.333333333333`, while `sequence_tcn` passed the artifact gate with `test_macro_f1=0`. Treat these as proof that the gated L2 model path runs, not as evidence of neural-model edge.

On Modal, after the expected-edge/L2 artifacts are present in the persistent volume:

```bash
MODE=sequence make modal-study
```

To write the dependency-free self-supervised L2 pretraining smoke artifact from verified normalized L2 rows:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-pretraining-smoke \
  --l2 data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/model_experiments/self_supervised_pretraining.csv \
  --depth 5 \
  --window 4
```

This produces a masked reconstruction baseline artifact. It is a tensor/readiness smoke, not a Torch Transformer/TCN training result.

To rerun the core verified result commands from processed data:

```bash
bash scripts/reproduce_core_results.sh results
```

Run the dependency-free verification suite:

```bash
bash scripts/run_tests.sh
```

Use the package directly from the source tree:

```bash
PYTHONPATH=src python3 -m lob_forge.cli datasets
PYTHONPATH=src python3 -m lob_forge.cli list --market futures/um --frequency daily --dataset bookDepth --symbol BTCUSDT --date 2023-01-01
PYTHONPATH=src python3 -m lob_forge.cli download --market futures/um --frequency daily --dataset bookDepth --symbol BTCUSDT --date 2023-01-01
PYTHONPATH=src python3 -m lob_forge.cli inspect data/raw/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2023-01-01.zip
```

Build the first quote/trade research sample:

```bash
PYTHONPATH=src python3 -m lob_forge.cli download --market futures/um --frequency daily --dataset bookTicker --symbol BTCUSDT --date 2023-05-16
PYTHONPATH=src python3 -m lob_forge.cli download --market futures/um --frequency daily --dataset aggTrades --symbol BTCUSDT --date 2023-05-16
PYTHONPATH=src python3 -m lob_forge.cli download --market futures/um --frequency daily --dataset bookDepth --symbol BTCUSDT --date 2023-05-16
PYTHONPATH=src python3 -m lob_forge.cli build-sample \
  --book-ticker-zip data/raw/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05-16.zip \
  --agg-trades-zip data/raw/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-05-16.zip \
  --book-depth-zip data/raw/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2023-05-16.zip \
  --output data/processed/BTCUSDT-2023-05-16-quote-trade-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --execution-latency-ms 0 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600
PYTHONPATH=src python3 -m lob_forge.cli describe-features data/processed/BTCUSDT-2023-05-16-quote-trade-features.csv
PYTHONPATH=src python3 -m lob_forge.cli baseline data/processed/BTCUSDT-2023-05-16-quote-trade-features.csv --top 10 --taker-fee-bps 5 --slippage-bps 0
```

Build and evaluate a small two-day robustness sample:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --combined-output data/processed/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --execution-latency-ms 0 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600 \
  --with-book-depth
PYTHONPATH=src python3 -m lob_forge.cli eval-rule data/processed/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --feature top_imbalance \
  --threshold 0.75 \
  --source-date 2023-05-17 \
  --taker-fee-bps 5
PYTHONPATH=src python3 -m lob_forge.cli baseline data/processed/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --top 5 \
  --taker-fee-bps 5 \
  --sort-by validation_net_pnl
```

Latency sensitivity uses the same command with a different output directory:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/latency_1000 \
  --combined-output data/processed/latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600
```

Horizon sensitivity uses the same pattern with `--horizon-ms 5000` or `--horizon-ms 10000`; see [docs/pipeline.md](docs/pipeline.md) for current results.

Cross-asset validation currently covers BTCUSDT and ETHUSDT under the same Binance Vision setup; see [docs/cross_asset_validation.md](docs/cross_asset_validation.md).

Regime diagnostics decompose a fixed rule by recent spread, volatility, and aggregate depth-band imbalance; see [docs/regime_analysis.md](docs/regime_analysis.md).

Conditional execution tests whether validation-selected regime filters survive the next test window; see [docs/conditional_execution.md](docs/conditional_execution.md).

Expected-edge modeling predicts executable long/short taker edge in bps and trades only when predicted net edge clears a threshold; see [docs/expected_edge.md](docs/expected_edge.md).

Multi-day validation expands the same setup to four BTCUSDT and ETHUSDT first-hour slices; see [docs/multiday_validation.md](docs/multiday_validation.md).

Passive fill diagnostics measure conservative maker-entry fill rate and adverse selection; see [docs/passive_fill_diagnostics.md](docs/passive_fill_diagnostics.md).

Fee sensitivity reports the best validation-economic rule at each taker-fee tier plus its break-even fee estimate:

```bash
PYTHONPATH=src python3 -m lob_forge.cli fee-sweep \
  data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --fees 0,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5 \
  --sort-by validation_net_pnl
```

Regime diagnostics use a fixed rule and equal-count bins over market-state columns:

```bash
PYTHONPATH=src python3 -m lob_forge.cli regime \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --feature microprice_deviation \
  --threshold 0.05 \
  --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
  --bins 3 \
  --taker-fee-bps 0
```

Conditional walk-forward selection lets the validation window choose either an unconditioned threshold rule, a regime-gated threshold rule, or `always_flat`:

```bash
PYTHONPATH=src python3 -m lob_forge.cli conditional-walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
  --regime-bins 3 \
  --min-validation-trades 50 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Expected-edge walk-forward trains a ridge model on realized long/short taker edge magnitude:

```bash
PYTHONPATH=src python3 -m lob_forge.cli edge-walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --l2 1 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Four-day validation can be reproduced with:

```bash
bash scripts/reproduce_multiday_results.sh all
```

Passive fill diagnostics can be reproduced with:

```bash
bash scripts/reproduce_multiday_results.sh fills
```

Passive fill diagnostics by regime bucket can be reproduced with:

```bash
bash scripts/reproduce_multiday_results.sh fill-regimes
```

Feature and threshold lists can be overridden when a signal family has a different natural scale:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/ofi_depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --features quote_ofi_normalized,quote_ofi_5_normalized,mid_return_1,mid_return_5,top_imbalance_mean_5 \
  --thresholds 0,0.00001,0.00005,0.001,0.01,0.05,0.1,0.25,0.5,1,2,5 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Purged walk-forward validation repeatedly selects on one validation window and scores the selected rule on the next window:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Execution model can be switched from taker/taker to a conservative maker-entry plus taker-exit model:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --execution-model maker_entry \
  --maker-fee-bps 0 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Multivariate classical baseline:

```bash
PYTHONPATH=src python3 -m lob_forge.cli logistic-walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --epochs 120 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Data files are intentionally ignored by git.

## Research Standard

The project should be judged by research discipline:

- No random train/test splits.
- No accuracy-only claims.
- No PnL without fees, spread, slippage, and latency.
- No DeepLOB claims unless the data is true level-by-level order book data.
- Report where the edge dies as clearly as where it appears.
