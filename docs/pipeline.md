# Pipeline Notes

## Current State

The repo can now perform the first reproducible Binance Vision pipeline:

1. List archive objects by market, dataset, symbol, and date.
2. Download exact ZIP files.
3. Verify Binance-published SHA-256 checksums.
4. Inspect CSV schemas inside ZIP files.
5. Build a bucketed quote/trade/depth-band feature dataset from `bookTicker`, `aggTrades`, and optional `bookDepth`, with raw quote-event entry/exit resolution by default.
6. Summarize the output label balance and basic ranges.

## First Real Sample

Command:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-sample \
  --book-ticker-zip data/raw/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05-16.zip \
  --agg-trades-zip data/raw/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-05-16.zip \
  --output data/processed/BTCUSDT-2023-05-16-quote-trade-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600
```

Observed summary:

```text
rows: 3599
label_counts: {'0': 2383, '-1': 632, '1': 584}
rows_with_trades: 3587
spread_min_max: 0.0999999999985, 2.4
delta_mid_min_max: -28.45, 41.0
```

Interpretation:

- The first sample is intentionally small: about one hour of one-second buckets.
- The neutral class dominates, which is expected under a half-spread plus tick threshold.
- Buy/sell labels both exist, so the first classification task is non-degenerate.
- Almost every quote bucket has at least one aggregate trade.

## Feature Semantics

`bookTicker` is collapsed to the last quote in each fixed time bucket. Features include:

- bid/ask price and quantity,
- mid and spread,
- relative spread,
- top-of-book size imbalance,
- microprice and microprice deviation,
- quote update count in the bucket.
- quote order-flow imbalance between adjacent buckets,
- rolling five-bucket quote OFI, mid return, realized volatility, mean spread, and mean top imbalance.

`aggTrades` is aggregated into the same buckets:

- aggressive buy quantity/notional,
- aggressive sell quantity/notional,
- trade imbalance,
- trade count,
- large trade count.

For Binance `is_buyer_maker`:

- `false` means the buyer was the taker, so the trade is treated as aggressive buy flow.
- `true` means the seller was the taker, so the trade is treated as aggressive sell flow.

## Label Semantics

The label uses future mid-price from `bookTicker`.

For row time `t`, the signal features come from the quote bucket at `t`.

The executable entry quote is the first raw `bookTicker` quote event at or after:

```text
t + execution_latency_ms
```

The exit/future quote is the first raw `bookTicker` quote event at or after:

```text
t + execution_latency_ms + horizon_ms
```

Use `--execution-quote-resolution bucket` only when reproducing older studies that resolved execution from retained bucket quotes.

The label uses executable movement:

```text
delta_mid = future_mid - entry_mid
```

Default label threshold:

```text
theta_t = max(0.5 * entry_spread_t, min_tick)
```

Then:

```text
label = 1 if future_mid - entry_mid > theta_t
label = -1 if future_mid - entry_mid < -theta_t
label = 0 otherwise
```

## Next Step

The next serious step is a baseline model runner:

- create train/validation/test splits by contiguous time,
- fit simple threshold/logistic baselines on the generated CSV,
- report macro F1, balanced accuracy, calibration, and class confusion,
- then add a transaction-cost-aware decision rule.

The first dependency-free threshold baseline runner is now available:

```bash
PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest \
  data/processed/BTCUSDT-2023-05-16-quote-trade-features.csv \
  --output artifacts/holdout_manifests/manual/btcusdt_2023_05_16_event_time_holdout.json \
  --split-column event_time \
  --holdout-values <final-event-time> \
  --source-root "$PWD"
PYTHONPATH=src python3 -m lob_forge.cli baseline \
  data/processed/BTCUSDT-2023-05-16-quote-trade-features.csv \
  --holdout-manifest artifacts/holdout_manifests/manual/btcusdt_2023_05_16_event_time_holdout.json \
  --top 10
```

It evaluates simple directional rules over:

- `microprice_deviation`,
- `top_imbalance`,
- `trade_imbalance`,
- and an `always_flat` baseline.

This is intentionally primitive. Its job is to prevent fake progress: if a simple imbalance rule dominates, a heavier model is not justified yet.

Observed top result on the first one-hour BTCUSDT sample:

```text
top_imbalance_threshold at threshold 0.75
validation_macro_f1: 0.516085
validation_balanced_accuracy: 0.506326
test_macro_f1: 0.489508
test_balanced_accuracy: 0.481550
test_accuracy: 0.526389
test_coverage: 0.363889
```

Cost-aware result for the same rule on the test segment:

```text
zero fees:
  trades: 262
  gross_pnl: 235.500000
  mean_net_bps_per_trade: 0.332213
  win_rate: 0.534351

5 bps taker fee, zero extra slippage:
  trades: 262
  gross_pnl: 235.500000
  net_pnl: -6853.455950
  mean_net_bps_per_trade: -9.667793
  win_rate: 0.000000
```

This is the right kind of early negative result: a simple imbalance rule can look directionally useful before fees, but taker economics destroy it. It is not strong enough to claim economic value. It is enough to justify the next experiment:

- run the same pipeline across multiple days,
- compare results by volatility/spread regime,
- add cost-aware signal thresholds,
- and only then move to logistic/gradient boosting baselines.

## Two-Day Holdout Check

The range builder can download missing daily archives, build per-day samples, and concatenate them:

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
  --max-quote-buckets 3600
```

Combined sample summary:

```text
rows: 7198
label_counts: {'0': 5206, '-1': 1025, '1': 967}
rows_with_trades: 7181
```

Second day only:

```text
rows: 3599
label_counts: {'0': 2823, '1': 383, '-1': 393}
rows_with_trades: 3594
```

Fixed-rule holdout evaluation:

Rule selected from the first-day ranking:

```text
feature: top_imbalance
threshold: 0.75
```

Evaluated on `source_date=2023-05-17`:

```text
zero fees:
  n: 3599
  macro_f1: 0.542160
  balanced_accuracy: 0.619842
  accuracy: 0.668797
  coverage: 0.371770
  trades: 1338
  gross_pnl: 550.000000
  mean_net_bps_per_trade: 0.153210
  win_rate: 0.322870

5 bps taker fee, zero extra slippage:
  trades: 1338
  gross_pnl: 550.000000
  net_pnl: -35348.953900
  mean_net_bps_per_trade: -9.846786
  win_rate: 0.000000
```

Interpretation:

The rule generalizes directionally from the first one-hour sample to the next day's first one-hour sample, but its edge is tiny relative to taker fees. This pushes the project toward either:

- better selectivity and higher expected move thresholds,
- maker/passive execution modeling,
- longer horizons where fee drag is less dominant,
- or futures fee-tier sensitivity analysis.

## Economic Rule Selection

Classification-first ranking can be misleading. The CLI now supports sorting by the validation economic objective:

```bash
PYTHONPATH=src python3 -m lob_forge.cli baseline \
  data/processed/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --top 5 \
  --taker-fee-bps 5 \
  --sort-by validation_net_pnl
```

At 5 bps taker fees, the economically selected result is no-trade/no-coverage:

```text
rank 1: always_flat
validation_trades: 0
validation_net_pnl: 0
test_trades: 0
test_net_pnl: 0
```

The best classification-ranked active rule remains `top_imbalance > 0.75`, but under the same 5 bps taker-fee assumption:

```text
validation_trades: 563
validation_gross_pnl: 247.700000
validation_net_pnl: -14851.826550
test_trades: 499
test_gross_pnl: 198.100000
test_net_pnl: -13197.861650
```

At zero fees, validation-net ranking chooses active microprice/imbalance rules:

```text
rank 1: microprice_deviation > 0.1
validation_trades: 1282
validation_net_pnl: 300.400000
test_trades: 1249
test_net_pnl: 187.200000
```

This is exactly the intended research behavior: classification metrics can find weak directional structure, but the economic objective prevents pretending that a spread-crossing taker strategy exists when fees dominate the edge.

## Latency Sensitivity

The feature builder now separates:

- signal quote: `bid`, `ask`, `mid`, `top_imbalance`, `microprice_deviation`,
- executable entry quote: `entry_bid`, `entry_ask`, `entry_mid`,
- exit quote: `future_bid`, `future_ask`, `future_mid`.

Build a one-second execution-latency variant:

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

Observed combined sample:

```text
latency 0 ms:
  rows: 7198
  label_counts: {'0': 5206, '-1': 1025, '1': 967}

latency 1000 ms:
  rows: 7196
  label_counts: {'0': 5204, '-1': 1025, '1': 967}
```

Economic selection at zero fees:

```text
latency 0 ms, validation_net_pnl sort:
  rank 1: microprice_deviation > 0.1
  validation_net_pnl: 300.400000
  test_net_pnl: 187.200000

latency 1000 ms, validation_net_pnl sort:
  rank 1: top_imbalance > 0.5
  validation_net_pnl: 155.300000
  test_net_pnl: 125.500000
```

Economic selection at 5 bps taker fees with 1000 ms latency:

```text
rank 1: always_flat
validation_net_pnl: 0
test_net_pnl: 0
```

Interpretation:

Latency reduces the already-small gross edge. Fees still dominate. That makes the next high-value research branch either longer holding horizons, passive/maker execution assumptions, or stronger selectivity thresholds tied to expected move size rather than raw imbalance alone.

## Depth-Band Feature Join

`bookDepth` is now an optional feature input. It is joined as the latest aggregate depth snapshot at or before each quote event, using negative percentage bands as bid-side depth and positive percentage bands as ask-side depth. Generated columns include:

```text
depth_snapshot_age_ms
bid_depth_1pct, ask_depth_1pct, depth_imbalance_1pct
bid_depth_5pct, ask_depth_5pct, depth_imbalance_5pct
bid_notional_1pct, ask_notional_1pct, notional_imbalance_1pct
bid_notional_5pct, ask_notional_5pct, notional_imbalance_5pct
```

Build a two-day depth-enhanced sample:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/depth_1000_twoday \
  --combined-output data/processed/depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600 \
  --with-book-depth
```

Observed combined sample:

```text
rows: 7196
label_counts: {'0': 5204, '-1': 1025, '1': 967}
rows_with_trades: 7179
rows_with_depth: 7164
```

Zero-fee economic ranking over the expanded feature grid still selects top-of-book imbalance first:

```text
rank 1: top_imbalance > 0.5
validation_net_pnl: 155.300000
test_net_pnl: 125.500000
validation_break_even_fee_bps: 0.029971
test_break_even_fee_bps: 0.027179
```

Depth bands do appear in the classification ranking on the one-day sample, for example `depth_imbalance_5pct > 0.05` reached test net PnL of `89.100000` at zero fees, but its validation gross PnL was negative. Under 5 bps taker fees, the two-day economic selector remains:

```text
rank 1: always_flat
validation_net_pnl: 0
test_net_pnl: 0
```

Interpretation:

Aggregate depth bands are useful as liquidity-regime features, but this first implementation does not overturn the taker-fee conclusion. The next serious depth experiment should use depth features as conditioning variables or interactions, not as standalone threshold rules only.

## Quote OFI And Rolling Context

The feature builder now computes causal quote order-flow imbalance and rolling context features:

```text
quote_ofi
quote_ofi_normalized
quote_ofi_5
quote_ofi_5_normalized
mid_return_1
mid_return_5
realized_volatility_5
spread_mean_5
top_imbalance_mean_5
```

Rebuild the two-day depth-enhanced sample with the expanded schema:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/ofi_depth_1000_twoday \
  --combined-output data/processed/ofi_depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600 \
  --with-book-depth
```

The custom feature/threshold CLI options are useful because OFI and returns are not on the same scale as imbalance:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/ofi_depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --features quote_ofi_normalized,quote_ofi_5_normalized,mid_return_1,mid_return_5,top_imbalance_mean_5 \
  --thresholds 0,0.00001,0.000025,0.00005,0.0001,0.001,0.01,0.025,0.05,0.1,0.25,0.5,0.75,1,2,5 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

OFI/rolling-only purged walk-forward result at zero fees:

```text
fold 1: quote_ofi_5_normalized > 0.05, validation_net_pnl 218.100000, test_net_pnl 72.400000
fold 2: quote_ofi_5_normalized > 1, validation_net_pnl 83.700000, test_net_pnl 64.500000
summary test_net_pnl: 136.900000
summary test_break_even_fee_bps: 0.014132
```

At 5 bps taker fees, the same OFI/rolling-only protocol selects `always_flat` in both folds.

Interpretation:

Rolling quote OFI is directionally useful on this small sample, but weaker than the broader microprice/imbalance grid. That is still a valuable result: the pipeline can now test canonical microstructure variables directly, and it shows which simple signals deserve model capacity.

## Conservative Maker-Entry Execution

The feature builder now records whether a passive maker entry would be conservatively fillable inside the holding horizon:

```text
horizon_min_ask
horizon_max_bid
maker_long_fillable
maker_short_fillable
maker_long_fill_event_time
maker_short_fill_event_time
```

The model is intentionally strict:

- long signal posts at `entry_bid`; it is fillable only if a later best ask crosses down to `entry_bid`,
- short signal posts at `entry_ask`; it is fillable only if a later best bid crosses up to `entry_ask`,
- filled positions exit at the horizon with taker execution.

Build maker-aware one-second and five-second samples:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/maker_ofi_depth_1000_twoday \
  --combined-output data/processed/maker_ofi_depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 1000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600 \
  --with-book-depth
```

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/maker_horizon_5000_latency_1000 \
  --combined-output data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 5000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600 \
  --with-book-depth
```

Observed fillability:

```text
1s horizon, 1s latency:
  rows: 7196
  long_fillable_rate: 0.146748
  short_fillable_rate: 0.138966

5s horizon, 1s latency:
  rows: 7188
  long_fillable_rate: 0.374513
  short_fillable_rate: 0.369087
```

Run maker-entry walk-forward selection:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --execution-model maker_entry \
  --maker-fee-bps 0 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Result:

```text
1s horizon, zero maker fee, zero taker exit fee:
  fold 1: always_flat
  fold 2: always_flat
  summary test_net_pnl: 0

5s horizon, zero maker fee, zero taker exit fee:
  fold 1: always_flat
  fold 2: always_flat
  summary test_net_pnl: 0

5s horizon, zero maker fee, 5 bps taker exit fee:
  fold 1: always_flat
  fold 2: always_flat
  summary test_net_pnl: 0
```

Interpretation:

The strict maker-fill proxy is adverse-selection heavy: fills tend to happen when price moves against the passive order. Under this conservative model, passive entry does not rescue the simple threshold rules. A stronger passive study needs queue-position assumptions, fill probability modeling, cancellation logic, and possibly a maker-exit or inventory-holding model rather than forced taker exit.

## Multivariate Logistic Baseline

The project now includes a dependency-free softmax logistic regression baseline. It standardizes features using train-window statistics only, uses optional balanced class weights, then selects an alpha threshold on validation:

```text
alpha = p(up) - p(down)
long if alpha > threshold
short if alpha < -threshold
flat otherwise
```

Run purged walk-forward logistic regression:

```bash
PYTHONPATH=src python3 -m lob_forge.cli logistic-walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --epochs 120 \
  --learning-rate 0.05 \
  --l2 0.001 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

5s horizon, 1s latency, zero fees:

```text
fold 1: alpha_threshold 0.15, validation_net_pnl 937.100000, test_net_pnl -141.500000
fold 2: alpha_threshold 0.5, validation_net_pnl 185.500000, test_net_pnl 343.300000
summary test_net_pnl: 201.800000
summary test_break_even_fee_bps: 0.022382
```

5s horizon, 1s latency, 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Interpretation:

The multivariate baseline is useful as a sanity check, but it does not dominate the simple threshold baselines on this small sample. Fold 1 overfits validation badly: high validation PnL becomes negative test PnL. This is exactly why the project should keep simple baselines and purged walk-forward selection central before moving to neural architectures.

## Cross-Asset Check

The same two-day, 5s-horizon, 1s-latency pipeline has now been run on `ETHUSDT`. The result is directionally consistent with BTCUSDT: zero-fee structure exists, threshold baselines beat the logistic baseline on this sample, 5 bps taker fees select `always_flat`, and conservative maker-entry also selects `always_flat`.

See [cross_asset_validation.md](cross_asset_validation.md) for the ETHUSDT build command and exact fold outputs.

## Holding-Horizon Sensitivity

To test whether the edge is too short-horizon to pay fees, build longer holding horizons while keeping one-second execution latency:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/horizon_5000_latency_1000 \
  --combined-output data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 5000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600
```

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/horizon_10000_latency_1000 \
  --combined-output data/processed/horizon_10000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 10000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.1 \
  --max-quote-buckets 3600
```

Label balance:

```text
1s horizon, 1s latency:
  rows: 7196
  label_counts: {'0': 5204, '-1': 1025, '1': 967}

5s horizon, 1s latency:
  rows: 7188
  label_counts: {'0': 2444, '-1': 2384, '1': 2360}

10s horizon, 1s latency:
  rows: 7178
  label_counts: {'-1': 2873, '1': 3039, '0': 1266}
```

Zero-fee economic selection:

```text
5s horizon:
  rank 1: top_imbalance > 0.025
  validation_net_pnl: 665.000000
  test_net_pnl: 508.700000

10s horizon:
  rank 1: top_imbalance > 0.025
  validation_net_pnl: 543.400000
  test_net_pnl: 555.300000
```

5 bps taker-fee economic selection:

```text
5s horizon:
  rank 1: always_flat
  validation_net_pnl: 0
  test_net_pnl: 0

10s horizon:
  rank 1: always_flat
  validation_net_pnl: 0
  test_net_pnl: 0
```

Interpretation:

Longer horizons make the labels less neutral and increase gross PnL before fees. They still do not overcome a 5 bps taker-fee assumption in this small two-day sample. The next experiment should not be "try a bigger model" yet. It should be one of:

- fee-tier sensitivity from realistic USD-M futures maker/taker schedules,
- selective trading rules based on expected move versus break-even cost,
- passive/maker execution with conservative fill assumptions,
- or multi-day validation across volatility regimes to find when gross edge is large enough to matter.

## Fee-Tier Sensitivity

The baseline runner now reports `break_even_fee_bps`, defined as the per-side taker fee that would make the selected rule net flat after the configured slippage assumption:

```text
break_even_taker_fee_bps = gross_pnl / sum(entry_price + exit_price) * 10000 - slippage_bps
```

Run a fee sweep:

```bash
PYTHONPATH=src python3 -m lob_forge.cli fee-sweep \
  data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --fees 0,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5 \
  --sort-by validation_net_pnl
```

5s horizon, 1s latency:

```text
fee 0.000 bps: top_imbalance > 0.025, validation_net_pnl 665.000000, test_net_pnl 508.700000, validation_break_even_fee_bps 0.087494, test_break_even_fee_bps 0.066910
fee 0.050 bps: top_imbalance > 0.5, validation_net_pnl 330.654093, test_net_pnl 252.659639, validation_break_even_fee_bps 0.113945, test_break_even_fee_bps 0.104845
fee 0.100 bps: top_imbalance > 0.75, validation_net_pnl 173.682762, test_net_pnl 72.619192, validation_break_even_fee_bps 0.157718, test_break_even_fee_bps 0.127159
fee 0.250 bps: always_flat, validation_net_pnl 0, test_net_pnl 0
```

10s horizon, 1s latency:

```text
fee 0.000 bps: top_imbalance > 0.025, validation_net_pnl 543.400000, test_net_pnl 555.300000, validation_break_even_fee_bps 0.071597, test_break_even_fee_bps 0.073142
fee 0.050 bps: top_imbalance > 0.75, validation_net_pnl 219.309227, test_net_pnl 322.215158, validation_break_even_fee_bps 0.123010, test_break_even_fee_bps 0.171238
fee 0.100 bps: top_imbalance > 0.75, validation_net_pnl 69.118455, test_net_pnl 189.330315, validation_break_even_fee_bps 0.123010, test_break_even_fee_bps 0.171238
fee 0.250 bps: always_flat, validation_net_pnl 0, test_net_pnl 0
```

Interpretation:

The longer-horizon taker rules are not dead at all fee levels; they are dead at ordinary taker-fee levels. On this two-day BTCUSDT sample, active rules require roughly sub-0.2 bps per-side taker costs before they beat the flat baseline. That is a useful microstructure-research result: the project is already separating directional predictability from tradable edge.

## Purged Walk-Forward Validation

The `walk-forward` command now repeats the selection/test process across contiguous windows:

1. Use a historical train window for reporting.
2. Select the best threshold rule on the next validation window.
3. Score that selected rule on the following test window.
4. Purge rows whose `future_event_time` crosses the next window boundary, so label horizons do not leak across validation/test boundaries.

Example:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/depth_1000_twoday/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Depth-enhanced 1s horizon, 1s latency, zero fees:

```text
fold 1: microprice_deviation > 0.075, validation_net_pnl 270.700000, test_net_pnl 85.100000
fold 2: microprice_deviation > 0.3, validation_net_pnl 105.700000, test_net_pnl 152.300000
summary test_net_pnl: 237.400000
summary test_break_even_fee_bps: 0.024878
```

Depth-enhanced 1s horizon, 1s latency, 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

5s horizon, 1s latency, zero fees:

```text
fold 1: top_imbalance_mean_5 > 0, validation_net_pnl 1021.700000, test_net_pnl 222.600000
fold 2: microprice_deviation > 0.05, validation_net_pnl 509.200000, test_net_pnl 566.300000
summary test_net_pnl: 788.900000
summary test_break_even_fee_bps: 0.063335
```

5s horizon, 1s latency, 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Interpretation:

The purged walk-forward protocol strengthens the same conclusion instead of weakening it. There is repeatable zero-fee directional structure, especially at 5s, but the selected taker rules need sub-basis-point economics. A credible next step is a maker/passive fill model or a conditional model that trades only when expected move materially exceeds spread plus fee drag.

## Regime Diagnostics

The `regime` command evaluates one fixed rule inside quantile buckets of market-state columns:

```bash
PYTHONPATH=src python3 -m lob_forge.cli regime \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --feature microprice_deviation \
  --threshold 0.05 \
  --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
  --bins 3 \
  --taker-fee-bps 0
```

BTCUSDT, 5s horizon, 1s latency, zero fees:

```text
rule: microprice_deviation > 0.05
full-file net_pnl: 2893.000000
full-file break_even_fee_bps: 0.079161
best bucket: middle realized_volatility_5
best bucket net_pnl: 1411.100000
best bucket break_even_fee_bps: 0.114078
```

ETHUSDT, 5s horizon, 1s latency, zero fees:

```text
rule: microprice_deviation > 0.15
full-file net_pnl: 225.890000
full-file break_even_fee_bps: 0.102880
best bucket: high notional_imbalance_1pct
best bucket net_pnl: 87.120000
best bucket break_even_fee_bps: 0.118677
```

Interpretation:

Regime conditioning is useful diagnostically but does not rescue taker economics. The strongest BTC bucket is middle realized volatility, not the highest volatility state. The strongest ETH bucket is high 1 percent notional-depth imbalance. Both remain sub-basis-point break-even taker-fee regimes, so the next serious branch is conditional expected-move modeling plus a better passive-fill model, not a larger classifier.

## Conditional Walk-Forward Selection

The `conditional-walk-forward` command turns regime diagnostics into an actual model-selection test:

1. Build regime bucket boundaries on the validation window.
2. Evaluate unconditioned threshold rules, regime-gated threshold rules, and `always_flat`.
3. Select by validation net PnL.
4. Score the selected rule on the next test window.

Example:

```bash
PYTHONPATH=src python3 -m lob_forge.cli conditional-walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
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

BTCUSDT, zero fees:

```text
fold 1: top_imbalance_mean_5 > 0, no regime filter, test_net_pnl 222.600000
fold 2: microprice_deviation > 0.05, no regime filter, test_net_pnl 566.300000
summary test_net_pnl: 788.900000
summary break_even_fee_bps: 0.063335
```

ETHUSDT, zero fees:

```text
fold 1: notional_imbalance_1pct > 0 inside spread_mean_5 bucket 1, test_net_pnl -20.860000
fold 2: microprice_deviation > 0.075, no regime filter, test_net_pnl 21.710000
summary test_net_pnl: 0.850000
summary break_even_fee_bps: 0.001045
```

At 5 bps taker fees, BTCUSDT and ETHUSDT both select `always_flat` in every fold.

Interpretation:

This check is harsher than the fixed regime table and more useful. BTC rejects simple regime filters. ETH shows validation overfit in one fold. Simple regime gating is not enough; the next candidate should estimate expected move or expected net PnL directly and require the forecast to clear cost drag.

## Expected-Edge Walk-Forward

The `edge-walk-forward` command directly models executable taker edge:

```text
long_gross_bps  = 10000 * (future_bid - entry_ask) / entry_ask
short_gross_bps = 10000 * (entry_bid - future_ask) / entry_bid
```

It fits two ridge regressions on the train window, subtracts estimated fee/slippage drag, then selects a predicted net-edge threshold on validation.

Example:

```bash
PYTHONPATH=src python3 -m lob_forge.cli edge-walk-forward \
  data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest <holdout-manifest.json> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --l2 1 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

BTCUSDT, zero fees:

```text
fold 1: ridge_expected_edge, edge_threshold_bps 0.1, validation_net_pnl 834.700000, test_net_pnl -156.300000
fold 2: always_flat
summary test_net_pnl: -156.300000
summary break_even_fee_bps: -0.024308
```

ETHUSDT, zero fees:

```text
fold 1: ridge_expected_edge, edge_threshold_bps 0.3, validation_net_pnl 31.540000, test_net_pnl 10.840000
fold 2: always_flat
summary test_net_pnl: 10.840000
summary break_even_fee_bps: 0.076012
```

At 5 bps taker fees, both symbols select `always_flat` in every fold.

Interpretation:

The expected-edge target is more economically honest than pure classification, but the linear version is not robust on this small sample. BTC overfits. ETH is positive but weaker than the simple threshold baseline. This pushes the next research branch toward more data and passive-fill modeling before nonlinear expected-PnL models.

## Four-Day Validation

The multi-day script expands the same setup to four consecutive first-hour samples:

```bash
bash scripts/reproduce_multiday_results.sh all
```

Setup:

```text
dates: 2023-05-16 to 2023-05-19
symbols: BTCUSDT, ETHUSDT
bucket_ms: 1000
horizon_ms: 5000
execution_latency_ms: 1000
max_quote_buckets: 3600 per day
walk-forward: train 3600, validation 1800, test 1800, step 1800
```

BTCUSDT threshold baseline, zero fees:

```text
fold 1 test_net_pnl: 755.400000
fold 2 test_net_pnl: 1177.200000
fold 3 test_net_pnl: 661.200000
fold 4 test_net_pnl: 1132.800000
summary test_net_pnl: 3726.600000
summary break_even_fee_bps: 0.109470
```

BTCUSDT expected-edge, zero fees:

```text
summary test_net_pnl: 2359.400000
summary break_even_fee_bps: 0.081053
```

ETHUSDT threshold baseline, zero fees:

```text
fold 1 test_net_pnl: 44.470000
fold 2 test_net_pnl: 51.030000
fold 3 test_net_pnl: 35.640000
fold 4 test_net_pnl: 27.740000
summary test_net_pnl: 158.880000
summary break_even_fee_bps: 0.069716
```

ETHUSDT expected-edge, zero fees:

```text
summary test_net_pnl: 138.750000
summary break_even_fee_bps: 0.078187
```

At 5 bps taker fees:

```text
BTCUSDT threshold: always_flat in all folds
BTCUSDT expected-edge: always_flat in all folds
ETHUSDT threshold: always_flat in all folds
ETHUSDT expected-edge: always_flat in all folds
```

Interpretation:

The larger sample strengthens the pre-fee signal finding, especially for BTCUSDT: microprice-deviation threshold rules are positive in all four test folds. It also strengthens the cost conclusion. The break-even fee remains around `0.07` to `0.11` bps, so normal taker execution is still not viable. The next serious engineering branch is passive execution and fill-probability modeling, not a larger classifier on the same target.

## Passive Fill Diagnostics

The `fill-diagnostics` command evaluates the conservative maker-entry proxy for a fixed threshold rule:

```bash
PYTHONPATH=src python3 -m lob_forge.cli fill-diagnostics \
  data/processed/multiday_20230516_20230519_btc_5s_latency_1000/BTCUSDT-2023-05-16_2023-05-19-combined-features.csv \
  --feature microprice_deviation \
  --threshold 0.1 \
  --by-source-date \
  --maker-fee-bps 0 \
  --taker-fee-bps 0
```

BTCUSDT, four-day aggregate:

```text
signals: 12584
fills: 2858
fill_rate: 0.227114
net_pnl: -8265.800000
mean_net_bps_per_fill: -1.069475
win_rate: 0.074878
mean_fill_latency_ms: 2676.362491
```

ETHUSDT, four-day aggregate:

```text
signals: 12765
fills: 2914
fill_rate: 0.228280
net_pnl: -534.120000
mean_net_bps_per_fill: -1.012167
win_rate: 0.079959
mean_fill_latency_ms: 2652.849691
```

Interpretation:

Passive orders do fill under the crossing proxy, but fills are adverse-selection dominated. The filled passive entries lose about 1 bps per fill before explicit fees, and filled win rates are below 9%. The next execution model needs queue position, cancellation, partial fills, maker exit, and signal-decay cancel logic; loosening the binary fill proxy without modeling those mechanics would create fake edge.

### Fill Regime Buckets

The `fill-regime` command tests whether passive-entry adverse selection is concentrated in specific market states:

```bash
PYTHONPATH=src python3 -m lob_forge.cli fill-regime \
  data/processed/multiday_20230516_20230519_btc_5s_latency_1000/BTCUSDT-2023-05-16_2023-05-19-combined-features.csv \
  --feature microprice_deviation \
  --threshold 0.1 \
  --regime-features spread_mean_5,realized_volatility_5,trade_imbalance,notional_imbalance_1pct \
  --bins 3 \
  --maker-fee-bps 0 \
  --taker-fee-bps 0
```

BTCUSDT:

```text
least bad bucket: realized_volatility_5 bucket 2
fill_rate: 0.194923
mean_net_bps_per_fill: -0.935581
win_rate: 0.036047

highest fill-rate bucket: realized_volatility_5 bucket 3
fill_rate: 0.306980
mean_net_bps_per_fill: -1.150885
win_rate: 0.124517
```

ETHUSDT:

```text
least bad bucket: spread_mean_5 bucket 2
fill_rate: 0.216600
mean_net_bps_per_fill: -0.853974
win_rate: 0.058952

highest fill-rate bucket: realized_volatility_5 bucket 3
fill_rate: 0.329977
mean_net_bps_per_fill: -1.068377
win_rate: 0.115811
```

Interpretation:

No tested regime bucket turns passive entry positive before explicit fees. High-volatility buckets fill more often, but their filled trades are still adverse-selection dominated.
