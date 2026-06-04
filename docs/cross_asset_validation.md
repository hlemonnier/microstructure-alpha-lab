# Cross-Asset Validation

## Setup

The first cross-asset check uses the same Binance Vision pipeline on `BTCUSDT` and `ETHUSDT`:

```text
market: futures/um
dates: 2023-05-16 to 2023-05-17
bucket_ms: 1000
horizon_ms: 5000
execution_latency_ms: 1000
threshold: half_spread
max_quote_buckets: 3600 per day
datasets: bookTicker, aggTrades, bookDepth
```

ETHUSDT build command:

```bash
PYTHONPATH=src python3 -m lob_forge.cli build-range \
  --symbol ETHUSDT \
  --start 2023-05-16 \
  --end 2023-05-17 \
  --output-dir data/processed/eth_maker_horizon_5000_latency_1000 \
  --combined-output data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --bucket-ms 1000 \
  --horizon-ms 5000 \
  --execution-latency-ms 1000 \
  --threshold half_spread \
  --min-tick 0.01 \
  --max-quote-buckets 3600 \
  --with-book-depth
```

## ETHUSDT Data Summary

```text
rows: 7188
label_counts: {'0': 2478, '-1': 2399, '1': 2311}
rows_with_trades: 7010
rows_with_depth: 7156
spread_min_max: 0.00999999999999, 0.17
delta_mid_min_max: -4.2, 4.93
```

Conservative maker-entry fillability:

```text
long_fillable_rate: 0.383834
short_fillable_rate: 0.364775
```

## Threshold Walk-Forward

Command:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --taker-fee-bps 0 \
  --sort-by validation_net_pnl
```

Zero-fee result:

```text
fold 1: microprice_deviation > 0.15, validation_net_pnl 50.680000, test_net_pnl 35.930000
fold 2: microprice_deviation > 0.075, validation_net_pnl 45.280000, test_net_pnl 21.710000
summary test_net_pnl: 57.640000
summary test_break_even_fee_bps: 0.075444
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

## Logistic Walk-Forward

Command:

```bash
PYTHONPATH=src python3 -m lob_forge.cli logistic-walk-forward \
  data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv \
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

Zero-fee result:

```text
fold 1: alpha_threshold 0.15, validation_net_pnl 31.010000, test_net_pnl 25.120000
fold 2: alpha_threshold 0.025, validation_net_pnl 15.800000, test_net_pnl 19.690000
summary test_net_pnl: 44.810000
summary test_break_even_fee_bps: 0.061468
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

## Conservative Maker Entry

Command:

```bash
PYTHONPATH=src python3 -m lob_forge.cli walk-forward \
  data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv \
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
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

## Interpretation

ETHUSDT confirms the main BTC result but with a smaller zero-fee edge.

The threshold baseline beats the logistic baseline on this sample, which reinforces the decision to keep simple baselines central. Both models select `always_flat` at 5 bps taker fees. Conservative maker-entry also selects `always_flat` even at zero explicit fees, which suggests adverse-selection-heavy fills under the strict fillability proxy.

The cross-asset conclusion is not "there is no signal." It is more precise:

```text
There is short-horizon directional structure in BTCUSDT and ETHUSDT Binance Vision quote/trade/depth-band data, but simple spread-crossing or strict passive-entry rules do not produce robust tradable edge under realistic costs on this small sample.
```
