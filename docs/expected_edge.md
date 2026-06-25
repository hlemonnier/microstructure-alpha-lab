# Expected-Edge Modeling

## Purpose

Classification answers:

```text
Will the future mid move up, flat, or down?
```

Execution asks a harder question:

```text
Is the expected executable move large enough to pay spread, fees, slippage, and latency?
```

This experiment trains a dependency-free ridge regression model on realized taker edge magnitude. It fits two targets on the train window:

```text
long_gross_bps  = 10000 * (future_bid - entry_ask) / entry_ask
short_gross_bps = 10000 * (entry_bid - future_ask) / entry_bid
```

At decision time, it estimates:

```text
long_net_bps  = predicted_long_gross_bps  - 2 * (taker_fee_bps + slippage_bps)
short_net_bps = predicted_short_gross_bps - 2 * (taker_fee_bps + slippage_bps)
```

Then it trades the side with the larger predicted net edge only if it exceeds a validation-selected threshold.

## Command

```bash
PYTHONPATH=src python3 -m lob_forge.cli edge-walk-forward <feature_csv> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --l2 1 \
  --taker-fee-bps <fee> \
  --sort-by validation_net_pnl
```

`edge-walk-forward` streams by default, keeping only the current train/validation/test window in memory. Use `--no-stream` only for tiny debug CSVs or on a high-RAM machine.

The reproducibility script has a dedicated mode:

```bash
bash scripts/reproduce_core_results.sh edge
```

## BTCUSDT Result

Setup:

```text
dates: 2023-05-16 to 2023-05-17
horizon: 5s
latency: 1s
model: ridge expected-edge regression
l2: 1
features: default microstructure feature set
```

Zero-fee result:

```text
fold 1: ridge_expected_edge, edge_threshold_bps 0.1
  validation_net_pnl: 834.700000
  test_net_pnl: -156.300000

fold 2: always_flat
  validation_net_pnl: 0
  test_net_pnl: 0

summary test_net_pnl: -156.300000
summary break_even_fee_bps: -0.024308
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Interpretation:

BTC is the useful negative case. A model that looks economically strong on validation fails on the next test window. That suggests the realized-edge target is noisier than the simple directional threshold rule on this tiny two-day sample.

## ETHUSDT Result

Zero-fee result:

```text
fold 1: ridge_expected_edge, edge_threshold_bps 0.3
  validation_net_pnl: 31.540000
  test_net_pnl: 10.840000

fold 2: always_flat
  validation_net_pnl: 0
  test_net_pnl: 0

summary test_net_pnl: 10.840000
summary break_even_fee_bps: 0.076012
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Interpretation:

ETH gets a small positive zero-fee result, but it is weaker than the simple threshold walk-forward baseline and remains far below realistic taker-fee requirements.

## Research Consequence

Expected-edge regression is conceptually the right direction, but this first linear version does not produce a robust tradable rule.

The next higher-upside branches are:

- expand the sample from two one-hour slices to many days and regimes,
- model expected net PnL with richer nonlinear methods after the data size supports it,
- add passive fill probability as a target,
- use true historical L2 or live diff-depth capture if the project moves toward full-depth TCN/Transformer architectures.
