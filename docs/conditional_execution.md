# Conditional Execution

## Purpose

The regime analysis showed where gross signal concentrates, but a hindsight bucket table is not enough. This experiment asks whether a regime-conditioned rule can be selected on validation data and then survive the next test window.

The selector searches:

- unconditioned threshold rules,
- the same threshold rules gated by one regime-feature quantile bucket,
- `always_flat`.

Regime bucket boundaries are learned from the validation window and then applied unchanged to the test window. Rows outside the selected bucket are forced flat.

## Command

```bash
PYTHONPATH=src python3 -m lob_forge.cli conditional-walk-forward <feature_csv> \
  --train-size 2400 \
  --validation-size 1200 \
  --test-size 1200 \
  --step-size 1200 \
  --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
  --regime-bins 3 \
  --min-validation-trades 50 \
  --taker-fee-bps <fee> \
  --sort-by validation_net_pnl
```

The reproducibility script has a dedicated mode:

```bash
bash scripts/reproduce_core_results.sh conditional
```

## BTCUSDT Result

Setup:

```text
dates: 2023-05-16 to 2023-05-17
horizon: 5s
latency: 1s
regime features: spread_mean_5, realized_volatility_5, notional_imbalance_1pct
```

Zero-fee conditional selection:

```text
fold 1: top_imbalance_mean_5 > 0, no regime filter
  validation_net_pnl: 1021.700000
  test_net_pnl: 222.600000

fold 2: microprice_deviation > 0.05, no regime filter
  validation_net_pnl: 509.200000
  test_net_pnl: 566.300000

summary test_net_pnl: 788.900000
summary break_even_fee_bps: 0.063335
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Interpretation:

The conditional selector had access to regime filters but rejected them. For BTCUSDT, the unconditioned validation-economic rules remained best on this two-day sample.

## ETHUSDT Result

Zero-fee conditional selection:

```text
fold 1: notional_imbalance_1pct > 0 inside spread_mean_5 bucket 1
  validation_net_pnl: 80.150000
  test_net_pnl: -20.860000

fold 2: microprice_deviation > 0.075, no regime filter
  validation_net_pnl: 45.280000
  test_net_pnl: 21.710000

summary test_net_pnl: 0.850000
summary break_even_fee_bps: 0.001045
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Interpretation:

The first ETH fold is the useful failure case: a validation-selected regime filter overfit and lost money in the next test window. That is exactly why the project needs purged walk-forward selection rather than reporting the best hindsight regime table.

## Research Consequence

Simple regime gating does not create a robust tradable rule here.

The next credible branch is not more univariate slicing. It is one of:

- expected-move modeling, where the target is realized gross move or net PnL instead of only direction,
- probability calibration, where trades require `E[move] > spread + fee + slippage`,
- passive fill probability modeling with queue/cancellation assumptions,
- larger multi-day samples so the conditional selector has enough regime variation to avoid one-window overfit.
