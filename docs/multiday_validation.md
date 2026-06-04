# Multi-Day Validation

## Purpose

The first result set used two one-hour daily slices. This experiment expands the same Binance Vision setup to four consecutive first-hour slices:

```text
dates: 2023-05-16 to 2023-05-19
symbols: BTCUSDT, ETHUSDT
market: Binance USD-M futures
bucket: 1s
horizon: 5s
execution latency: 1s
max_quote_buckets: 3600 per day
datasets: bookTicker, aggTrades, bookDepth
```

This is still not a full-market backtest, but it is a better falsification test than the two-day sample because the walk-forward protocol now has four test folds.

## Reproducibility

Build and run:

```bash
bash scripts/reproduce_multiday_results.sh all
```

Run only results from already-built data:

```bash
bash scripts/reproduce_multiday_results.sh results
```

The combined files are:

```text
data/processed/multiday_20230516_20230519_btc_5s_latency_1000/BTCUSDT-2023-05-16_2023-05-19-combined-features.csv
data/processed/multiday_20230516_20230519_eth_5s_latency_1000/ETHUSDT-2023-05-16_2023-05-19-combined-features.csv
```

## BTCUSDT Data Summary

```text
rows: 14376
label_counts: {'0': 5145, '-1': 4682, '1': 4549}
rows_with_trades: 14307
rows_with_depth: 12935
spread_min_max: 0.0999999999985, 2.4
delta_mid_min_max: -54.6, 59.5
```

## BTCUSDT Threshold Walk-Forward

Zero-fee result:

```text
fold 1: microprice_deviation > 0.05, test_net_pnl 755.400000
fold 2: microprice_deviation > 0.075, test_net_pnl 1177.200000
fold 3: microprice_deviation > 0.1, test_net_pnl 661.200000
fold 4: microprice_deviation > 0.15, test_net_pnl 1132.800000

summary test_net_pnl: 3726.600000
summary break_even_fee_bps: 0.109470
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
fold 3: always_flat
fold 4: always_flat
summary test_net_pnl: 0
```

Interpretation:

BTCUSDT is the strongest result so far. The same microprice-deviation family is selected across all four folds, and all zero-fee test folds are positive. But the average break-even taker fee is still only `0.109470` bps, which remains far below ordinary taker-fee assumptions.

## BTCUSDT Expected-Edge Walk-Forward

Zero-fee result:

```text
summary test_net_pnl: 2359.400000
summary break_even_fee_bps: 0.081053
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
fold 3: always_flat
fold 4: always_flat
summary test_net_pnl: 0
```

Interpretation:

Expected-edge regression is positive across the expanded BTC sample, but it underperforms the simpler microprice-deviation threshold baseline. This argues against adding model complexity before increasing data coverage and improving execution modeling.

## ETHUSDT Data Summary

```text
rows: 14376
label_counts: {'0': 5811, '-1': 4308, '1': 4257}
rows_with_trades: 13916
rows_with_depth: 12935
spread_min_max: 0.00999999999999, 0.17
delta_mid_min_max: -4.2, 4.93
```

## ETHUSDT Threshold Walk-Forward

Zero-fee result:

```text
fold 1: microprice_deviation > 0.1, test_net_pnl 44.470000
fold 2: microprice_deviation > 0, test_net_pnl 51.030000
fold 3: top_imbalance > 0.5, test_net_pnl 35.640000
fold 4: microprice_deviation > 0.025, test_net_pnl 27.740000

summary test_net_pnl: 158.880000
summary break_even_fee_bps: 0.069716
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
fold 3: always_flat
fold 4: always_flat
summary test_net_pnl: 0
```

## ETHUSDT Expected-Edge Walk-Forward

Zero-fee result:

```text
summary test_net_pnl: 138.750000
summary break_even_fee_bps: 0.078187
```

At 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
fold 3: always_flat
fold 4: always_flat
summary test_net_pnl: 0
```

Interpretation:

ETHUSDT remains positive before fees, but weaker than BTCUSDT. The expected-edge model is slightly below the threshold baseline in absolute PnL but slightly higher in break-even bps because it trades less.

## Conclusion

The four-day expansion strengthens the central conclusion:

```text
There is repeatable short-horizon microstructure structure in Binance Vision quote/trade/depth-band data, especially in BTCUSDT, but the simple taker strategies still require sub-basis-point fees.
```

The project should now prioritize:

- more calendar coverage,
- passive fill probability and queue assumptions,
- conditional expected net PnL only after the sample is much larger,
- true L2 capture if the goal shifts back toward DeepLOB.
