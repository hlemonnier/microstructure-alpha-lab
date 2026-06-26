# Regime Analysis

## Purpose

The first walk-forward results show weak short-horizon structure before fees and no viable taker strategy at realistic retail-like fees. Regime analysis asks a sharper question:

```text
Where does the gross signal appear, and is any market state close to economically usable?
```

This is a diagnostic layer, not a new claim of profitability. It bins a fixed rule by market state and reuses the same execution-cost evaluator as the baseline reports.

## Command Pattern

```bash
PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest <feature_csv> \
  --output <holdout-manifest.json> \
  --split-column source_date \
  --holdout-values <final-date> \
  --source-root "$PWD"
PYTHONPATH=src python3 -m lob_forge.cli regime <feature_csv> \
  --holdout-manifest <holdout-manifest.json> \
  --feature microprice_deviation \
  --threshold <threshold> \
  --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
  --bins 3 \
  --taker-fee-bps 0
```

The bins are quantile buckets over the selected regime feature. `spread_mean_5` is used instead of the raw one-second spread because BTCUSDT and ETHUSDT are often one tick wide, so raw spread buckets can be duplicated.

## BTCUSDT

Setup:

```text
dates: 2023-05-16 to 2023-05-17
horizon: 5s
latency: 1s
rule: microprice_deviation > 0.05 => long, < -0.05 => short
fees: 0 bps taker
```

Full-file fixed-rule economics:

```text
signals: 6782
gross_pnl: 2893.000000
net_pnl: 2893.000000
break_even_fee_bps: 0.079161
```

Regime summary:

```text
spread_mean_5 buckets:
  low:    net_pnl 912.700000,  break_even_fee_bps 0.074471
  middle: net_pnl 1386.200000, break_even_fee_bps 0.113506
  high:   net_pnl 594.100000,  break_even_fee_bps 0.049191

realized_volatility_5 buckets:
  low:    net_pnl 491.900000,  break_even_fee_bps 0.041129
  middle: net_pnl 1411.100000, break_even_fee_bps 0.114078
  high:   net_pnl 990.000000,  break_even_fee_bps 0.081040

notional_imbalance_1pct buckets:
  low:    net_pnl 1341.500000, break_even_fee_bps 0.108986
  middle: net_pnl 1145.200000, break_even_fee_bps 0.094282
  high:   net_pnl 406.300000,  break_even_fee_bps 0.033606
```

Interpretation:

- BTC gross edge is strongest in the middle volatility bucket, not the highest volatility bucket.
- Wider recent spread is worse, which is consistent with spread drag and noisy adverse-selection states.
- The highest positive 1 percent notional-imbalance bucket is weaker than the low/middle buckets. Aggregate depth-band imbalance is therefore better treated as a conditioning variable than as a naive "more bid depth means buy" rule.
- The best bucket still only reaches about `0.11` bps break-even taker fee, so the economic conclusion does not change.

## ETHUSDT

Setup:

```text
dates: 2023-05-16 to 2023-05-17
horizon: 5s
latency: 1s
rule: microprice_deviation > 0.15 => long, < -0.15 => short
fees: 0 bps taker
```

Full-file fixed-rule economics:

```text
signals: 6062
gross_pnl: 225.890000
net_pnl: 225.890000
break_even_fee_bps: 0.102880
```

Regime summary:

```text
spread_mean_5 buckets:
  low:    net_pnl 76.720000, break_even_fee_bps 0.104104
  middle: net_pnl 65.680000, break_even_fee_bps 0.090235
  high:   net_pnl 83.490000, break_even_fee_bps 0.114240

realized_volatility_5 buckets:
  low:    net_pnl 70.950000, break_even_fee_bps 0.107044
  middle: net_pnl 90.900000, break_even_fee_bps 0.116261
  high:   net_pnl 64.040000, break_even_fee_bps 0.085274

notional_imbalance_1pct buckets:
  low:    net_pnl 54.180000, break_even_fee_bps 0.074764
  middle: net_pnl 84.590000, break_even_fee_bps 0.114793
  high:   net_pnl 87.120000, break_even_fee_bps 0.118677
```

Interpretation:

- ETH has a smaller absolute gross edge, but its break-even fee by bucket is in the same sub-basis-point range as BTC.
- The middle realized-volatility bucket is again best.
- Depth imbalance is less adverse than BTC in this sample, but the best bucket is still far below common taker-fee levels.

## Research Consequence

The next branch should not be "train a bigger model." The better branch is conditional execution research:

- trade only in volatility/liquidity states with materially higher break-even fee,
- calibrate a passive fill probability model instead of using a binary crossing proxy,
- learn expected move magnitude, not only direction,
- require expected move to clear spread plus fee drag before emitting a signal.
