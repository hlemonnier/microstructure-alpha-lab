# Microstructure Case Study: Binance Vision Microstructure Research

## One-Line Summary

Built a reproducible crypto microstructure research stack on Binance Vision USD-M futures archives, then tested whether short-horizon quote/trade/depth-band signals survive walk-forward validation, latency, spread crossing, fee tiers, and conservative passive-execution assumptions.

## Why This Project Exists

The initial idea was a deep order-book project. The first research decision was to verify the data source instead of assuming it supported the desired model.

Binance Vision does not provide historical full level-by-level Spot order books. For USD-M futures, it provides:

- `bookTicker`: best bid/ask updates,
- `aggTrades`: aggregate taker flow,
- `bookDepth`: aggregate percentage-band depth, not full L2 tensors.

So the project was reframed correctly:

```text
Not a fake full-depth LOB replication.
Yes: a quote/trade/depth-band microstructure lab with cost-aware validation.
```

That reframing is a feature, not a compromise. It shows data-source skepticism and prevents a common quant-project failure mode: training a model on a dataset that does not actually support the claimed market structure.

## Data And Scope

Source:

```text
https://data.binance.vision/
```

Market:

```text
Binance USD-M futures
```

Symbols tested:

```text
BTCUSDT
ETHUSDT
```

Core datasets:

```text
bookTicker
aggTrades
bookDepth
```

Main validation sample:

```text
dates: 2023-05-16 to 2023-05-17
bucket: 1 second
horizon: 5 seconds
execution latency: 1 second
label threshold: max(0.5 * entry spread, min tick)
split: purged walk-forward
```

Expanded validation sample:

```text
dates: 2023-05-16 to 2023-05-19
symbols: BTCUSDT, ETHUSDT
max_quote_buckets: 3600 per day
split: four-fold purged walk-forward
```

## Pipeline

The repo implements a dependency-free first pass:

1. List and download exact Binance Vision archives.
2. Verify Binance-published SHA-256 checksums.
3. Inspect ZIP CSV schemas without manual extraction.
4. Build quote/trade/depth-band feature CSVs.
5. Add causal microstructure features:
   - top-of-book imbalance,
   - microprice deviation,
   - quote order-flow imbalance,
   - rolling five-bucket OFI,
   - signed taker-flow imbalance,
   - aggregate depth-band imbalance,
   - rolling spread and volatility context.
6. Label executable future mid-price movement after latency.
7. Evaluate:
   - univariate threshold baselines,
   - softmax logistic regression,
   - fee sweeps and break-even fee estimates,
   - taker/taker execution,
   - conservative maker-entry/taker-exit execution.

## Key Results

### BTCUSDT, 5s Horizon, 1s Latency

Threshold baseline, zero fees:

```text
fold 1 test_net_pnl: 222.600000
fold 2 test_net_pnl: 566.300000
summary test_net_pnl: 788.900000
summary break_even_fee_bps: 0.063335
```

Threshold baseline, 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Softmax logistic baseline, zero fees:

```text
fold 1 test_net_pnl: -141.500000
fold 2 test_net_pnl: 343.300000
summary test_net_pnl: 201.800000
summary break_even_fee_bps: 0.022382
```

Softmax logistic baseline, 5 bps taker fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

Conservative maker-entry model, zero explicit fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

### ETHUSDT, 5s Horizon, 1s Latency

Threshold baseline, zero fees:

```text
fold 1 test_net_pnl: 35.930000
fold 2 test_net_pnl: 21.710000
summary test_net_pnl: 57.640000
summary break_even_fee_bps: 0.075444
```

Softmax logistic baseline, zero fees:

```text
fold 1 test_net_pnl: 25.120000
fold 2 test_net_pnl: 19.690000
summary test_net_pnl: 44.810000
summary break_even_fee_bps: 0.061468
```

At 5 bps taker fees:

```text
threshold baseline: always_flat
logistic baseline: always_flat
```

Conservative maker-entry model, zero explicit fees:

```text
fold 1: always_flat
fold 2: always_flat
summary test_net_pnl: 0
```

### Four-Day Robustness Check

The four-day run expands the same first-hour setup to `2023-05-16` through `2023-05-19`.

BTCUSDT threshold baseline, zero fees:

```text
fold 1 test_net_pnl: 755.400000
fold 2 test_net_pnl: 1177.200000
fold 3 test_net_pnl: 661.200000
fold 4 test_net_pnl: 1132.800000
summary test_net_pnl: 3726.600000
summary break_even_fee_bps: 0.109470
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

At 5 bps taker fees:

```text
BTCUSDT: always_flat in every fold
ETHUSDT: always_flat in every fold
```

The detailed output is in [multiday_validation.md](multiday_validation.md). This strengthens the pre-fee signal finding, especially for BTCUSDT, while preserving the core economic conclusion.

## Passive Fill Diagnostics

Since taker fees kill the edge, the next obvious question is passive entry. The repo now measures conservative maker-entry fillability and adverse selection for fixed rules.

BTCUSDT, four-day `microprice_deviation` threshold `0.1`:

```text
signals: 12584
fills: 2858
fill_rate: 0.227114
net_pnl before explicit fees: -8265.800000
mean_net_bps_per_fill: -1.069475
win_rate: 0.074878
mean_fill_latency_ms: 2676.362491
```

ETHUSDT, four-day `microprice_deviation` threshold `0.1`:

```text
signals: 12765
fills: 2914
fill_rate: 0.228280
net_pnl before explicit fees: -534.120000
mean_net_bps_per_fill: -1.012167
win_rate: 0.079959
mean_fill_latency_ms: 2652.849691
```

The detailed output is in [passive_fill_diagnostics.md](passive_fill_diagnostics.md). This explains why passive entry is not a trivial fix: fills happen, but they are adverse-selection dominated under the current binary crossing proxy.

Regime-bucket fill diagnostics strengthen that conclusion:

```text
BTCUSDT best tested all-side bucket: mean_net_bps_per_fill -0.935581
ETHUSDT best tested all-side bucket: mean_net_bps_per_fill -0.853974
```

No tested spread, volatility, trade-flow, or aggregate-depth bucket turns passive entry positive before explicit fees.

## Regime Diagnostics

The repo now includes fixed-rule regime decomposition by recent spread, realized volatility, and aggregate 1 percent notional-depth imbalance:

```bash
PYTHONPATH=src python3 -m lob_forge.cli regime <feature_csv> \
  --feature microprice_deviation \
  --threshold <threshold> \
  --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
  --bins 3 \
  --taker-fee-bps 0
```

BTCUSDT fixed rule:

```text
rule: microprice_deviation threshold 0.05
full-file gross_pnl: 2893.000000
full-file break_even_fee_bps: 0.079161
best regime bucket: middle realized_volatility_5
best bucket net_pnl: 1411.100000
best bucket break_even_fee_bps: 0.114078
```

ETHUSDT fixed rule:

```text
rule: microprice_deviation threshold 0.15
full-file gross_pnl: 225.890000
full-file break_even_fee_bps: 0.102880
best regime bucket: high notional_imbalance_1pct
best bucket net_pnl: 87.120000
best bucket break_even_fee_bps: 0.118677
```

The detailed output is in [regime_analysis.md](regime_analysis.md). The important conclusion is that regime conditioning identifies where gross signal is concentrated, but even the best buckets remain sub-basis-point on taker break-even fees.

## Conditional Execution Check

The next test was stricter than fixed regime diagnostics. The `conditional-walk-forward` command lets the validation window select:

- an unconditioned threshold rule,
- a threshold rule gated by a spread/volatility/depth regime bucket,
- or `always_flat`.

The chosen rule is then scored on the next test window.

BTCUSDT zero-fee result:

```text
fold 1: top_imbalance_mean_5 > 0, no regime filter
fold 2: microprice_deviation > 0.05, no regime filter
summary test_net_pnl: 788.900000
summary break_even_fee_bps: 0.063335
```

ETHUSDT zero-fee result:

```text
fold 1: notional_imbalance_1pct > 0 inside spread_mean_5 bucket 1
  validation_net_pnl: 80.150000
  test_net_pnl: -20.860000
fold 2: microprice_deviation > 0.075, no regime filter
  test_net_pnl: 21.710000
summary test_net_pnl: 0.850000
summary break_even_fee_bps: 0.001045
```

At 5 bps taker fees, both BTCUSDT and ETHUSDT select `always_flat` in every fold.

The detailed output is in [conditional_execution.md](conditional_execution.md). This is a useful negative result: simple regime gating does not robustly transform the signal into a tradable taker strategy.

## Expected-Edge Check

The next branch changed the target from direction to executable edge magnitude. The `edge-walk-forward` command fits two ridge regressions:

```text
predicted long taker gross bps
predicted short taker gross bps
```

The decision rule subtracts estimated fee/slippage drag and trades only if predicted net edge clears a validation-selected threshold.

BTCUSDT zero-fee result:

```text
fold 1: ridge_expected_edge, edge_threshold_bps 0.1
  validation_net_pnl: 834.700000
  test_net_pnl: -156.300000
fold 2: always_flat
summary test_net_pnl: -156.300000
```

ETHUSDT zero-fee result:

```text
fold 1: ridge_expected_edge, edge_threshold_bps 0.3
  validation_net_pnl: 31.540000
  test_net_pnl: 10.840000
fold 2: always_flat
summary test_net_pnl: 10.840000
summary break_even_fee_bps: 0.076012
```

At 5 bps taker fees, both BTCUSDT and ETHUSDT select `always_flat` in every fold.

The detailed output is in [expected_edge.md](expected_edge.md). This is another useful falsification: linear expected-edge modeling is conceptually better aligned with trading, but it is not robust enough on the current sample.

## Interpretation

The result is not "there is no signal."

The correct conclusion is sharper:

```text
There is repeatable short-horizon directional structure in BTCUSDT and ETHUSDT Binance Vision quote/trade/depth-band data, but simple spread-crossing or strict passive-entry rules do not produce robust tradable edge under realistic costs on these first-hour samples.
```

The most important finding is economic, not predictive:

- Directional structure appears before fees.
- The BTCUSDT microprice-deviation family stays positive across four test folds.
- Break-even taker fees are sub-basis-point.
- Retail-style taker fees kill the edge.
- A conservative passive fill proxy is adverse-selection heavy, with filled passive entries losing about 1 bps per fill before explicit fees.
- Regime filtering increases fill-rate in some states but does not remove passive-entry adverse selection.
- Simple validation-selected regime gating does not rescue taker economics.
- Linear expected-edge regression is not robust enough to beat the simpler threshold baseline.
- Logistic regression does not dominate simple threshold baselines, so bigger models are not yet justified.

That is the kind of negative result that matters in quant research. It prevents wasting time on model complexity before execution economics are understood.

## Why This Is Relevant

This project demonstrates:

- skepticism about data quality and market-structure claims,
- reproducible archive-based data engineering,
- microstructure feature design,
- latency-aware labeling,
- walk-forward validation instead of random splits,
- transaction-cost-aware objective selection,
- fee-tier sensitivity and break-even fee analysis,
- humility around model complexity,
- ability to turn a vague HFT idea into a falsifiable research pipeline.

It also gives a clear next research path instead of a hand-wavy conclusion.

## Next Experiments

The next high-value branches are:

1. Expand from four first-hour slices to broader calendar regimes:
   - full-day samples,
   - high-volatility days,
   - low-volatility days,
   - funding-event days,
   - news-like shock days.
2. Model passive fills more realistically:
   - queue-position assumptions,
   - cancellation-ahead estimates,
   - maker exit,
   - fill probability calibration.
3. Add conditional trading:
   - estimate expected move magnitude on a much larger sample,
   - trade only when calibrated expected move exceeds spread plus fee drag.
4. Use live diff-depth capture or another full L2 source if a true full-depth neural extension is required.

## Reproducibility

Core commands are collected in:

```text
scripts/reproduce_core_results.sh
```

Run result checks from already-built processed data:

```bash
bash scripts/reproduce_core_results.sh results
```

Build the BTCUSDT and ETHUSDT two-day samples from Binance Vision:

```bash
bash scripts/reproduce_core_results.sh build
```

Run the four-day validation:

```bash
bash scripts/reproduce_multiday_results.sh all
```
