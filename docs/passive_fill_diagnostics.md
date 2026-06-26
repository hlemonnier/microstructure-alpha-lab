# Passive Fill Diagnostics

## Purpose

Taker execution is too expensive for the observed short-horizon edge. The obvious next question is whether passive entry helps.

The feature builder already records a conservative passive fill proxy:

```text
long signal: post at entry_bid
long fillable if a later best ask crosses down to entry_bid before the horizon

short signal: post at entry_ask
short fillable if a later best bid crosses up to entry_ask before the horizon

exit: taker at the horizon
```

This diagnostic quantifies fill rate, fill timing, and adverse selection for a fixed threshold rule. It still does not model queue position, partial fills, cancellations ahead, maker exit, or inventory.

## Command

```bash
PYTHONPATH=src python3 -m lob_forge.cli fill-diagnostics <feature_csv> \
  --feature microprice_deviation \
  --threshold 0.1 \
  --by-source-date \
  --maker-fee-bps 0 \
  --taker-fee-bps 0
```

The multi-day script has a dedicated mode:

```bash
bash scripts/reproduce_multiday_results.sh fills
```

Regime-bucket fill diagnostics can be reproduced with:

```bash
bash scripts/reproduce_multiday_results.sh fill-regimes
```

## BTCUSDT Result

Setup:

```text
dates: 2023-05-16 to 2023-05-19
rule: microprice_deviation > 0.1 => post passive long
      microprice_deviation < -0.1 => post passive short
fees: 0 explicit maker/taker fees
```

Aggregate result:

```text
signals: 12584
fills: 2858
fill_rate: 0.227114
gross_pnl: -8265.800000
net_pnl: -8265.800000
break_even_exit_taker_fee_bps: -1.069571
mean_net_pnl_per_fill: -2.892162
mean_net_bps_per_fill: -1.069475
win_rate: 0.074878
mean_fill_latency_ms: 2676.362491
```

By side:

```text
long fill_rate: 0.221718, net_pnl: -3923.300000, win_rate: 0.074100
short fill_rate: 0.232682, net_pnl: -4342.500000, win_rate: 0.075642
```

Every day is negative:

```text
2023-05-16 net_pnl: -3190.700000
2023-05-17 net_pnl: -1531.600000
2023-05-18 net_pnl: -1862.800000
2023-05-19 net_pnl: -1680.700000
```

## ETHUSDT Result

Aggregate result:

```text
signals: 12765
fills: 2914
fill_rate: 0.228280
gross_pnl: -534.120000
net_pnl: -534.120000
break_even_exit_taker_fee_bps: -1.012088
mean_net_pnl_per_fill: -0.183294
mean_net_bps_per_fill: -1.012167
win_rate: 0.079959
mean_fill_latency_ms: 2652.849691
```

By side:

```text
long fill_rate: 0.224362, net_pnl: -244.230000, win_rate: 0.088698
short fill_rate: 0.232017, net_pnl: -289.890000, win_rate: 0.071900
```

Every day is negative:

```text
2023-05-16 net_pnl: -181.240000
2023-05-17 net_pnl: -141.960000
2023-05-18 net_pnl: -98.670000
2023-05-19 net_pnl: -112.250000
```

## Interpretation

The passive fill proxy is adverse-selection dominated.

The problem is not that passive orders never fill. They fill about `22-23%` of the time. The problem is that they fill when price has crossed against the posted order, and forced taker exit at the horizon realizes that adverse selection. The win rate on filled passive entries is below `9%` for both BTCUSDT and ETHUSDT in this diagnostic.

This explains why the earlier `maker_entry` walk-forward selector often chose `always_flat`: the conservative fill model is not a free spread-capture mechanism.

## Regime Buckets

The `fill-regime` command bins rows by market state and reruns the same passive fill diagnostic inside each bucket:

```bash
PYTHONPATH=src python3 -m lob_forge.cli fill-regime <feature_csv> \
  --feature microprice_deviation \
  --threshold 0.1 \
  --regime-features spread_mean_5,realized_volatility_5,trade_imbalance,notional_imbalance_1pct \
  --bins 3 \
  --maker-fee-bps 0 \
  --taker-fee-bps 0
```

BTCUSDT key buckets:

```text
best all-side break_even bucket: realized_volatility_5 bucket 2
  fill_rate: 0.194923
  net_pnl: -2176.200000
  mean_net_bps_per_fill: -0.935581
  win_rate: 0.036047

highest fill-rate bucket: realized_volatility_5 bucket 3
  fill_rate: 0.306980
  net_pnl: -4026.200000
  mean_net_bps_per_fill: -1.150885
  win_rate: 0.124517
```

ETHUSDT key buckets:

```text
best all-side break_even bucket: spread_mean_5 bucket 2
  fill_rate: 0.216600
  net_pnl: -141.890000
  mean_net_bps_per_fill: -0.853974
  win_rate: 0.058952

highest fill-rate bucket: realized_volatility_5 bucket 3
  fill_rate: 0.329977
  net_pnl: -279.060000
  mean_net_bps_per_fill: -1.068377
  win_rate: 0.115811
```

Interpretation:

No tested spread, volatility, trade-flow, or aggregate-depth bucket turns the conservative passive-entry proxy positive before explicit fees. High-volatility regimes increase fill probability, but they also increase adverse selection.

## Research Consequence

The project now includes a research simulator for the missing execution mechanics in `lob_forge.execution_sim` and passive capacity estimates in `lob_forge.passive_capacity`:

- queue position and displayed size ahead,
- cancellation ahead of our order,
- partial fills,
- maker exit or inventory carry instead of forced taker exit,
- cancel/replace logic when the signal decays,
- fill probability calibrated by spread, volatility, top imbalance, and trade intensity.

This is still not a production maker model until its queue and fill-probability assumptions are calibrated against paper/live fills.

## Shadow Fill Validation Gate

The artifact-level comparison path is now executable through `edge-shadow-decisions`, `features-to-market-events`, `simulate-shadow-fills`, and `validate-shadow-fills`.

Expected simulated prediction CSV columns:

```text
decision_id,simulated_fill_price,simulated_fill_size
```

Expected shadow/paper observation CSV columns are the `ShadowDecision` logger columns:

```text
decision_id,timestamp_ms,venue,symbol,model_name,predicted_side,predicted_edge_bps,order_type,intended_price,intended_size,observed_fill_price,observed_fill_size,realized_pnl,notes
```

Export OOS decisions from the same expected-edge walk-forward protocol used for the result artifacts:

```bash
PYTHONPATH=src python3 -m lob_forge.cli create-holdout-manifest \
  data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --output artifacts/holdout_manifests/shadow_validation/btcusdt_5s_source_date_holdout.json \
  --split-column source_date \
  --holdout-values <final-date> \
  --source-root "$PWD"
PYTHONPATH=src python3 -m lob_forge.cli edge-shadow-decisions \
  data/processed/horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv \
  --holdout-manifest artifacts/holdout_manifests/shadow_validation/btcusdt_5s_source_date_holdout.json \
  --output results/shadow_validation/shadow_decisions.csv \
  --venue binance \
  --symbol BTCUSDT \
  --intended-notional 100 \
  --train-size 7200 \
  --validation-size 3600 \
  --test-size 3600 \
  --step-size 3600 \
  --taker-fee-bps 0
```

For offline dry runs, derive compatible market-event rows from the feature CSV and simulate fills:

```bash
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
```

Prepare the paper/live import template from the shadow decisions:

```bash
PYTHONPATH=src python3 -m lob_forge.cli observed-fill-template \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --output results/shadow_validation/observed_fills_template.csv \
  --limit 50
```

The helper wrapper is dry-run-first:

```bash
bash scripts/prepare_shadow_fill_validation_session.sh
DRY_RUN=0 bash scripts/prepare_shadow_fill_validation_session.sh
```

Populate `observed_fills.csv` from a real paper/live order export. Blank template rows are ignored by `import-observed-fills`; they do not count as no-fill evidence. Use explicit `cumExecQty=0` rows for real no-fill observations. Positive fill sizes require a fill price.

The local API source and normalization runbook is [paper_demo_fill_sources.md](paper_demo_fill_sources.md). The recommended order is Bybit Demo Trading first, then OKX Demo Trading, then Binance USD-M Futures Testnet if the demo-account path is blocked. Binance Spot Testnet is only for spot checks, and Alpaca Paper is only a generic simulator/API sanity check. Raw `.json`, `.jsonl`, or `.csv` provider exports can be normalized before import:

```bash
PYTHONPATH=src python3 -m lob_forge.cli normalize-observed-fills \
  --provider bybit \
  --input results/shadow_validation/raw_bybit_executions.json \
  --output results/shadow_validation/observed_fills.csv
```

Import observed fills, then run the real gate:

```bash
PYTHONPATH=src python3 -m lob_forge.cli import-observed-fills \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --observed results/shadow_validation/observed_fills.csv \
  --output results/shadow_validation/shadow_decisions_observed.csv
```

Run the real gate only after `observed_fill_price` and `observed_fill_size` are populated from shadow/paper observations:

```bash
PYTHONPATH=src python3 -m lob_forge.cli validate-shadow-fills \
  --simulated results/shadow_validation/simulated_fills.csv \
  --shadow results/shadow_validation/shadow_decisions_observed.csv \
  --max-price-error 0.5 \
  --max-size-error 0.01 \
  --max-fill-rate-error 0.05
```

The command joins by `decision_id`, reports missing simulated or shadow rows, computes mean absolute price error, mean absolute size error, and fill-rate error, and exits non-zero if any threshold fails. This is a validation gate, not evidence by itself; the real TODO remains running it on actual shadow or paper observations.
