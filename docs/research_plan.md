# Research Plan

## Research Question

Can Binance Vision USD-M futures quote, trade, and depth-band archives produce a short-horizon microstructure signal that survives realistic transaction costs, latency assumptions, and walk-forward validation?

## Instruments

V1:

- `BTCUSDT`
- `ETHUSDT`

Cross-asset validation later:

- `SOLUSDT`
- `BNBUSDT`
- `XRPUSDT`

## Data

Primary datasets:

- `bookTicker` for best bid/ask, spread, mid, top-of-book quantity imbalance, and labels.
- `trades` or `aggTrades` for signed trade flow and intensity.
- `bookDepth` for aggregate liquidity bands and regime features.

The project deliberately starts with USD-M futures because the archive contains richer quote/depth data than Spot.

## Core Variables

At quote time `t`:

```text
bid_t = best bid price
ask_t = best ask price
bid_qty_t = best bid quantity
ask_qty_t = best ask quantity
mid_t = (bid_t + ask_t) / 2
spread_t = ask_t - bid_t
relative_spread_t = spread_t / mid_t
top_imbalance_t = (bid_qty_t - ask_qty_t) / (bid_qty_t + ask_qty_t)
microprice_t = (ask_t * bid_qty_t + bid_t * ask_qty_t) / (bid_qty_t + ask_qty_t)
microprice_deviation_t = (microprice_t - mid_t) / spread_t
```

Quote order-flow features between bucket `t-1` and `t`:

```text
ofi_t =
  1[bid_t >= bid_{t-1}] * bid_qty_t
  - 1[bid_t <= bid_{t-1}] * bid_qty_{t-1}
  - 1[ask_t <= ask_{t-1}] * ask_qty_t
  + 1[ask_t >= ask_{t-1}] * ask_qty_{t-1}

quote_ofi_normalized_t = ofi_t / (bid_qty_t + ask_qty_t)
quote_ofi_5_normalized_t = sum(ofi_{t-4:t}) / (bid_qty_t + ask_qty_t)
```

Rolling context features are causal and use current-or-earlier quote buckets only:

```text
mid_return_1
mid_return_5
realized_volatility_5
spread_mean_5
top_imbalance_mean_5
```

Trade-flow features over window `W`:

```text
buy_volume_W
sell_volume_W
trade_imbalance_W = (buy_volume_W - sell_volume_W) / (buy_volume_W + sell_volume_W)
trade_count_W
large_trade_count_W
realized_volatility_W
```

Depth-band features from `bookDepth`:

```text
depth_bid_1pct, depth_ask_1pct
depth_bid_5pct, depth_ask_5pct
notional_bid_1pct, notional_ask_1pct
band_depth_imbalance_k = (depth_bid_k - depth_ask_k) / (depth_bid_k + depth_ask_k)
```

The implemented parser treats negative `percentage` rows as bid-side aggregate depth and positive rows as ask-side aggregate depth. These features are slow liquidity-regime context, not full level-by-level order book tensors.

## Labels

Use future mid-price movement from `bookTicker`, not trade price.

For horizon `h`:

```text
entry_mid_t = mid_{t+latency}
future_mid_t_h = mid_{t+latency+h}
delta_mid_t_h = future_mid_t_h - entry_mid_t
```

Three-class label:

```text
y_t = +1 if delta_mid_t_h > theta_t
y_t = 0 if abs(delta_mid_t_h) <= theta_t
y_t = -1 if delta_mid_t_h < -theta_t
```

Candidate thresholds:

```text
theta_t = 0.5 * spread_t
theta_t = 1 tick
theta_t = c * rolling_std(delta_mid)
```

The default should be spread-aware. Tiny moves that cannot pay the spread and fees are not useful alpha.

## Horizons

Clock-time horizons:

- 100 ms
- 500 ms
- 1 s
- 5 s
- 10 s

Event-time horizons:

- 10 quote updates
- 50 quote updates
- 100 quote updates

Both matter. Event time is often cleaner for microstructure; clock time is closer to execution and latency evaluation.

## Baselines

Start with classical baselines:

- majority/no-trade baseline,
- softmax logistic regression with train-window standardization and validation-selected alpha thresholds,
- ridge classifier,
- gradient boosting,
- simple threshold rules on imbalance and trade flow.

Deep models only make sense after these baselines:

- MLP/TCN on top-of-book sequences,
- compact transformer over quote/trade event sequences,
- DeepLOB only after true L2 data is collected.

## Backtest Logic

Prediction output:

```text
p_up, p_flat, p_down
alpha_t = p_up - p_down
```

Taker rule:

```text
long if alpha_t > tau
short if alpha_t < -tau
flat otherwise
```

Long entry/exit:

```text
entry_t = ask_{t + latency}
exit_t_h = bid_{t + latency + h}
pnl_gross = exit_t_h - entry_t
pnl_net = pnl_gross - fee_entry * entry_t - fee_exit * exit_t_h - slippage
```

Short entry/exit:

```text
entry_t = bid_{t + latency}
exit_t_h = ask_{t + latency + h}
pnl_gross = entry_t - exit_t_h
pnl_net = pnl_gross - fee_entry * entry_t - fee_exit * exit_t_h - slippage
```

Conservative maker-entry model:

```text
long passive entry price = bid_{t + latency}
long fillable only if min ask over (t + latency, t + latency + h] <= entry_bid
long exit = bid_{t + latency + h}

short passive entry price = ask_{t + latency}
short fillable only if max bid over (t + latency, t + latency + h] >= entry_ask
short exit = ask_{t + latency + h}
```

This is deliberately conservative and does not model queue position, partial fills, cancellations ahead, or order priority. It is a filter for whether passive execution is worth deeper modeling, not a production passive backtest.

Break-even intuition:

```text
break_even_move ~= spread + 2 * fee_rate * mid + slippage
```

## Validation

Use purged walk-forward validation:

- train on earlier days,
- validate on later contiguous days,
- test on future days,
- purge samples around fold boundaries when label horizons overlap.

Never use random splits.

Report:

- balanced accuracy and macro F1,
- calibration,
- precision/recall by class,
- trade selectivity,
- turnover,
- net PnL per trade,
- drawdown,
- sensitivity to fees,
- sensitivity to latency,
- performance by spread and volatility regime.

## Kill Criteria

Stop or pivot if:

- data alignment cannot be made deterministic,
- quote/trade timestamps cannot support the intended horizons,
- apparent accuracy disappears under purged walk-forward validation,
- all edge is killed by one-spread-plus-fees assumptions,
- model performance is dominated by a simple imbalance threshold.
