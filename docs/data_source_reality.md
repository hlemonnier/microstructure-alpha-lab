# Binance Vision Data Source Reality

Date checked: 2026-06-02.

Binance Vision exposes a public S3-style archive under:

`https://data.binance.vision/data/`

Useful top-level branches:

- `data/spot/daily`
- `data/spot/monthly`
- `data/futures/um/daily`
- `data/futures/um/monthly`
- `data/futures/cm/daily`
- `data/futures/cm/monthly`

## Critical Discovery

Spot archive data has trades, aggregate trades, and klines. It does not expose historical full L2 depth files.

USD-M futures daily archive has `bookDepth`, `bookTicker`, `trades`, `aggTrades`, and several contextual kline/metric datasets.

Sample `bookDepth` row:

```csv
timestamp,percentage,depth,notional
2023-01-01 00:06:05,-5,16770.30000000,271987123.35882000
```

This is aggregate depth by percentage band, not a tensor of price levels. It cannot support a strict DeepLOB replication by itself.

## Implication

The best v1 project is:

Quote/trade/depth-band microstructure modeling on Binance USD-M futures using Binance Vision archives.

The DeepLOB-style extension remains valid, but it requires one of:

- live collection of Binance diff-depth streams plus snapshots,
- a paid or third-party historical L2 archive,
- or a public benchmark dataset such as FI-2010 for the deep LOB component.

## V1 Data Roles

`bookTicker`

- Best bid/ask price and size.
- Source for mid-price, spread, top-of-book imbalance, quote update intensity, and labels.
- Large files, often tens to hundreds of MB per day zipped.

`trades` / `aggTrades`

- Trade price, quantity, timestamp, and side proxy.
- Source for signed flow, trade imbalance, aggressor pressure, large trade indicators, and realized volume.

`bookDepth`

- Aggregate liquidity bands around price.
- Source for slow liquidity regime features: plus/minus 1 percent depth, 5 percent depth, notional depth asymmetry, and liquidity shocks.
- The current parser treats negative `percentage` bands as bid-side depth and positive bands as ask-side depth.
- Small enough for quick downloader testing.

## V1 Success Criteria

The first credible result is not a profitable strategy. It is evidence that the data pipeline can:

- download deterministic files by dataset, symbol, and date,
- verify checksums,
- parse schemas correctly,
- align quote and trade time series without leakage,
- produce labels only from future quote states,
- and evaluate signals under costs and latency.
