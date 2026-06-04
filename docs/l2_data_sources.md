# L2 Data Source Map

This document is the implementation companion to `src/lob_forge/data_sources.py`.
It separates replay-grade inputs from useful but weaker sample or feature sources.

## Priority

1. OKX is the primary free historical L2 route. Its historical data page lists high-resolution L2 order book data from March 2023 onward:
   https://www.okx.com/en-gb/historical-data
2. Bybit is the second venue. The official V5 orderbook API documents snapshot fields, update ID `u`, cross sequence `seq`, and matching-engine timestamp `cts`:
   https://bybit-exchange.github.io/docs/v5/market/orderbook
3. Binance Data Vision is useful for public bulk market data, but deterministic replay should come from live REST snapshot plus diff-depth WebSocket stitching:
   https://github.com/binance/binance-public-data
   https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md
4. Coinbase is live-only for this project. The level2 WebSocket path is useful for collection, and sequence handling must be monitored:
   https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-overview
5. Tardis.dev and Crypto Lake are sample/validation sources. Tardis documents `incremental_book_L2`, `book_snapshot_25`, and first-day-of-month no-key samples:
   https://docs.tardis.dev/downloadable-csv-files/overview
   https://crypto-lake.com/free-data/
6. FI-2010 is an equity LOB benchmark for DeepLOB sanity checks, not crypto evidence:
   https://arxiv.org/abs/1705.03233

## Replay Gate

A file can be used for deterministic replay only if the schema validation passes and the rows expose:

- snapshot/delta semantics
- exchange timestamp or local receive timestamp
- side
- price
- size
- sequence or update ID for replay-grade sources

If sequence/update IDs are missing, the file can still be used for fixed-interval top-N tensor training or rough feature research, but the project must not call it deterministic L2 replay.

## Implemented Code

- `SourceDescriptor` ranks and documents OKX, Bybit, Binance, Coinbase, Tardis.dev, Crypto Lake, and FI-2010.
- `validate_l2_columns`, `validate_l2_row`, and `validate_l2_csv` enforce the replay gate with alias handling.
- `normalize_l2_row` converts CSV-like rows into a normalized event row.
- `l2-manifest`, `l2-resolve-okx`, `l2-resolve-bybit`, `l2-download-manifest`, `l2-validate`, `l2-import`, and `l2-import-manifest` provide the executable OKX/Bybit historical L2 acquisition and normalization workflow. The import commands support `--max-import-rows` for bounded laptop smoke tests.
- `iter_bybit_orderbook_data_zip` reads Bybit `.data.zip` orderBook archives and expands snapshot/delta JSON messages into normalized per-level L2 rows with `seq`, `u`, and `cts` preserved.
- `normalized_l2_parquet_path` and `write_normalized_l2_parquet` provide the Parquet storage hook when `pyarrow` is installed.
- `load_fi2010_matrix` and `load_fi2010_named_columns` load benchmark tensors for model sanity checks.
- `normalize_live_l2_message` converts Binance, OKX, Bybit, and Coinbase live snapshot/delta payloads into normalized L2 rows.
- `live-l2-capture` can fetch the configured REST snapshot when available, connect to the configured WebSocket, persist normalized rows to CSV, and fail fast if the sequence tracker detects a gap. With `--reset-on-gap`, it refetches the REST snapshot where available or resubscribes/resets state where that is the venue's public option.

Example live capture:

```bash
PYTHONPATH=src python3 -m lob_forge.cli live-l2-capture \
  --venue binance \
  --symbol BTCUSDT \
  --output data/live_l2/binance/BTCUSDT/session.csv \
  --seconds 60 \
  --depth 100 \
  --reset-on-gap
```
