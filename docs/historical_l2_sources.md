# Historical L2 Source Priority

Date checked: 2026-06-03.

This project now treats Binance Vision `bookTicker` / `aggTrades` / `bookDepth` as the v1 quote-trade-depth-band lab, and treats true L2 as a separate ingestion lane. The priority below is based on practical ability to obtain snapshots plus deltas, validate schema, replay sequence state, and run fill/queue research.

## Priority Order

0. Binance Vision control source
   - Source: <https://data.binance.vision/>
   - Supporting schema/documentation: <https://github.com/binance/binance-public-data>
   - Why it matters: this is the existing v1 data source for deterministic quote/trade/depth-band experiments.
   - Project adapter: use the existing Binance Vision downloader and feature builders, not the full-L2 replay adapters.
   - Caveat: `bookDepth` is aggregate percentage-band depth, not a historical price-level L2 delta stream.

1. OKX historical order book
   - Source: <https://www.okx.com/en-gb/historical-data>
   - Why it matters: OKX publishes an exchange historical-data page that lists high-resolution L2 order-book data from March 2023 onward.
   - Project adapter: `iter_okx_l2_csv` accepts timestamp, snapshot/update type, side, price, size, and sequence/update aliases.
   - Caveat: archive schemas can drift; every file must pass `validate_l2_csv_schema(..., source_format="okx")` before replay.

2. Bybit historical orderBook
   - Source: <https://www.bybit.com/derivatives/en/history-data>
   - Supporting schema: <https://bybit-exchange.github.io/docs/v5/market/orderbook>
   - Why it matters: Bybit exposes public history-data pages and documents order-book snapshot fields including update ID and cross sequence.
   - Project adapter: `iter_bybit_l2_csv` accepts Bybit-style `u` and `seq` metadata when present.
   - Caveat: regional pages may redirect, so checked-in experiments should record the exact downloaded URL and file hash.

3. Tardis.dev normalized CSV
   - Source: <https://docs.tardis.dev/faq/data>
   - Why it matters: Tardis documents normalized downloadable CSV files for incremental order book L2 updates and order-book snapshots.
   - Project adapter: `iter_tardis_incremental_book_l2_csv` expects `local_timestamp`, `timestamp`, `symbol`, `is_snapshot`, `side`, `price`, and `amount`.
   - Caveat: broad historical coverage is paid, but monthly first-day samples are useful for parser and replay tests.

4. Crypto Lake
   - Source: <https://crypto-lake.com/data/>
   - Why it matters: Crypto Lake exposes book snapshots, `book_delta_v2`, and deeper 1-minute book snapshots through its API/data products.
   - Project adapter: `iter_crypto_lake_book_delta_v2_csv` handles origin/received time aliases, side, price, and amount/quantity fields.
   - Caveat: exact dataframe schema and package version must be pinned in experiment metadata.

5. Coinbase Advanced `level2`
   - Source: <https://coinbase-cloud.mintlify.app/coinbase-app/advanced-trade-apis/websocket/websocket-overview>
   - Why it matters: Coinbase documents a real-time `level2` WebSocket channel.
   - Project status: use for live capture, not as a ready historical archive.
   - Caveat: historical replay requires prior capture or a vendor archive.

6. FI-2010 benchmark
   - Source: <https://arxiv.org/abs/1705.03233>
   - Why it matters: FI-2010 is a public academic benchmark for mid-price forecasting on normalized LOB matrices.
   - Project adapter: `load_fi2010_snapshots` projects the common 40-feature, 10-level matrix into top-N snapshot events.
   - Caveat: it is normalized equities data, not crypto execution evidence.

## Validation Contract

Every true-L2 file must provide or be mapped to:

- snapshot/delta flag,
- event timestamp,
- side,
- price,
- size,
- sequence or update identifiers when the source exposes them.

`validate_l2_csv_schema` enforces the required fields and checks the first rows for parseable timestamps, valid side, nonnegative size, and positive price. Files that do not pass validation should not enter replay, tensor extraction, passive fill simulation, or DeepLOB experiments.

## Storage Contract

Normalized events use `NormalizedL2Row`:

```text
event_type, exchange_timestamp, side, price, size,
local_timestamp, sequence, update_id, venue, symbol
```

The dependency-free path writes normalized CSV through `write_normalized_l2_csv`. The research path writes Parquet through `write_normalized_l2_parquet` when `pyarrow` is installed via `.[research]`.

## Acquisition And Import Workflow

Historical OKX/Bybit acquisition is manifest-driven because exchange pages can redirect and file URLs can change. The project records the source page, exact direct URL or local file path, file hash, validation result, and normalized output path instead of treating discovery as implicit evidence.

Create an acquisition manifest:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-manifest \
  --source okx \
  --symbols BTC-USDT-SWAP,ETH-USDT-SWAP \
  --start 2023-05-16 \
  --end 2023-07-14 \
  --output data/manifests/okx_l2_20230516_20230714.csv
```

For OKX, first try resolving each row through the same public download-link endpoint used by OKX's historical-data page. This endpoint is rate-limited and can legitimately return no files for a requested symbol/date/depth, so unresolved rows stay in the manifest with `no_url_found` or `resolve_failed` status:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-resolve-okx \
  data/manifests/okx_l2_20230516_20230714.csv \
  --depth 400 \
  --throttle-seconds 2
```

If the resolver does not return a URL, fill that row's `direct_url` manually with the exact exchange file URL or a local path to an already-downloaded file. Download/copy resolved files into the raw store:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-download-manifest \
  data/manifests/okx_l2_20230516_20230714.csv \
  --raw-root data/raw
```

Validate one file before importing a large batch:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-validate \
  data/raw/historical_l2/okx/BTC-USDT-SWAP/2023-05-16/example.csv \
  --source okx
```

Normalize all downloaded manifest entries into partitioned storage:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-import-manifest \
  data/manifests/okx_l2_20230516_20230714.csv \
  --output-root data \
  --format csv
```

For Bybit, use `--source bybit` and the instrument symbol used by the downloaded file. The resolver calls Bybit's public history-data `list-files` endpoint and records the returned `quote-saver.bycsi.com` orderBook file URL when available:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-manifest \
  --source bybit \
  --symbols BTCUSDT,ETHUSDT \
  --start 2023-05-16 \
  --end 2023-07-14 \
  --output data/manifests/bybit_l2_20230516_20230714.csv

PYTHONPATH=src python3 -m lob_forge.cli l2-resolve-bybit \
  data/manifests/bybit_l2_20230516_20230714.csv \
  --throttle-seconds 2
```

Bybit orderBook files are `.data.zip` archives containing newline-delimited JSON messages shaped like the V5 orderbook stream. `l2-validate`, `l2-import`, and `l2-import-manifest` now normalize those archives directly into the project L2 schema. Use `--require-sequence` when promoting a file into deterministic replay experiments; files without sequence/update identifiers can still be used for rough tensor or feature research, but not for deterministic replay claims.

For a 16 GB laptop smoke test, cap normalization instead of expanding a whole day:

```bash
PYTHONPATH=src python3 -m lob_forge.cli l2-import-manifest \
  data/manifests/bybit_l2_20230516_20230714.csv \
  --output-root data \
  --format csv \
  --max-import-rows 5000 \
  --require-sequence
```

The cap limits rows written per manifest entry without splitting one exchange update in half. It may therefore stop below the requested row count. It is not a research artifact; it is a bounded parser/readiness smoke path.

The packaged smoke runner wraps the same flow:

```bash
bash scripts/run_bybit_l2_smoke.sh
```

By default it resolves one BTCUSDT day, downloads the archive, imports 5,000 normalized rows, and then runs `model-readiness-gate` in reporting mode.
