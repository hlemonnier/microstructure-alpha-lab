# Local Free/Freemium API Source Plan

Date checked: 2026-06-26.

This page covers the second-part data gap: sources that can be used locally on an M1 Pro with 16 GB RAM. It does not claim that the full 60-90 day confirmatory matrix is laptop work. That remains a high-RAM/cloud run. The local objective is to collect or normalize the data needed to unblock:

- real demo/paper observed fills for `paper_live_fill_validation`;
- row-capped true-L2 smoke imports for replay/schema/model-gate evidence;
- public quote/trade/depth-band archives for bounded classical experiments.

The machine-readable catalog is exposed by:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --format markdown
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --gap paper_live_fill_validation --format json
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --evidence-gate real_shadow_fill_validation --format json
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --evidence-gate sequence_transformer_tcn_experiments --format csv
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --gap true_l2_laptop_smoke --format csv
```

## Priority

| Rank | Gap | Source | Why |
| --- | --- | --- | --- |
| 1 | paper/live fills | Bybit Demo Trading | Closest fit for BTCUSDT/ETHUSDT crypto demo fills; `orderLinkId` maps directly to `decision_id`; trade history returns `execPrice`, `execQty`, `execTime`, `isMaker`, and `execType`. |
| 2 | paper/live fills | OKX Demo Trading | Good cross-venue check; demo mode uses the normal API surface with demo credentials plus `x-simulated-trading: 1`; transaction details expose `clOrdId`, `fillPx`, `fillSz`, and `fillPnl`. |
| 3 | paper/live fills | Binance USD-M Futures Testnet | Better fallback than Spot Testnet for this project; testnet REST/WebSocket endpoints support USD-M futures orders and `ORDER_TRADE_UPDATE` user-data events that can be normalized with the Binance importer. |
| 4 | paper/live fills | Binance Spot Testnet | Useful free spot-only order event plumbing through `executionReport` and FULL order `fills`; not a derivatives queue source. |
| 5 | paper/live fills | Alpaca Paper | Useful importer/order-lifecycle fallback, but weak evidence for crypto queue behavior because paper fills omit queue position, market impact, information leakage, and latency slippage. |
| 6 | true-L2 smoke | OKX Historical Market Data | Free public downloads include high-resolution L2 order book data from March 2023 onward. Use row caps locally. |
| 7 | true-L2 smoke | Bybit historical orderBook | Public history-data orderBook archives plus V5 order-book fields `u`, `seq`, and `cts` make it useful for replay validation. |
| 8 | true-L2 smoke | Coinbase public level2 WebSocket | No-key live L2 capture path already supported by `live-l2-capture`; useful when public historical links are missing. |
| 9 | true-L2 smoke | Tardis.dev CSV samples | Freemium first-day-of-month CSV samples need no API key and are useful as a cross-vendor schema sanity corpus. |
| 10 | true-L2 smoke | Crypto Lake free samples | Extra 20-level book/trade sample corpus for parser and tensor sanity checks; inspect coverage before using it as study evidence. |
| 11 | quote/trade/depth-band | Binance Public Data Archives | Free daily/monthly public archives are still the best local classical feature source, but `bookDepth` is aggregate percentage-depth bands, not full L2. |

## Gate Mapping

| Evidence gate | Local source class | Minimum artifact |
| --- | --- | --- |
| `real_shadow_fill_validation` / `paper_live_fill_validation` | Bybit or OKX demo fills first, Binance/Alpaca fallback | raw provider export plus non-empty `results/shadow_validation/observed_fills.csv` merged into shadow decisions |
| `sequence_transformer_tcn_experiments` | OKX/Bybit historical L2, Coinbase live L2, Tardis/Crypto Lake samples | row-capped normalized L2 CSV that passes `l2-validate` and model-readiness checks |
| `self_supervised_l2_pretraining` | same true-L2 smoke sources | normalized L2 CSV plus pretraining smoke artifact whose `l2_path` matches the checked file |
| `capped_60day_btc_eth` | Binance public archives | local16 study artifacts under `results/expected_edge_local16_20230516_20230714/` |
| `kelly_variance_stability` | Binance public archives | audited Kelly candidate artifacts under `results/kelly_candidate_search/` |

## Local Fill Acquisition Loop

1. Generate shadow decisions and simulated fills from an OOS stream.
2. Create the observed-fill template:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli observed-fill-template \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --output results/shadow_validation/observed_fills_template.csv \
  --limit 50
```

3. Place demo/paper orders using `decision_id` as the provider client ID:

```text
Bybit:  orderLinkId
OKX:    clOrdId
Binance USD-M Futures Testnet: newClientOrderId, returned as ORDER_TRADE_UPDATE.o.c
Binance Spot Testnet: newClientOrderId / executionReport.c
Alpaca: client_order_id
```

For OKX demo, send requests to the documented OKX REST/WebSocket demo URLs and include `x-simulated-trading: 1` on REST requests.

4. Save the raw provider response under `results/shadow_validation/`.
   The Bybit and OKX demo REST pulls are executable:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli fetch-observed-fills \
  --provider bybit \
  --symbol BTCUSDT \
  --start-time-ms 1700000000000 \
  --end-time-ms 1700000600000 \
  --output results/shadow_validation/raw_bybit_executions.json

PYTHONPATH=src .venv/bin/python -m lob_forge.cli fetch-observed-fills \
  --provider okx \
  --symbol BTC-USDT-SWAP \
  --start-time-ms 1700000000000 \
  --end-time-ms 1700000600000 \
  --output results/shadow_validation/raw_okx_fills.json
```

5. Normalize, import, and validate:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli normalize-observed-fills \
  --provider bybit \
  --input results/shadow_validation/raw_bybit_executions.json \
  --output results/shadow_validation/observed_fills.csv

PYTHONPATH=src .venv/bin/python -m lob_forge.cli import-observed-fills \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --observed results/shadow_validation/observed_fills.csv \
  --output results/shadow_validation/shadow_decisions_observed.csv

PYTHONPATH=src .venv/bin/python -m lob_forge.cli validate-shadow-fills \
  --simulated results/shadow_validation/simulated_fills.csv \
  --shadow results/shadow_validation/shadow_decisions_observed.csv \
  --max-price-error 0.5 \
  --max-size-error 0.01 \
  --max-fill-rate-error 0.05
```

## Local True-L2 Smoke Loop

For OKX:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-manifest \
  --source okx \
  --symbols BTC-USDT-SWAP \
  --start 2023-05-16 \
  --end 2023-05-16 \
  --output artifacts/l2/okx_btc_manifest.csv

PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-resolve-okx artifacts/l2/okx_btc_manifest.csv
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-download-manifest artifacts/l2/okx_btc_manifest.csv
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-import-manifest \
  artifacts/l2/okx_btc_manifest.csv \
  --max-import-rows 50000 \
  --require-sequence
```

For Bybit:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-manifest \
  --source bybit \
  --symbols BTCUSDT \
  --start 2023-05-16 \
  --end 2023-05-16 \
  --output artifacts/l2/bybit_btc_manifest.csv

PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-resolve-bybit artifacts/l2/bybit_btc_manifest.csv
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-download-manifest artifacts/l2/bybit_btc_manifest.csv
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-import-manifest \
  artifacts/l2/bybit_btc_manifest.csv \
  --max-import-rows 50000 \
  --require-sequence
```

For Coinbase live L2:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli live-l2-capture \
  --venue coinbase \
  --symbol BTC-USD \
  --output data/live_l2/coinbase/BTC-USD/session.csv \
  --seconds 60 \
  --max-messages 1000 \
  --reset-on-gap

PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-validate \
  data/live_l2/coinbase/BTC-USD/session.csv \
  --source coinbase
```

The laptop definition of done is not "download everything." It is:

- at least one row-capped OKX or Bybit normalized true-L2 file;
- `l2-validate`/`l2-import` passes with sequence metadata when available;
- `make verify-evidence-gates` keeps true-L2/model-readiness gates green;
- the full 60-90 day cloud gate stays honestly red until the high-RAM job finishes.

## Low-Signal Or Rejected Sources

- Coinbase Advanced Trade sandbox is useful for static response-shape tests only. Its official sandbox responses are mocked and pre-defined, so it cannot satisfy real shadow/paper fill validation.
- Generic broker paper trading is lower priority than Bybit, OKX, or Binance futures testnet because this project needs crypto exchange-style maker/taker and client-order-id observations.

## Current Call

Use Bybit Demo Trading first for the observed-fill gap, then OKX Demo Trading as a cross-venue check. If either account path is blocked, use Binance USD-M Futures Testnet before Spot Testnet because it matches the BTCUSDT/ETHUSDT futures research surface more closely. Use OKX or Bybit public historical L2 first for local smoke imports, Coinbase live level2 when historical download discovery fails, and Tardis.dev or Crypto Lake as sample sanity corpora. Use Binance Public Data for classical quote/trade/depth-band experiments, not for full L2 replay claims.
