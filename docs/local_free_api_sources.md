# Local Free/Freemium API Source Plan

Date checked: 2026-06-25.

This page covers the second-part data gap: sources that can be used locally on an M1 Pro with 16 GB RAM. It does not claim that the full 60-90 day confirmatory matrix is laptop work. That remains a high-RAM/cloud run. The local objective is to collect or normalize the data needed to unblock:

- real demo/paper observed fills for `paper_live_fill_validation`;
- row-capped true-L2 smoke imports for replay/schema/model-gate evidence;
- public quote/trade/depth-band archives for bounded classical experiments.

The machine-readable catalog is exposed by:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --format markdown
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --gap paper_live_fill_validation --format json
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources --gap true_l2_laptop_smoke --format csv
```

## Priority

| Rank | Gap | Source | Why |
| --- | --- | --- | --- |
| 1 | paper/live fills | Bybit Demo Trading | Closest fit for BTCUSDT/ETHUSDT crypto demo fills; `orderLinkId` maps directly to `decision_id`; trade history returns `execPrice`, `execQty`, `execTime`, `isMaker`, and `execType`. |
| 2 | paper/live fills | OKX Demo Trading | Good cross-venue check; demo mode uses the normal API surface with simulated-trading credentials/header; transaction details expose `clOrdId`, `fillPx`, `fillSz`, and `fillPnl`. |
| 3 | paper/live fills | Binance Spot Testnet | Useful free spot-only order event plumbing through `executionReport` and FULL order `fills`; not a derivatives queue source. |
| 4 | paper/live fills | Alpaca Paper | Useful importer/order-lifecycle fallback, but weak evidence for crypto queue behavior because paper fills omit queue position, market impact, information leakage, and latency slippage. |
| 5 | true-L2 smoke | OKX Historical Market Data | Free public downloads include high-resolution L2 order book data from March 2023 onward. Use row caps locally. |
| 6 | true-L2 smoke | Bybit historical orderBook | Public history-data orderBook archives plus V5 order-book fields `u`, `seq`, and `cts` make it useful for replay validation. |
| 7 | true-L2 smoke | Tardis.dev CSV samples | Freemium first-day-of-month CSV samples need no API key and are useful as a cross-vendor schema sanity corpus. |
| 8 | quote/trade/depth-band | Binance Public Data Archives | Free daily/monthly public archives are still the best local classical feature source, but `bookDepth` is aggregate percentage-band depth, not full L2. |

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
Binance newClientOrderId
Alpaca: client_order_id
```

4. Save the raw provider response under `results/shadow_validation/`.
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

The laptop definition of done is not "download everything." It is:

- at least one row-capped OKX or Bybit normalized true-L2 file;
- `l2-validate`/`l2-import` passes with sequence metadata when available;
- `make verify-evidence-gates` keeps true-L2/model-readiness gates green;
- the full 60-90 day cloud gate stays honestly red until the high-RAM job finishes.

## Current Call

Use Bybit Demo Trading first for the observed-fill gap, then OKX Demo Trading as a cross-venue check. Use OKX or Bybit public historical L2 for local smoke imports. Use Tardis.dev only as a freemium schema sanity corpus unless a paid API key is available. Use Binance Public Data for classical quote/trade/depth-band experiments, not for full L2 replay claims.
