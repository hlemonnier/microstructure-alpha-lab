# Paper And Demo Fill Sources

Date checked: 2026-06-25.

This runbook covers the local, non-GPU path for the remaining paper/live fill validation gap. The objective is not to prove profitability; it is to collect real paper/demo execution observations that can be joined back to `decision_id` and compared against the repository's simulated fill predictions.

The same source priority is also available as machine-readable CLI output:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources \
  --gap paper_live_fill_validation \
  --format markdown

PYTHONPATH=src .venv/bin/python -m lob_forge.cli free-api-sources \
  --evidence-gate real_shadow_fill_validation \
  --format json
```

## Source Priority

1. Bybit Demo Trading
   - Official docs: <https://bybit-exchange.github.io/docs/v5/demo>
   - Execution endpoint: <https://bybit-exchange.github.io/docs/v5/order/execution>
   - Why first: demo trading supports V5 order placement, order history, trade history, private WebSocket execution streams, and demo funds. The project already has Bybit L2 adapters, and `orderLinkId` is a direct fit for `decision_id`.
   - Use for: BTCUSDT/ETHUSDT crypto paper fill observations with client order IDs.
   - Import fields: `orderLinkId -> decision_id`, `execPrice -> avgPrice`, `execQty -> cumExecQty`, `symbol -> symbol`.

2. OKX Demo Trading
   - Official docs: <https://my.okx.com/docs-v5/en/>
   - Why second: OKX supports demo trading through the normal API with the `x-simulated-trading: 1` header and exposes transaction detail fields with `clOrdId`, `fillPx`, `fillSz`, `fillPnl`, and maker/taker `execType`.
   - Use for: cross-checking passive fill behavior on OKX instruments such as `BTC-USDT-SWAP`.
   - Import fields: `clOrdId -> decision_id`, `fillPx -> avgPrice`, `fillSz -> cumExecQty`, `fillPnl -> realizedPnl`, `instId -> symbol`.

3. Binance USD-M Futures Testnet
   - Official docs: <https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info>
   - Trading endpoint docs: <https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api>
   - User data stream docs: <https://developers.binance.com/docs/derivatives/usds-margined-futures/user-data-streams>
   - Why third: it is a free testnet path for USD-M futures orders and user-data order updates, so it is a better project fallback than the spot-only testnet when Bybit/OKX demo access is blocked.
   - Use for: BTCUSDT/ETHUSDT futures testnet order-event plumbing with `newClientOrderId` mapped to `decision_id`.
   - Import fields: `ORDER_TRADE_UPDATE.o.c -> decision_id`, `ORDER_TRADE_UPDATE.o.L -> avgPrice`, `ORDER_TRADE_UPDATE.o.l -> cumExecQty`, `ORDER_TRADE_UPDATE.o.rp -> realizedPnl`, `ORDER_TRADE_UPDATE.o.s -> symbol`.

4. Binance Spot Testnet
   - Official docs: <https://developers.binance.com/docs/binance-spot-api-docs/testnet/user-data-stream>
   - Trading endpoint docs: <https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/trading-endpoints>
   - Why fourth: it is free and simple for spot-only order-event capture. The `executionReport` stream carries client order IDs plus last fill price/quantity, and FULL order responses include a `fills` array.
   - Use for: spot-only sanity checks and importer regression data, not as the main venue for derivatives or deep passive queue claims.
   - Import fields: `c/clientOrderId -> decision_id`, `L/fills[].price -> avgPrice`, `l/fills[].qty -> cumExecQty`, `s/symbol -> symbol`.

5. Alpaca Paper
   - Official docs: <https://docs.alpaca.markets/us/docs/paper-trading>
   - Trade updates docs: <https://docs.alpaca.markets/us/docs/websocket-streaming>
   - Why fifth: it is free and API-friendly, but it is a broker simulator rather than a crypto exchange L2 venue. Alpaca documents that paper trading does not account for order queue position, market impact, information leakage, or latency slippage.
   - Use for: generic order lifecycle and importer checks. Treat it as weak evidence for crypto microstructure execution quality.
   - Import fields: `client_order_id -> decision_id`, `price/filled_avg_price -> avgPrice`, `qty/filled_qty -> cumExecQty`, `symbol -> symbol`.

## Local Workflow

Generate a shadow fill template first. The `client_order_id` column is intentionally set equal to `decision_id`; use that value as:

- Bybit `orderLinkId`
- OKX `clOrdId`
- Binance USD-M Futures Testnet `newClientOrderId`, visible as `ORDER_TRADE_UPDATE.o.c`
- Binance Spot Testnet `newClientOrderId`, visible as `executionReport.c`
- Alpaca `client_order_id`

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli observed-fill-template \
  --shadow results/shadow_validation/shadow_decisions.csv \
  --output results/shadow_validation/observed_fills_template.csv \
  --limit 50
```

After a paper/demo session, save the raw API response as `.json`, `.jsonl`, or `.csv`, then normalize it:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli normalize-observed-fills \
  --provider bybit \
  --input results/shadow_validation/raw_bybit_executions.json \
  --output results/shadow_validation/observed_fills.csv
```

Supported providers are `bybit`, `okx`, `binance`, and `alpaca`. The Binance normalizer accepts both Spot Testnet `executionReport`/FULL order payloads and USD-M Futures Testnet `ORDER_TRADE_UPDATE` payloads. The normalizer writes the canonical observed-fill columns:

```text
decision_id,client_order_id,venue,symbol,avgPrice,cumExecQty,realizedPnl,notes
```

Then merge observed fills into the shadow decisions and run the validation gate:

```bash
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

## Evidence Rules

- Do not count a blank template as an observation.
- Do not hand-edit fills into existence. Keep the raw provider response next to the normalized CSV.
- Positive fill size requires a fill price.
- Bybit and OKX transaction-history exports usually contain only actual fills; unfilled paper orders require an explicit terminal no-fill row if they should count as observed no-fill evidence.
- Binance USD-M Futures Testnet `ORDER_TRADE_UPDATE` rows are converted when they contain positive last-fill quantity or terminal unfilled status.
- Binance `executionReport` rows are only converted when they contain a positive last execution quantity or a terminal unfilled status.
- Alpaca is useful for API plumbing but weak for queue-position inference; do not use it as the main argument that passive crypto execution is validated.
- Coinbase Advanced Trade sandbox is not counted here because its sandbox responses are static and mocked, not real paper fills.

## Minimal Local Definition Of Done

For the second-part local gap, the repo is ready once:

- raw demo/paper exports are saved under `results/shadow_validation/`,
- the chosen source appears in `free-api-sources --evidence-gate real_shadow_fill_validation`,
- `normalize-observed-fills` produces a non-empty observed-fill CSV,
- `import-observed-fills` reports matched decisions,
- `validate-shadow-fills` passes the configured thresholds,
- `make verify-evidence-gates` no longer fails the shadow fill validation item.
