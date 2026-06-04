#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-results}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

START_DATE="${START_DATE:-2023-05-16}"
END_DATE="${END_DATE:-2023-05-19}"
MAX_QUOTE_BUCKETS="${MAX_QUOTE_BUCKETS:-3600}"
START_SLUG="${START_DATE//-/}"
END_SLUG="${END_DATE//-/}"

BTC_DIR="data/processed/multiday_${START_SLUG}_${END_SLUG}_btc_5s_latency_1000"
ETH_DIR="data/processed/multiday_${START_SLUG}_${END_SLUG}_eth_5s_latency_1000"
BTC_COMBINED="$BTC_DIR/BTCUSDT-${START_DATE}_${END_DATE}-combined-features.csv"
ETH_COMBINED="$ETH_DIR/ETHUSDT-${START_DATE}_${END_DATE}-combined-features.csv"

run_builds() {
  python3 -m lob_forge.cli build-range \
    --symbol BTCUSDT \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --output-dir "$BTC_DIR" \
    --combined-output "$BTC_COMBINED" \
    --bucket-ms 1000 \
    --horizon-ms 5000 \
    --execution-latency-ms 1000 \
    --threshold half_spread \
    --min-tick 0.1 \
    --max-quote-buckets "$MAX_QUOTE_BUCKETS" \
    --with-book-depth

  python3 -m lob_forge.cli build-range \
    --symbol ETHUSDT \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --output-dir "$ETH_DIR" \
    --combined-output "$ETH_COMBINED" \
    --bucket-ms 1000 \
    --horizon-ms 5000 \
    --execution-latency-ms 1000 \
    --threshold half_spread \
    --min-tick 0.01 \
    --max-quote-buckets "$MAX_QUOTE_BUCKETS" \
    --with-book-depth
}

run_results_for_symbol() {
  local symbol="$1"
  local path="$2"

  echo "== ${symbol} multiday feature summary =="
  python3 -m lob_forge.cli describe-features "$path"

  echo
  echo "== ${symbol} multiday threshold walk-forward, zero fees =="
  python3 -m lob_forge.cli walk-forward "$path" \
    --train-size 3600 \
    --validation-size 1800 \
    --test-size 1800 \
    --step-size 1800 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ${symbol} multiday threshold walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli walk-forward "$path" \
    --train-size 3600 \
    --validation-size 1800 \
    --test-size 1800 \
    --step-size 1800 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl

  echo
  echo "== ${symbol} multiday expected-edge walk-forward, zero fees =="
  python3 -m lob_forge.cli edge-walk-forward "$path" \
    --train-size 3600 \
    --validation-size 1800 \
    --test-size 1800 \
    --step-size 1800 \
    --l2 1 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ${symbol} multiday expected-edge walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli edge-walk-forward "$path" \
    --train-size 3600 \
    --validation-size 1800 \
    --test-size 1800 \
    --step-size 1800 \
    --l2 1 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl
}

run_results() {
  run_results_for_symbol BTCUSDT "$BTC_COMBINED"
  echo
  run_results_for_symbol ETHUSDT "$ETH_COMBINED"
}

run_fills() {
  echo "== BTCUSDT multiday passive fill diagnostics =="
  python3 -m lob_forge.cli fill-diagnostics "$BTC_COMBINED" \
    --feature microprice_deviation \
    --threshold 0.1 \
    --by-source-date \
    --maker-fee-bps 0 \
    --taker-fee-bps 0

  echo
  echo "== ETHUSDT multiday passive fill diagnostics =="
  python3 -m lob_forge.cli fill-diagnostics "$ETH_COMBINED" \
    --feature microprice_deviation \
    --threshold 0.1 \
    --by-source-date \
    --maker-fee-bps 0 \
    --taker-fee-bps 0
}

run_fill_regimes() {
  echo "== BTCUSDT multiday passive fill regime diagnostics =="
  python3 -m lob_forge.cli fill-regime "$BTC_COMBINED" \
    --feature microprice_deviation \
    --threshold 0.1 \
    --regime-features spread_mean_5,realized_volatility_5,trade_imbalance,notional_imbalance_1pct \
    --bins 3 \
    --maker-fee-bps 0 \
    --taker-fee-bps 0

  echo
  echo "== ETHUSDT multiday passive fill regime diagnostics =="
  python3 -m lob_forge.cli fill-regime "$ETH_COMBINED" \
    --feature microprice_deviation \
    --threshold 0.1 \
    --regime-features spread_mean_5,realized_volatility_5,trade_imbalance,notional_imbalance_1pct \
    --bins 3 \
    --maker-fee-bps 0 \
    --taker-fee-bps 0
}

case "$MODE" in
  build)
    run_builds
    ;;
  results)
    run_results
    ;;
  fills)
    run_fills
    ;;
  fill-regimes)
    run_fill_regimes
    ;;
  all)
    run_builds
    run_results
    run_fills
    run_fill_regimes
    ;;
  *)
    echo "Usage: bash scripts/reproduce_multiday_results.sh [build|results|fills|fill-regimes|all]" >&2
    exit 2
    ;;
esac
