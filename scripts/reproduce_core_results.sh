#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-results}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

BTC_5S="data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv"
ETH_5S="data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv"

run_builds() {
  python3 -m lob_forge.cli build-range \
    --symbol BTCUSDT \
    --start 2023-05-16 \
    --end 2023-05-17 \
    --output-dir data/processed/maker_horizon_5000_latency_1000 \
    --combined-output "$BTC_5S" \
    --bucket-ms 1000 \
    --horizon-ms 5000 \
    --execution-latency-ms 1000 \
    --threshold half_spread \
    --min-tick 0.1 \
    --max-quote-buckets 3600 \
    --with-book-depth

  python3 -m lob_forge.cli build-range \
    --symbol ETHUSDT \
    --start 2023-05-16 \
    --end 2023-05-17 \
    --output-dir data/processed/eth_maker_horizon_5000_latency_1000 \
    --combined-output "$ETH_5S" \
    --bucket-ms 1000 \
    --horizon-ms 5000 \
    --execution-latency-ms 1000 \
    --threshold half_spread \
    --min-tick 0.01 \
    --max-quote-buckets 3600 \
    --with-book-depth
}

run_results() {
  echo "== BTCUSDT 5s threshold walk-forward, zero fees =="
  python3 -m lob_forge.cli walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s threshold walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s fixed-rule regime diagnostics, zero fees =="
  python3 -m lob_forge.cli regime "$BTC_5S" \
    --feature microprice_deviation \
    --threshold 0.05 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --bins 3 \
    --taker-fee-bps 0

  echo
  echo "== BTCUSDT 5s logistic walk-forward, zero fees =="
  python3 -m lob_forge.cli logistic-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --epochs 120 \
    --learning-rate 0.05 \
    --l2 0.001 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s threshold walk-forward, zero fees =="
  python3 -m lob_forge.cli walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s logistic walk-forward, zero fees =="
  python3 -m lob_forge.cli logistic-walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --epochs 120 \
    --learning-rate 0.05 \
    --l2 0.001 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s fixed-rule regime diagnostics, zero fees =="
  python3 -m lob_forge.cli regime "$ETH_5S" \
    --feature microprice_deviation \
    --threshold 0.15 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --bins 3 \
    --taker-fee-bps 0

  echo
  echo "== ETHUSDT 5s threshold walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl
}

run_conditionals() {
  echo "== BTCUSDT 5s conditional walk-forward, zero fees =="
  python3 -m lob_forge.cli conditional-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --regime-bins 3 \
    --min-validation-trades 50 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s conditional walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli conditional-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --regime-bins 3 \
    --min-validation-trades 50 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s conditional walk-forward, zero fees =="
  python3 -m lob_forge.cli conditional-walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --regime-bins 3 \
    --min-validation-trades 50 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s conditional walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli conditional-walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --regime-bins 3 \
    --min-validation-trades 50 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl
}

run_edge() {
  echo "== BTCUSDT 5s expected-edge ridge walk-forward, zero fees =="
  python3 -m lob_forge.cli edge-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s expected-edge ridge walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli edge-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s expected-edge ridge walk-forward, zero fees =="
  python3 -m lob_forge.cli edge-walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s expected-edge ridge walk-forward, 5 bps taker fees =="
  python3 -m lob_forge.cli edge-walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl
}

case "$MODE" in
  build)
    run_builds
    ;;
  results)
    run_results
    ;;
  conditional)
    run_conditionals
    ;;
  edge)
    run_edge
    ;;
  all)
    run_builds
    run_results
    run_conditionals
    run_edge
    ;;
  *)
    echo "Usage: bash scripts/reproduce_core_results.sh [build|results|conditional|edge|all]" >&2
    exit 2
    ;;
esac
