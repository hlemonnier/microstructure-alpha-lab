#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-results}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH
if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
source scripts/holdout_manifest.sh

BTC_5S="data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv"
ETH_5S="data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv"
HOLDOUT_MANIFEST_DIR="${HOLDOUT_MANIFEST_DIR:-artifacts/holdout_manifests/core}"
HOLDOUT_SPLIT_COLUMN="${HOLDOUT_SPLIT_COLUMN:-event_time}"

holdout_value_for() {
  python3 - "$1" "$HOLDOUT_SPLIT_COLUMN" <<'PY'
import csv
import sys

path, column = sys.argv[1:3]
last = ""
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    if column not in (reader.fieldnames or []):
        raise SystemExit(f"missing holdout split column: {column}")
    for row in reader:
        if row.get(column):
            last = row[column]
if not last:
    raise SystemExit("could not infer holdout value")
print(last)
PY
}

ensure_holdout_manifest() {
  local path="$1"
  local slug
  slug="$(basename "$path" .csv)"
  local manifest="$HOLDOUT_MANIFEST_DIR/${slug}_${HOLDOUT_SPLIT_COLUMN}_holdout.json"
  mkdir -p "$HOLDOUT_MANIFEST_DIR"
  if [[ ! -s "$manifest" ]]; then
    local value
    value="$(holdout_value_for "$path")"
    python3 -m lob_forge.cli create-holdout-manifest "$path" \
      --output "$manifest" \
      --split-column "$HOLDOUT_SPLIT_COLUMN" \
      --holdout-values "$value" \
      --source-root "$ROOT_DIR" \
      --notes "Auto-created by scripts/reproduce_core_results.sh" >&2
  fi
  printf '%s' "$manifest"
}

run_with_holdout() {
  local command="$1"
  local path="$2"
  shift 2
  local manifest
  manifest="$(ensure_holdout_manifest "$path")"
  python3 -m lob_forge.cli "$command" "$path" --holdout-manifest "$manifest" "$@"
}

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
  run_with_holdout walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s threshold walk-forward, 5 bps taker fees =="
  run_with_holdout walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s fixed-rule regime diagnostics, zero fees =="
  run_with_holdout regime "$BTC_5S" \
    --feature microprice_deviation \
    --threshold 0.05 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --bins 3 \
    --taker-fee-bps 0

  echo
  echo "== BTCUSDT 5s logistic walk-forward, zero fees =="
  run_with_holdout logistic-walk-forward "$BTC_5S" \
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
  run_with_holdout walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s logistic walk-forward, zero fees =="
  run_with_holdout logistic-walk-forward "$ETH_5S" \
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
  run_with_holdout regime "$ETH_5S" \
    --feature microprice_deviation \
    --threshold 0.15 \
    --regime-features spread_mean_5,realized_volatility_5,notional_imbalance_1pct \
    --bins 3 \
    --taker-fee-bps 0

  echo
  echo "== ETHUSDT 5s threshold walk-forward, 5 bps taker fees =="
  run_with_holdout walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl
}

run_conditionals() {
  echo "== BTCUSDT 5s conditional walk-forward, zero fees =="
  run_with_holdout conditional-walk-forward "$BTC_5S" \
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
  run_with_holdout conditional-walk-forward "$BTC_5S" \
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
  run_with_holdout conditional-walk-forward "$ETH_5S" \
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
  run_with_holdout conditional-walk-forward "$ETH_5S" \
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
  run_with_holdout edge-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== BTCUSDT 5s expected-edge ridge walk-forward, 5 bps taker fees =="
  run_with_holdout edge-walk-forward "$BTC_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 5 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s expected-edge ridge walk-forward, zero fees =="
  run_with_holdout edge-walk-forward "$ETH_5S" \
    --train-size 2400 \
    --validation-size 1200 \
    --test-size 1200 \
    --step-size 1200 \
    --l2 1 \
    --taker-fee-bps 0 \
    --sort-by validation_net_pnl

  echo
  echo "== ETHUSDT 5s expected-edge ridge walk-forward, 5 bps taker fees =="
  run_with_holdout edge-walk-forward "$ETH_5S" \
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
