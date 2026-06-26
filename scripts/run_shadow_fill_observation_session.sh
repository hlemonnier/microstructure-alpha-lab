#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

PROVIDER="${PROVIDER:-bybit}"
SHADOW_PATH="${SHADOW_PATH:-results/shadow_validation/shadow_decisions.csv}"
SIMULATED_PATH="${SIMULATED_PATH:-results/shadow_validation/simulated_fills.csv}"
OUT_DIR="${OUT_DIR:-results/shadow_validation}"
LIMIT="${LIMIT:-50}"
START_TIME_MS="${START_TIME_MS:-}"
END_TIME_MS="${END_TIME_MS:-}"
DRY_RUN="${DRY_RUN:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
SUBMIT_ORDERS="${SUBMIT_ORDERS:-0}"
MAX_PRICE_ERROR="${MAX_PRICE_ERROR:-0.5}"
MAX_SIZE_ERROR="${MAX_SIZE_ERROR:-0.01}"
MAX_FILL_RATE_ERROR="${MAX_FILL_RATE_ERROR:-0.05}"

case "$PROVIDER" in
  bybit)
    : "${SYMBOL:=BTCUSDT}"
    RAW_OUTPUT="${RAW_OUTPUT:-$OUT_DIR/raw_bybit_executions.json}"
    OBSERVED_OUTPUT="${OBSERVED_OUTPUT:-$OUT_DIR/observed_fills.csv}"
    MERGED_SHADOW_OUTPUT="${MERGED_SHADOW_OUTPUT:-$OUT_DIR/shadow_decisions_observed.csv}"
    VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-$OUT_DIR/shadow_fill_validation.txt}"
    ORDER_PLAN_PATH="${ORDER_PLAN_PATH:-$OUT_DIR/bybit_order_plan.jsonl}"
    ORDER_SUBMISSION_OUTPUT="${ORDER_SUBMISSION_OUTPUT:-$OUT_DIR/submitted_bybit_orders.jsonl}"
    REQUIRED_ENV=(BYBIT_DEMO_API_KEY BYBIT_DEMO_API_SECRET)
    ;;
  okx)
    : "${SYMBOL:=BTC-USDT-SWAP}"
    RAW_OUTPUT="${RAW_OUTPUT:-$OUT_DIR/raw_okx_fills.json}"
    OBSERVED_OUTPUT="${OBSERVED_OUTPUT:-$OUT_DIR/observed_fills.csv}"
    MERGED_SHADOW_OUTPUT="${MERGED_SHADOW_OUTPUT:-$OUT_DIR/shadow_decisions_observed.csv}"
    VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-$OUT_DIR/shadow_fill_validation.txt}"
    ORDER_PLAN_PATH="${ORDER_PLAN_PATH:-$OUT_DIR/okx_order_plan.jsonl}"
    ORDER_SUBMISSION_OUTPUT="${ORDER_SUBMISSION_OUTPUT:-$OUT_DIR/submitted_okx_orders.jsonl}"
    REQUIRED_ENV=(OKX_DEMO_API_KEY OKX_DEMO_API_SECRET OKX_DEMO_API_PASSPHRASE)
    ;;
  binance)
    : "${SYMBOL:=BTCUSDT}"
    RAW_OUTPUT="${RAW_OUTPUT:-$OUT_DIR/raw_binance_usdm_orders.json}"
    OBSERVED_OUTPUT="${OBSERVED_OUTPUT:-$OUT_DIR/observed_fills.csv}"
    MERGED_SHADOW_OUTPUT="${MERGED_SHADOW_OUTPUT:-$OUT_DIR/shadow_decisions_observed.csv}"
    VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-$OUT_DIR/shadow_fill_validation.txt}"
    ORDER_PLAN_PATH="${ORDER_PLAN_PATH:-$OUT_DIR/binance_usdm_order_plan.jsonl}"
    ORDER_SUBMISSION_OUTPUT="${ORDER_SUBMISSION_OUTPUT:-$OUT_DIR/submitted_binance_usdm_orders.jsonl}"
    REQUIRED_ENV=(BINANCE_USDM_TESTNET_API_KEY BINANCE_USDM_TESTNET_API_SECRET)
    ;;
  *)
    echo "unknown PROVIDER=$PROVIDER; expected bybit, okx, or binance" >&2
    exit 2
    ;;
esac

if [[ ! -s "$SHADOW_PATH" ]]; then
  echo "missing shadow decisions: $SHADOW_PATH" >&2
  exit 2
fi
if [[ "$RUN_VALIDATE" == "1" && ! -s "$SIMULATED_PATH" ]]; then
  echo "missing simulated fills: $SIMULATED_PATH" >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" && "$SUBMIT_ORDERS" == "1" && ! -s "$ORDER_PLAN_PATH" ]]; then
  echo "missing order plan for submission: $ORDER_PLAN_PATH" >&2
  exit 2
fi

missing_env=()
for name in "${REQUIRED_ENV[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    missing_env+=("$name")
  fi
done

fetch_command=(
  "$PYTHON_BIN" -m lob_forge.cli fetch-observed-fills
  --provider "$PROVIDER"
  --symbol "$SYMBOL"
  --limit "$LIMIT"
  --output "$RAW_OUTPUT"
)
if [[ -n "$START_TIME_MS" ]]; then
  fetch_command+=(--start-time-ms "$START_TIME_MS")
fi
if [[ -n "$END_TIME_MS" ]]; then
  fetch_command+=(--end-time-ms "$END_TIME_MS")
fi

submit_command=(
  "$PYTHON_BIN" -m lob_forge.cli submit-paper-orders
  --provider "$PROVIDER"
  --plan "$ORDER_PLAN_PATH"
  --output "$ORDER_SUBMISSION_OUTPUT"
  --limit "$LIMIT"
)
normalize_command=(
  "$PYTHON_BIN" -m lob_forge.cli normalize-observed-fills
  --provider "$PROVIDER"
  --input "$RAW_OUTPUT"
  --output "$OBSERVED_OUTPUT"
)
import_command=(
  "$PYTHON_BIN" -m lob_forge.cli import-observed-fills
  --shadow "$SHADOW_PATH"
  --observed "$OBSERVED_OUTPUT"
  --output "$MERGED_SHADOW_OUTPUT"
)
validate_command=(
  "$PYTHON_BIN" -m lob_forge.cli validate-shadow-fills
  --simulated "$SIMULATED_PATH"
  --shadow "$MERGED_SHADOW_OUTPUT"
  --max-price-error "$MAX_PRICE_ERROR"
  --max-size-error "$MAX_SIZE_ERROR"
  --max-fill-rate-error "$MAX_FILL_RATE_ERROR"
)

printf 'provider=%s symbol=%s shadow=%s simulated=%s dry_run=%s run_validate=%s\n' \
  "$PROVIDER" "$SYMBOL" "$SHADOW_PATH" "$SIMULATED_PATH" "$DRY_RUN" "$RUN_VALIDATE"
printf 'raw_output=%s observed_output=%s merged_shadow_output=%s validation_output=%s\n' \
  "$RAW_OUTPUT" "$OBSERVED_OUTPUT" "$MERGED_SHADOW_OUTPUT" "$VALIDATION_OUTPUT"
printf 'submit_orders=%s order_plan=%s order_submission_output=%s\n' \
  "$SUBMIT_ORDERS" "$ORDER_PLAN_PATH" "$ORDER_SUBMISSION_OUTPUT"
printf 'required_env=%s\n' "${REQUIRED_ENV[*]}"
if [[ "${#missing_env[@]}" -gt 0 ]]; then
  printf 'missing_env=%s\n' "${missing_env[*]}"
fi

if [[ "$DRY_RUN" != "0" ]]; then
  if [[ "$SUBMIT_ORDERS" == "1" ]]; then
    printf 'would_run '
    printf '%q ' "${submit_command[@]}"
    printf '\n'
  else
    printf 'order_submission_skipped=1\n'
  fi
  printf 'would_run '
  printf '%q ' "${fetch_command[@]}"
  printf '\nwould_run '
  printf '%q ' "${normalize_command[@]}"
  printf '\nwould_run '
  printf '%q ' "${import_command[@]}"
  if [[ "$RUN_VALIDATE" == "1" ]]; then
    printf '\nwould_run '
    printf '%q ' "${validate_command[@]}"
    printf '> %q\n' "$VALIDATION_OUTPUT"
  else
    printf '\nvalidation_skipped=1\n'
  fi
  printf 'Set SUBMIT_ORDERS=1 DRY_RUN=0 to submit demo/testnet orders from the matching paper-order-plan before fetching fills.\n'
  exit 0
fi

if [[ "${#missing_env[@]}" -gt 0 ]]; then
  echo "missing provider credentials: ${missing_env[*]}" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
if [[ "$SUBMIT_ORDERS" == "1" ]]; then
  "${submit_command[@]}" --execute
  printf 'order_submission_report=%s\n' "$ORDER_SUBMISSION_OUTPUT"
else
  printf 'order_submission_skipped=1\n'
fi
"${fetch_command[@]}"
"${normalize_command[@]}"
"${import_command[@]}"
if [[ "$RUN_VALIDATE" == "1" ]]; then
  "${validate_command[@]}" > "$VALIDATION_OUTPUT"
  printf 'validation_report=%s\n' "$VALIDATION_OUTPUT"
else
  printf 'validation_skipped=1\n'
fi
