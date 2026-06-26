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
SHADOW_PATH="${SHADOW_PATH:-results/shadow_validation/shadow_decisions.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-results/shadow_validation/observed_fills_template.csv}"
ORDER_PLAN_DIR="${ORDER_PLAN_DIR:-results/shadow_validation}"
ORDER_PLAN_PROVIDERS="${ORDER_PLAN_PROVIDERS:-bybit okx binance}"
LIMIT="${LIMIT:-50}"
DRY_RUN="${DRY_RUN:-1}"

if [[ ! -s "$SHADOW_PATH" ]]; then
  cat >&2 <<EOF
missing shadow decisions: $SHADOW_PATH
Run edge-shadow-decisions first, then rerun this helper.
EOF
  exit 2
fi

printf 'shadow=%s\n' "$SHADOW_PATH"
printf 'observed_fill_template=%s\n' "$OUTPUT_PATH"
printf 'order_plan_dir=%s\n' "$ORDER_PLAN_DIR"
printf 'order_plan_providers=%s\n' "$ORDER_PLAN_PROVIDERS"
printf 'limit=%s\n' "$LIMIT"
printf 'dry_run=%s\n' "$DRY_RUN"

order_plan_path_for() {
  local provider="$1"
  if [[ "$provider" == "binance" ]]; then
    printf '%s/binance_usdm_order_plan.jsonl' "$ORDER_PLAN_DIR"
  else
    printf '%s/%s_order_plan.jsonl' "$ORDER_PLAN_DIR" "$provider"
  fi
}

if [[ "$DRY_RUN" != "0" ]]; then
  printf 'would_run %s -m lob_forge.cli observed-fill-template --shadow %s --output %s --limit %s\n' \
    "$PYTHON_BIN" "$SHADOW_PATH" "$OUTPUT_PATH" "$LIMIT"
  for provider in $ORDER_PLAN_PROVIDERS; do
    plan_path="$(order_plan_path_for "$provider")"
    printf 'would_run %s -m lob_forge.cli paper-order-plan --shadow %s --provider %s --output %s --limit %s\n' \
      "$PYTHON_BIN" "$SHADOW_PATH" "$provider" "$plan_path" "$LIMIT"
  done
  printf 'Set DRY_RUN=0 to write the template and provider order plans. Blank templates are not evidence and are ignored by import-observed-fills.\n'
  exit 0
fi

"$PYTHON_BIN" -m lob_forge.cli observed-fill-template \
  --shadow "$SHADOW_PATH" \
  --output "$OUTPUT_PATH" \
  --limit "$LIMIT"

for provider in $ORDER_PLAN_PROVIDERS; do
  plan_path="$(order_plan_path_for "$provider")"
  "$PYTHON_BIN" -m lob_forge.cli paper-order-plan \
    --shadow "$SHADOW_PATH" \
    --provider "$provider" \
    --output "$plan_path" \
    --limit "$LIMIT"
done
