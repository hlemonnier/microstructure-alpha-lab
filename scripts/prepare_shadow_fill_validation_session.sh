#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

PYTHON_BIN="${PYTHON_BIN:-python3}"
SHADOW_PATH="${SHADOW_PATH:-results/shadow_validation/shadow_decisions.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-results/shadow_validation/observed_fills_template.csv}"
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
printf 'limit=%s\n' "$LIMIT"
printf 'dry_run=%s\n' "$DRY_RUN"

if [[ "$DRY_RUN" != "0" ]]; then
  printf 'would_run %s -m lob_forge.cli observed-fill-template --shadow %s --output %s --limit %s\n' \
    "$PYTHON_BIN" "$SHADOW_PATH" "$OUTPUT_PATH" "$LIMIT"
  printf 'Set DRY_RUN=0 to write the template. A blank template is not evidence and is ignored by import-observed-fills.\n'
  exit 0
fi

"$PYTHON_BIN" -m lob_forge.cli observed-fill-template \
  --shadow "$SHADOW_PATH" \
  --output "$OUTPUT_PATH" \
  --limit "$LIMIT"
