#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_TAG="${1:-expected_edge_60day_20230516_20230714}"
PROCESSED_TARGET="data/processed/${TARGET_TAG}"
RESULT_TARGET="results/${TARGET_TAG}"

printf 'This will delete:\n'
printf '  %s\n' "$PROCESSED_TARGET"
printf '  %s\n' "$RESULT_TARGET"
printf 'Raw downloads under data/raw are kept for reuse.\n'
printf 'Type DELETE to continue: '
read -r answer

if [[ "$answer" != "DELETE" ]]; then
  echo "aborted"
  exit 0
fi

rm -rf "$PROCESSED_TARGET" "$RESULT_TARGET"
echo "deleted partial study artifacts for ${TARGET_TAG}"
