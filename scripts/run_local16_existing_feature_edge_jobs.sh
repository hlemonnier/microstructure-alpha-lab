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
source scripts/holdout_manifest.sh
source scripts/source_provenance.sh

RESULT_DIR="${RESULT_DIR:-results/expected_edge_local16_20230516_20230714}"
PLAN_PATH="${PLAN_PATH:-$RESULT_DIR/run_plan.json}"
FEATURE_STATUS_PATH="${FEATURE_STATUS_PATH:-$RESULT_DIR/feature_status.json}"
DRY_RUN="${DRY_RUN:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
MAX_FOLDS="${MAX_FOLDS:-}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-20}"
RERUN_UNDERFOLDED="${RERUN_UNDERFOLDED:-1}"
MAX_LOAD_MEMORY_GB="${MAX_LOAD_MEMORY_GB:-10}"
LOB_FORGE_MAX_PROCESS_MEMORY_GB="${LOB_FORGE_MAX_PROCESS_MEMORY_GB:-8}"
EDGE_THRESHOLDS_BPS="${EDGE_THRESHOLDS_BPS:-0,0.05,0.1,0.2,0.5}"
FEES_BPS="${FEES_BPS:-}"

export LOB_FORGE_MAX_PROCESS_MEMORY_GB

if [[ "$DRY_RUN" == "0" && "$(source_worktree_dirty)" == "true" ]]; then
  echo "refusing certified edge jobs from a dirty working tree; commit the exact source first" >&2
  exit 2
fi

"$PYTHON_BIN" -m lob_forge.study_features \
  --plan "$PLAN_PATH" \
  --output "$FEATURE_STATUS_PATH" \
  || true

read -r TRAIN_SIZE VALIDATION_SIZE TEST_SIZE STEP_SIZE DEFAULT_FEES <<< "$(
python3 - "$PLAN_PATH" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    plan = json.load(handle)
fees = " ".join(format(float(fee), "g") for fee in plan.get("fees_bps", []))
print(
    int(plan["train_size"]),
    int(plan["validation_size"]),
    int(plan["test_size"]),
    int(plan["step_size"]),
    fees,
)
PY
)"

if [[ -z "$FEES_BPS" ]]; then
  FEES_BPS="$DEFAULT_FEES"
fi

mkdir -p "$RESULT_DIR"
COMPLETE_JOBS_FILE="$(mktemp)"
cleanup() {
  rm -f "$COMPLETE_JOBS_FILE"
  rm -f "$RESULT_DIR"/*.tmp.$$
}
trap cleanup EXIT
job_count=0

audit_meets_min_fold_count() {
  local audit_path="$1"
  python3 - "$audit_path" "$MIN_AUDIT_FOLD_COUNT" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
minimum = int(sys.argv[2])
try:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty audit")
    fold_count = int(float(rows[0].get("fold_count", "0") or 0))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if fold_count >= minimum else 1)
PY
}

python3 - "$FEATURE_STATUS_PATH" > "$COMPLETE_JOBS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    status = json.load(handle)
for job in status.get("feature_jobs", []):
    if job.get("complete"):
        print("\t".join([job["symbol"], str(job["horizon_ms"]), job["combined_path"]]))
PY

if [[ ! -s "$COMPLETE_JOBS_FILE" ]]; then
  echo "no complete feature jobs found in $FEATURE_STATUS_PATH" >&2
  exit 1
fi

while IFS=$'\t' read -r symbol horizon_ms combined; do
  if [[ "$MAX_JOBS" != "0" && "$job_count" -ge "$MAX_JOBS" ]]; then
    break
  fi
  job_count=$((job_count + 1))
  for fee in $FEES_BPS; do
    safe_fee="$(printf '%s' "$fee" | tr '.' 'p')"
    result="$RESULT_DIR/${symbol}_${horizon_ms}ms_fee_${safe_fee}_edge.csv"
    audit="$RESULT_DIR/${symbol}_${horizon_ms}ms_fee_${safe_fee}_edge_audit.csv"
    provenance="${result}.provenance.json"
    holdout_manifest="$(holdout_manifest_for "$combined")"
    if [[ -s "$result" && -s "$audit" && -s "$provenance" ]] && \
      "$PYTHON_BIN" -m lob_forge.study_provenance verify \
        --plan "$PLAN_PATH" \
        --result "$result" \
        --audit "$audit" \
        --symbol "$symbol" \
        --horizon-ms "$horizon_ms" \
        --taker-fee-bps "$fee" \
        --output "$provenance" \
        --require-source-files >/dev/null; then
      if [[ "$RERUN_UNDERFOLDED" == "1" ]] && ! audit_meets_min_fold_count "$audit"; then
        printf 'rerun underfolded symbol=%s horizon_ms=%s fee=%s audit=%s min_audit_fold_count=%s\n' \
          "$symbol" "$horizon_ms" "$fee" "$audit" "$MIN_AUDIT_FOLD_COUNT"
      else
        printf 'skip existing symbol=%s horizon_ms=%s fee=%s\n' "$symbol" "$horizon_ms" "$fee"
        continue
      fi
    elif [[ -s "$result" || -s "$audit" || -s "$provenance" ]]; then
      printf 'rerun invalid existing symbol=%s horizon_ms=%s fee=%s\n' "$symbol" "$horizon_ms" "$fee"
    fi
    if [[ "$DRY_RUN" != "0" ]]; then
      printf 'would_run symbol=%s horizon_ms=%s fee=%s combined=%s result=%s\n' \
        "$symbol" "$horizon_ms" "$fee" "$combined" "$result"
      continue
    fi
    printf 'run symbol=%s horizon_ms=%s fee=%s combined=%s result=%s\n' \
      "$symbol" "$horizon_ms" "$fee" "$combined" "$result"
    result_tmp="$result.tmp.$$"
    audit_tmp="$audit.tmp.$$"
    max_folds_args=()
    if [[ -n "$MAX_FOLDS" ]]; then
      max_folds_args=(--max-folds "$MAX_FOLDS")
    fi
    python3 -m lob_forge.cli edge-walk-forward "$combined" \
      --holdout-manifest "$holdout_manifest" \
      --train-size "$TRAIN_SIZE" \
      --validation-size "$VALIDATION_SIZE" \
      --test-size "$TEST_SIZE" \
      --step-size "$STEP_SIZE" \
      "${max_folds_args[@]}" \
      --l2 1 \
      --taker-fee-bps "$fee" \
      --max-load-memory-gb "$MAX_LOAD_MEMORY_GB" \
      --stream \
      --edge-thresholds-bps "$EDGE_THRESHOLDS_BPS" \
      --sort-by validation_net_pnl \
      > "$result_tmp"
    mv "$result_tmp" "$result"
    python3 -m lob_forge.cli audit-results "$result" \
      --assumed-cost-bps "$fee" \
      --cost-safety-multiple 2 \
      > "$audit_tmp"
    mv "$audit_tmp" "$audit"
    "$PYTHON_BIN" -m lob_forge.study_provenance write \
      --plan "$PLAN_PATH" \
      --feature "$combined" \
      --holdout-manifest "$holdout_manifest" \
      --result "$result" \
      --audit "$audit" \
      --symbol "$symbol" \
      --horizon-ms "$horizon_ms" \
      --taker-fee-bps "$fee" \
      --output "$provenance"
  done
done < "$COMPLETE_JOBS_FILE"

if [[ "$DRY_RUN" == "0" ]]; then
  python3 -m lob_forge.study_registry \
    --plan "$PLAN_PATH" \
    --result-dir "$RESULT_DIR" \
    --output "$RESULT_DIR/candidate_registry.jsonl" \
    --pvalues-output "$RESULT_DIR/pvalues.csv"
  python3 -m lob_forge.cli pvalue-correction "$RESULT_DIR/pvalues.csv" > "$RESULT_DIR/pvalue_corrections.csv"
  python3 -m lob_forge.study_status \
    --plan "$PLAN_PATH" \
    --result-dir "$RESULT_DIR" \
    --output "$RESULT_DIR/study_status.json" \
    --min-audit-fold-count "$MIN_AUDIT_FOLD_COUNT" \
    || true
fi
