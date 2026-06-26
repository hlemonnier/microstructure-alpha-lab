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

RESULT_DIR="${RESULT_DIR:-results/expected_edge_60day_20230516_20230714}"
PLAN_PATH="${PLAN_PATH:-$RESULT_DIR/run_plan.json}"
CANDIDATE_REGISTRY="${CANDIDATE_REGISTRY:-$RESULT_DIR/candidate_registry.jsonl}"
STUDY_STATUS_OUTPUT="${STUDY_STATUS_OUTPUT:-$RESULT_DIR/final_holdout_study_status.json}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-$RESULT_DIR/final_holdout_selection.json}"

DRY_RUN="${DRY_RUN:-1}"
REQUIRE_STUDY_COMPLETE="${REQUIRE_STUDY_COMPLETE:-1}"
REQUIRE_PVALUES="${REQUIRE_PVALUES:-1}"
REQUIRE_ACCEPTED_CANDIDATE="${REQUIRE_ACCEPTED_CANDIDATE:-1}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-20}"

SELECTION_METRIC="${SELECTION_METRIC:-validation_net_pnl}"
SORT_BY="${SORT_BY:-$SELECTION_METRIC}"
HOLDOUT_SPLIT_COLUMN="${HOLDOUT_SPLIT_COLUMN:-source_date}"
HOLDOUT_VALUES="${HOLDOUT_VALUES:-}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"

CANDIDATE_OUTPUT="${CANDIDATE_OUTPUT:-results/final_holdout/final_edge_candidate.json}"
DEVELOPMENT_MANIFEST_OUTPUT="${DEVELOPMENT_MANIFEST_OUTPUT:-results/final_holdout/development_edge_holdout_manifest.json}"
FINAL_MANIFEST_OUTPUT="${FINAL_MANIFEST_OUTPUT:-results/final_holdout/final_edge_holdout_manifest.json}"
FINAL_OUTPUT="${FINAL_OUTPUT:-results/final_holdout/final_holdout_edge_result.json}"

printf 'result_dir=%s\n' "$RESULT_DIR"
printf 'plan_path=%s\n' "$PLAN_PATH"
printf 'candidate_registry=%s\n' "$CANDIDATE_REGISTRY"
printf 'dry_run=%s require_study_complete=%s require_accepted_candidate=%s\n' \
  "$DRY_RUN" "$REQUIRE_STUDY_COMPLETE" "$REQUIRE_ACCEPTED_CANDIDATE"

if [[ "$REQUIRE_STUDY_COMPLETE" == "1" ]]; then
  if [[ "$REQUIRE_PVALUES" == "1" ]]; then
    study_status_command=(
      "$PYTHON_BIN" -m lob_forge.study_status
      --plan "$PLAN_PATH"
      --result-dir "$RESULT_DIR"
      --output "$STUDY_STATUS_OUTPUT"
      --min-audit-fold-count "$MIN_AUDIT_FOLD_COUNT"
    )
  else
    study_status_command=(
      "$PYTHON_BIN" -m lob_forge.study_status
      --plan "$PLAN_PATH"
      --result-dir "$RESULT_DIR"
      --output "$STUDY_STATUS_OUTPUT"
      --min-audit-fold-count "$MIN_AUDIT_FOLD_COUNT"
      --no-pvalues
    )
  fi
  if ! "${study_status_command[@]}"; then
    printf 'full_study_complete=0\n' >&2
    printf 'Refusing final-holdout preparation until the expected-edge study verifier passes.\n' >&2
    exit 2
  fi
fi

"$PYTHON_BIN" - "$PLAN_PATH" "$CANDIDATE_REGISTRY" "$SELECTION_OUTPUT" "$SELECTION_METRIC" "$REQUIRE_ACCEPTED_CANDIDATE" <<'PY'
import json
import math
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
selection_output = Path(sys.argv[3])
selection_metric = sys.argv[4]
require_accepted = sys.argv[5] == "1"

with plan_path.open() as handle:
    plan = json.load(handle)

attempts = []
with registry_path.open() as handle:
    for line in handle:
        if line.strip():
            attempts.append(json.loads(line))

def number(value: object, default: float = -math.inf) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

selected = [
    attempt
    for attempt in attempts
    if attempt.get("selected") is True
    and attempt.get("status") == "selected_in_artifact"
    and attempt.get("artifact_path")
]
if require_accepted:
    selected = [attempt for attempt in selected if attempt.get("audit_acceptance_passed") is True]
if not selected:
    accepted_text = " accepted" if require_accepted else ""
    raise SystemExit(f"no{accepted_text} selected expected-edge candidate found in {registry_path}")

def score(attempt: dict[str, object]) -> tuple[float, float, float, float, int, float, float]:
    return (
        1.0 if attempt.get("audit_acceptance_passed") is True else 0.0,
        number(attempt.get(selection_metric)),
        number(attempt.get("validation_net_pnl")),
        number(attempt.get("test_net_pnl")),
        int(number(attempt.get("selected_fold_count"), default=0.0)),
        -number(attempt.get("taker_fee_bps"), default=0.0),
        -number(attempt.get("edge_threshold_bps"), default=0.0),
    )

candidate = max(selected, key=score)
symbol = str(candidate["symbol"]).upper()
horizon_ms = int(candidate["horizon_ms"])
latency_ms = int(candidate.get("latency_ms", plan.get("latency_ms", 0)))
processed_root = Path(str(plan["processed_root"]))
feature_csv = (
    processed_root
    / f"{symbol.lower()}_{horizon_ms}ms_latency_{latency_ms}"
    / f"{symbol}-{plan['start_date']}_{plan['end_date']}-combined-features.csv"
)
artifact_path = Path(str(candidate["artifact_path"]))
payload = {
    "artifact_exists": artifact_path.exists() and artifact_path.stat().st_size > 0,
    "artifact_path": str(artifact_path),
    "audit_acceptance_passed": candidate.get("audit_acceptance_passed"),
    "audit_path": candidate.get("audit_path", ""),
    "edge_threshold_bps": candidate.get("edge_threshold_bps"),
    "feature_csv": str(feature_csv),
    "feature_csv_exists": feature_csv.exists() and feature_csv.stat().st_size > 0,
    "horizon_ms": horizon_ms,
    "latency_ms": latency_ms,
    "selection_metric": selection_metric,
    "selected_fold_count": candidate.get("selected_fold_count", 0),
    "status": candidate.get("status"),
    "symbol": symbol,
    "taker_fee_bps": candidate.get("taker_fee_bps"),
    "test_net_pnl": candidate.get("test_net_pnl"),
    "validation_net_pnl": candidate.get("validation_net_pnl"),
}
selection_output.parent.mkdir(parents=True, exist_ok=True)
selection_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"selection_output={selection_output}")
print(f"selected_candidate={symbol} horizon_ms={horizon_ms} taker_fee_bps={candidate.get('taker_fee_bps')} edge_threshold_bps={candidate.get('edge_threshold_bps')}")
print(f"feature_csv={feature_csv} exists={int(payload['feature_csv_exists'])}")
print(f"walk_forward_artifact={artifact_path} exists={int(payload['artifact_exists'])}")
PY

json_field() {
  "$PYTHON_BIN" - "$SELECTION_OUTPUT" "$1" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
value = payload[sys.argv[2]]
print("" if value is None else value)
PY
}

FEATURE_CSV="$(json_field feature_csv)"
WALK_FORWARD_ARTIFACT="$(json_field artifact_path)"
TAKER_FEE_BPS="$(json_field taker_fee_bps)"

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ -z "$HOLDOUT_VALUES" ]]; then
    printf 'HOLDOUT_VALUES is required when DRY_RUN=0.\n' >&2
    exit 2
  fi
  if [[ ! -s "$FEATURE_CSV" ]]; then
    printf 'selected feature CSV is missing or empty: %s\n' "$FEATURE_CSV" >&2
    exit 2
  fi
  if [[ ! -s "$WALK_FORWARD_ARTIFACT" ]]; then
    printf 'selected walk-forward artifact is missing or empty: %s\n' "$WALK_FORWARD_ARTIFACT" >&2
    exit 2
  fi
  for output in "$CANDIDATE_OUTPUT" "$DEVELOPMENT_MANIFEST_OUTPUT" "$FINAL_MANIFEST_OUTPUT" "$FINAL_OUTPUT"; do
    if [[ -e "$output" ]]; then
      printf 'refusing to overwrite existing final-holdout artifact: %s\n' "$output" >&2
      exit 2
    fi
  done
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p \
    "$(dirname "$CANDIDATE_OUTPUT")" \
    "$(dirname "$DEVELOPMENT_MANIFEST_OUTPUT")" \
    "$(dirname "$FINAL_MANIFEST_OUTPUT")" \
    "$(dirname "$FINAL_OUTPUT")"
fi

HOLDOUT_VALUES_FOR_CMD="$HOLDOUT_VALUES"
if [[ -z "$HOLDOUT_VALUES_FOR_CMD" ]]; then
  HOLDOUT_VALUES_FOR_CMD="<final_holdout_values>"
  printf 'holdout_values_missing=1\n'
  printf 'Set HOLDOUT_VALUES before running with DRY_RUN=0.\n'
fi

create_development_manifest=(
  "$PYTHON_BIN" -m lob_forge.cli create-holdout-manifest "$FEATURE_CSV"
  --output "$DEVELOPMENT_MANIFEST_OUTPUT"
  --split-column "$HOLDOUT_SPLIT_COLUMN"
  --holdout-values "$HOLDOUT_VALUES_FOR_CMD"
  --source-root "$SOURCE_ROOT"
  --notes "expected-edge development manifest for final-holdout candidate freezing"
)
freeze_candidate=(
  "$PYTHON_BIN" -m lob_forge.cli freeze-edge-candidate "$FEATURE_CSV"
  --holdout-manifest "$DEVELOPMENT_MANIFEST_OUTPUT"
  --output "$CANDIDATE_OUTPUT"
  --walk-forward-artifact "$WALK_FORWARD_ARTIFACT"
  --sort-by "$SORT_BY"
  --taker-fee-bps "$TAKER_FEE_BPS"
)
create_final_manifest=(
  "$PYTHON_BIN" -m lob_forge.cli create-holdout-manifest "$FEATURE_CSV"
  --output "$FINAL_MANIFEST_OUTPUT"
  --split-column "$HOLDOUT_SPLIT_COLUMN"
  --holdout-values "$HOLDOUT_VALUES_FOR_CMD"
  --candidate-json "$CANDIDATE_OUTPUT"
  --source-root "$SOURCE_ROOT"
  --notes "candidate-locked expected-edge final holdout manifest"
)
run_final_holdout=(
  "$PYTHON_BIN" -m lob_forge.cli final-holdout-edge "$FEATURE_CSV"
  --holdout-manifest "$FINAL_MANIFEST_OUTPUT"
  --candidate-json "$CANDIDATE_OUTPUT"
  --output "$FINAL_OUTPUT"
  --explicit-final-evaluation
)

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_or_print() {
  local label="$1"
  shift
  printf '%s\n' "$label"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

run_or_print 'step=create_development_holdout_manifest' "${create_development_manifest[@]}"
run_or_print 'step=freeze_edge_candidate' "${freeze_candidate[@]}"
run_or_print 'step=create_candidate_locked_final_manifest' "${create_final_manifest[@]}"
run_or_print 'step=run_final_holdout_edge' "${run_final_holdout[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'dry_run_complete=1\n'
else
  printf 'final_holdout_preparation_complete=1\n'
fi
