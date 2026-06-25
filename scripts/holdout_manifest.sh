#!/usr/bin/env bash

holdout_manifest_for() {
  local feature_csv="$1"
  local split_column="${HOLDOUT_SPLIT_COLUMN:-source_date}"
  local manifest_dir="${HOLDOUT_MANIFEST_DIR:-results/holdout_manifests}"
  local python_bin="${PYTHON_BIN:-python3}"
  local root_dir="${ROOT_DIR:-$(pwd)}"
  local slug
  local manifest
  local holdout_values

  slug="$(basename "$feature_csv" .csv)"
  manifest="$manifest_dir/${slug}.holdout.json"
  mkdir -p "$manifest_dir"
  if [[ "${RECREATE_HOLDOUT_MANIFESTS:-0}" == "1" ]]; then
    rm -f "$manifest"
  fi
  if [[ ! -f "$manifest" ]]; then
    holdout_values="${HOLDOUT_VALUES:-}"
    if [[ -z "$holdout_values" ]]; then
      holdout_values="$("$python_bin" - "$feature_csv" "$split_column" <<'PY'
import csv
import sys

path, split_column = sys.argv[1], sys.argv[2]
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    if split_column not in (reader.fieldnames or []):
        raise SystemExit(f"split column {split_column!r} missing from {path}")
    values = sorted({row[split_column] for row in reader if row.get(split_column)})
if not values:
    raise SystemExit(f"no nonempty {split_column!r} values found in {path}")
print(values[-1])
PY
)"
    fi
    "$python_bin" -m lob_forge.cli create-holdout-manifest "$feature_csv" \
      --output "$manifest" \
      --split-column "$split_column" \
      --holdout-values "$holdout_values" \
      --source-root "$root_dir" >&2
  fi
  printf '%s' "$manifest"
}
