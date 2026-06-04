#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PLAN_ONLY=1 bash "$ROOT_DIR/scripts/run_60day_expected_edge_study.sh"
