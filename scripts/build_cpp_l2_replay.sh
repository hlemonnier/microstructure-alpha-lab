#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT_DIR/build/l2_replay}"

mkdir -p "$(dirname "$OUTPUT")"
c++ -std=c++17 -O2 "$ROOT_DIR/cpp/l2_replay.cpp" -o "$OUTPUT"
echo "$OUTPUT"
