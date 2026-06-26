#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/source_provenance.sh

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DIST_DIR="${DIST_DIR:-dist}"
PACKAGE_NAME="${PACKAGE_NAME:-microstructure-alpha-lab-cloud-handoff-${STAMP}.zip}"
STAGING_DIR="$(mktemp -d)"
PROJECT_DIR="$STAGING_DIR/microstructure-alpha-lab"

cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

mkdir -p "$DIST_DIR" "$PROJECT_DIR"
PACKAGE_PATH="$(cd "$DIST_DIR" && pwd)/$PACKAGE_NAME"
rm -f "$PACKAGE_PATH"
SOURCE_GIT_COMMIT="$(source_git_rev)"

rsync -a \
  --include='/.gitignore' \
  --include='/Makefile' \
  --include='/README.md' \
  --include='/pyproject.toml' \
  --include='/requirements-ci.txt' \
  --include='/requirements-research.txt' \
  --include='/docs/***' \
  --include='/examples/***' \
  --include='/scripts/***' \
  --include='/src/***' \
  --include='/tests/***' \
  --exclude='*' \
  ./ "$PROJECT_DIR/"

printf '%s\n' "$SOURCE_GIT_COMMIT" > "$PROJECT_DIR/.source-git-commit"

find "$PROJECT_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$PROJECT_DIR" -type f -name '*.pyc' -delete

(
  cd "$STAGING_DIR"
  zip -qr "$PACKAGE_PATH" microstructure-alpha-lab
)

printf 'cloud_handoff_package=%s\n' "$PACKAGE_PATH"
printf 'source_git_commit=%s\n' "$SOURCE_GIT_COMMIT"
entry_count="$(unzip -Z1 "$PACKAGE_PATH" | wc -l | tr -d ' ')"
printf 'archive_entries=%s\n' "$entry_count"
