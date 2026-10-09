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
if [[ -e "$PACKAGE_PATH" ]]; then
  echo "Package already exists; choose a new PACKAGE_NAME: $PACKAGE_PATH" >&2
  exit 2
fi
SOURCE_GIT_COMMIT="$(source_git_rev)"
if [[ "$(source_worktree_dirty)" == "true" ]]; then
  echo "Source packaging requires a clean committed checkout." >&2
  exit 2
fi

rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='*.egg-info/' \
  --exclude='* [0-9].*' \
  --exclude='*.bak' \
  --exclude='*.tmp' \
  --exclude='*.swp' \
  --exclude='.DS_Store' \
  --include='/.gitignore' \
  --include='/.gitattributes' \
  --include='/.github/***' \
  --include='/Makefile' \
  --include='/README.md' \
  --include='/LICENSE' \
  --include='/CONTRIBUTING.md' \
  --include='/CITATION.cff' \
  --include='/IMPLEMENTATION_TRACEABILITY.md' \
  --include='/MANIFEST.in' \
  --include='/pyproject.toml' \
  --include='/requirements-ci.txt' \
  --include='/requirements-build.txt' \
  --include='/requirements-research.txt' \
  --include='/cpp/***' \
  --include='/data/' \
  --include='/data/README.md' \
  --include='/docs/***' \
  --include='/examples/***' \
  --include='/scripts/***' \
  --include='/src/***' \
  --include='/tests/***' \
  --exclude='*' \
  ./ "$PROJECT_DIR/"

printf '%s\n' "$SOURCE_GIT_COMMIT" > "$PROJECT_DIR/.source-git-commit"

(
  cd "$STAGING_DIR"
  zip -qr "$PACKAGE_PATH" microstructure-alpha-lab
)

printf 'cloud_handoff_package=%s\n' "$PACKAGE_PATH"
printf 'source_git_commit=%s\n' "$SOURCE_GIT_COMMIT"
entry_count="$(unzip -Z1 "$PACKAGE_PATH" | wc -l | tr -d ' ')"
printf 'archive_entries=%s\n' "$entry_count"
