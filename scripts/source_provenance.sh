#!/usr/bin/env bash

source_git_rev() {
  local root_dir="${ROOT_DIR:-$(pwd)}"
  local env_commit="${LOB_FORGE_SOURCE_GIT_COMMIT:-}"
  local archive_commit_file="${SOURCE_ARCHIVE_COMMIT_FILE:-.source-git-commit}"
  local archive_path="$root_dir/$archive_commit_file"
  local value

  if [[ -n "$env_commit" ]]; then
    if [[ "$env_commit" =~ ^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$ ]]; then
      printf '%s\n' "$env_commit"
      return 0
    fi
    echo "LOB_FORGE_SOURCE_GIT_COMMIT must be a 40- or 64-character Git commit hash" >&2
    return 2
  fi

  if value="$(git -C "$root_dir" rev-parse HEAD 2>/dev/null)"; then
    if [[ "$value" =~ ^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$ ]]; then
      printf '%s\n' "$value"
      return 0
    fi
    echo "git rev-parse HEAD did not return a valid commit hash" >&2
    return 2
  fi

  if [[ -f "$archive_path" ]]; then
    value="$(tr -d '[:space:]' < "$archive_path")"
    if [[ -n "$value" && "$value" != *'$Format'* ]]; then
      if [[ "$value" =~ ^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$ ]]; then
        printf '%s\n' "$value"
        return 0
      fi
      echo "$archive_commit_file must contain a 40- or 64-character Git commit hash" >&2
      return 2
    fi
  fi

  echo "source provenance requires Git metadata, an expanded $archive_commit_file, or LOB_FORGE_SOURCE_GIT_COMMIT=<commit-hash>" >&2
  return 2
}

source_worktree_dirty() {
  local root_dir="${ROOT_DIR:-$(pwd)}"
  local env_dirty="${LOB_FORGE_SOURCE_WORKING_TREE_DIRTY:-}"
  local normalized_env
  local status_output

  if [[ -n "$env_dirty" ]]; then
    normalized_env="$(printf '%s' "$env_dirty" | tr '[:upper:]' '[:lower:]')"
    case "$normalized_env" in
      true|1|dirty)
        printf 'true\n'
        return 0
        ;;
      false|0|clean)
        printf 'false\n'
        return 0
        ;;
      null|none|unknown)
        printf 'null\n'
        return 0
        ;;
      *)
        echo "LOB_FORGE_SOURCE_WORKING_TREE_DIRTY must be true, false, or null" >&2
        return 2
        ;;
    esac
  fi

  if git -C "$root_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    status_output="$(git -C "$root_dir" status --short --untracked-files=no)"
    if [[ -n "$status_output" ]]; then
      printf 'true\n'
    else
      printf 'false\n'
    fi
    return 0
  fi

  printf 'null\n'
}
