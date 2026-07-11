from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
SEQUENCE_ECONOMICS_VERSION = "flat_to_flat_label_horizon_v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify local reduced E2E artifact provenance when present.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="artifacts/research_manifest.json")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    manifest_path = _resolve(root, args.manifest)
    if not manifest_path.exists():
        print(f"reduced_e2e_manifest={manifest_path} present=0 skipped=1")
        return 0

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"reduced_e2e_manifest={manifest_path} valid_json=0 error={exc.msg}")
        return 1
    if not isinstance(manifest, dict):
        print(f"reduced_e2e_manifest={manifest_path} json_object=0")
        return 1

    errors = _verify_research_manifest(root=root, manifest_path=manifest_path, manifest=manifest)
    print(f"reduced_e2e_manifest={manifest_path}")
    print(f"present=1 errors={len(errors)}")
    for error in errors:
        print(f"error={error}")
    return 0 if not errors else 1


def _verify_research_manifest(*, root: Path, manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    git_commit = str(manifest.get("git_commit") or "").strip()
    if not GIT_COMMIT_RE.fullmatch(git_commit):
        errors.append(f"invalid git_commit {git_commit!r}")

    head = _current_git_head(root)
    if head is not None and git_commit != head:
        errors.append(f"git_commit does not match HEAD: manifest={git_commit[:12]} head={head[:12]}")
    elif head is not None and not _git_rev_has_tag(root, git_commit):
        errors.append(f"git_commit has no local tag: {git_commit[:12]}")
    elif head is None:
        archive_commit = _source_archive_git_commit(root)
        if archive_commit is not None and git_commit != archive_commit:
            errors.append(
                f"git_commit does not match source archive: manifest={git_commit[:12]} archive={archive_commit[:12]}"
            )

    working_tree_dirty = manifest.get("working_tree_dirty")
    if head is not None and working_tree_dirty is not False:
        errors.append("working_tree_dirty must be false for local reduced E2E artifacts")
    elif head is None and working_tree_dirty not in {False, None}:
        errors.append("working_tree_dirty must be false or null for source-package reduced E2E artifacts")

    reduced_manifest = str(manifest.get("reduced_e2e_manifest") or "").strip()
    if not reduced_manifest:
        errors.append("missing reduced_e2e_manifest")
    else:
        _verify_reduced_manifest_path(root=root, reduced_manifest=reduced_manifest, errors=errors)

    artifacts = manifest.get("reduced_e2e_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("missing reduced_e2e_artifacts")
    else:
        for key, value in sorted(artifacts.items()):
            rel_path = str(value or "").strip()
            if not rel_path:
                errors.append(f"reduced_e2e_artifacts.{key} is empty")
                continue
            path = _resolve(root, rel_path)
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"reduced_e2e_artifacts.{key} missing or empty: {rel_path}")

    if manifest_path.parent != root / "artifacts":
        errors.append(f"unexpected manifest location: {manifest_path}")
    return errors


def _verify_reduced_manifest_path(*, root: Path, reduced_manifest: str, errors: list[str]) -> None:
    path = _resolve(root, reduced_manifest)
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"missing or empty reduced_e2e_manifest: {reduced_manifest}")
        return
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid reduced_e2e_manifest JSON: {exc.msg}")
        return
    if not isinstance(payload, dict):
        errors.append("reduced_e2e_manifest must be a JSON object")
        return
    if payload.get("claim_scope") != "Synthetic reduced E2E fixture only; no live or historical profitability claim.":
        errors.append("reduced_e2e_manifest claim_scope is not the expected synthetic-fixture disclaimer")
    _verify_sequence_smokes(root=root, payload=payload, errors=errors)


def _verify_sequence_smokes(*, root: Path, payload: dict[str, Any], errors: list[str]) -> None:
    smokes = payload.get("sequence_smokes", {})
    if not isinstance(smokes, dict):
        errors.append("sequence_smokes must be a JSON object")
        return
    for model_name, entry in sorted(smokes.items()):
        if not isinstance(entry, dict) or entry.get("status") != "pipeline_completed":
            continue
        artifact = _resolve(root, str(entry.get("path", "")))
        try:
            with artifact.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            errors.append(f"sequence_smokes.{model_name} artifact is unreadable")
            continue
        if len(rows) != 1:
            errors.append(f"sequence_smokes.{model_name} artifact must contain exactly one row")
            continue
        row = rows[0]
        if row.get("economic_simulation_version") != SEQUENCE_ECONOMICS_VERSION:
            errors.append(f"sequence_smokes.{model_name} uses stale economic simulation semantics")
        try:
            final_inventory = float(row.get("test_stateful_final_inventory", "nan"))
        except ValueError:
            final_inventory = math.nan
        if not math.isfinite(final_inventory) or abs(final_inventory) > 1e-9:
            errors.append(f"sequence_smokes.{model_name} has nonzero or invalid residual inventory")
        baseline = _resolve(root, str(payload.get("baseline_audit_fixture", "")))
        if not baseline.is_file() or row.get("baseline_audit_sha256") != _sha256_file(baseline):
            errors.append(f"sequence_smokes.{model_name} baseline audit hash mismatch")
        for label, path_key, hash_field in (
            ("holdout manifest", "holdout_manifest", "holdout_manifest_sha256"),
            ("checkpoint", "checkpoint", "checkpoint_sha256"),
            ("predictions", "predictions", "prediction_output_sha256"),
        ):
            source = _resolve(root, str(entry.get(path_key, "")))
            if not source.is_file() or source.stat().st_size <= 0:
                errors.append(f"sequence_smokes.{model_name} {label} is missing")
                continue
            if row.get(hash_field) != _sha256_file(source):
                errors.append(f"sequence_smokes.{model_name} {label} hash mismatch")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _current_git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit if GIT_COMMIT_RE.fullmatch(commit) else None


def _git_rev_has_tag(root: Path, git_commit: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "tag", "--points-at", git_commit],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _source_archive_git_commit(root: Path) -> str | None:
    path = root / ".source-git-commit"
    if not path.exists():
        return None
    value = "".join(path.read_text().split())
    if not value or "$Format" in value:
        return None
    return value if GIT_COMMIT_RE.fullmatch(value) else None


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
