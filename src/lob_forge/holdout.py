from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


FORBIDDEN_GIT_SENTINELS = {"", "no" + "-git-commit"}
GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$")


@dataclass(frozen=True)
class HoldoutManifest:
    manifest_version: int
    source_path: str
    source_sha256: str
    split_column: str
    holdout_values: tuple[str, ...]
    created_at_utc: str
    dataset_fingerprint: str = ""
    date_range: tuple[str, str] = ("", "")
    feature_version: str = "feature_v1"
    target_version: str = "target_v1"
    git_commit: str = ""
    candidate_sha256: str = ""
    notes: str = ""


@dataclass(frozen=True)
class DevelopmentCsvResult:
    path: Path
    source_rows: int
    development_rows: int
    excluded_holdout_rows: int


def build_holdout_manifest(
    source_path: Path | str,
    *,
    split_column: str,
    holdout_values: list[str],
    created_at_utc: str,
    feature_version: str = "feature_v1",
    target_version: str = "target_v1",
    git_commit: str | None = None,
    candidate_sha256: str = "",
    notes: str = "",
    source_root: Path | str | None = None,
) -> HoldoutManifest:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not holdout_values:
        raise ValueError("holdout_values cannot be empty")
    source_sha256 = sha256_file(path)
    return HoldoutManifest(
        manifest_version=1,
        source_path=_manifest_source_path(path, source_root=source_root),
        source_sha256=source_sha256,
        split_column=split_column,
        holdout_values=tuple(sorted(set(holdout_values))),
        created_at_utc=created_at_utc,
        dataset_fingerprint=source_sha256,
        date_range=_date_range(path, split_column),
        feature_version=feature_version,
        target_version=target_version,
        git_commit=git_commit if git_commit is not None else current_git_commit(),
        candidate_sha256=candidate_sha256,
        notes=notes,
    )


def write_holdout_manifest(manifest: HoldoutManifest, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_holdout_manifest(path)
        if existing != manifest:
            raise FileExistsError(f"holdout manifest already exists with different content: {path}")
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return path


def read_holdout_manifest(path: Path | str) -> HoldoutManifest:
    payload = json.loads(Path(path).read_text())
    payload["holdout_values"] = tuple(payload["holdout_values"])
    payload["date_range"] = tuple(payload.get("date_range", ("", "")))
    payload.setdefault("candidate_sha256", "")
    return HoldoutManifest(**payload)


def verify_holdout_manifest(manifest: HoldoutManifest, *, source_root: Path | str | None = None) -> bool:
    if manifest.git_commit in FORBIDDEN_GIT_SENTINELS:
        return False
    if not GIT_COMMIT_RE.fullmatch(manifest.git_commit):
        return False
    if manifest.candidate_sha256 and not _is_sha256(manifest.candidate_sha256):
        return False
    if not manifest.dataset_fingerprint:
        return False
    source_path = _resolve_manifest_source_path(manifest, source_root=source_root)
    return source_path.exists() and sha256_file(source_path) == manifest.source_sha256 == manifest.dataset_fingerprint


def verify_holdout_manifest_file(path: Path | str) -> bool:
    manifest_path = Path(path)
    manifest = read_holdout_manifest(manifest_path)
    for source_root in _candidate_manifest_source_roots(manifest_path):
        if verify_holdout_manifest(manifest, source_root=source_root):
            return True
    return False


def holdout_manifest_source_root(path: Path | str) -> Path | None:
    manifest_path = Path(path)
    manifest = read_holdout_manifest(manifest_path)
    for source_root in _candidate_manifest_source_roots(manifest_path):
        if verify_holdout_manifest(manifest, source_root=source_root):
            return source_root
    return None


def assert_holdout_source_matches(feature_csv: Path | str, manifest: HoldoutManifest) -> None:
    source_hash = sha256_file(Path(feature_csv))
    if source_hash != manifest.source_sha256 or source_hash != manifest.dataset_fingerprint:
        raise ValueError("feature CSV content hash does not match holdout manifest source fingerprint")


def assert_no_holdout_leakage(
    feature_csv: Path | str,
    manifest: HoldoutManifest,
    *,
    training_values: list[str],
) -> None:
    overlap = set(training_values) & set(manifest.holdout_values)
    if overlap:
        raise ValueError(f"training split overlaps immutable holdout values: {sorted(overlap)}")
    with Path(feature_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if manifest.split_column not in (reader.fieldnames or []):
            raise ValueError(f"split column {manifest.split_column!r} missing from CSV")
        leaked_rows = [
            index
            for index, row in enumerate(reader, start=1)
            if row.get(manifest.split_column) in set(manifest.holdout_values) and row.get("_split") == "development"
        ]
        if leaked_rows:
            raise ValueError(f"development rows include holdout values at rows: {leaked_rows[:5]}")


def read_development_rows(feature_csv: Path | str, manifest: HoldoutManifest) -> list[dict[str, str]]:
    assert_holdout_source_matches(feature_csv, manifest)
    assert_no_holdout_leakage(feature_csv, manifest, training_values=[])
    with Path(feature_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get(manifest.split_column) not in set(manifest.holdout_values)]


def read_holdout_rows(feature_csv: Path | str, manifest: HoldoutManifest) -> list[dict[str, str]]:
    assert_holdout_source_matches(feature_csv, manifest)
    holdout_values = set(manifest.holdout_values)
    with Path(feature_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if manifest.split_column not in (reader.fieldnames or []):
            raise ValueError(f"split column {manifest.split_column!r} missing from CSV")
        rows = [row for row in reader if row.get(manifest.split_column) in holdout_values]
    if not rows:
        raise ValueError("holdout manifest selected zero rows from the feature CSV")
    return rows


def write_development_csv(
    feature_csv: Path | str,
    manifest: HoldoutManifest,
    output_path: Path | str,
) -> DevelopmentCsvResult:
    source = Path(feature_csv)
    output = Path(output_path)
    assert_holdout_source_matches(source, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    holdout_values = set(manifest.holdout_values)
    source_rows = 0
    development_rows = 0
    excluded_rows = 0
    with source.open(newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = reader.fieldnames or []
        if manifest.split_column not in fieldnames:
            raise ValueError(f"split column {manifest.split_column!r} missing from CSV")
        with output.open("w", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                source_rows += 1
                if row.get(manifest.split_column) in holdout_values:
                    excluded_rows += 1
                    continue
                writer.writerow(row)
                development_rows += 1
    if source_rows == 0:
        raise ValueError("feature CSV contains no rows")
    if excluded_rows == 0:
        raise ValueError("holdout manifest excluded zero rows from the feature CSV")
    if development_rows == 0:
        raise ValueError("holdout manifest excluded all rows; no development data remains")
    return DevelopmentCsvResult(
        path=output,
        source_rows=source_rows,
        development_rows=development_rows,
        excluded_holdout_rows=excluded_rows,
    )


def write_final_holdout_result(
    *,
    manifest: HoldoutManifest,
    metrics: dict[str, object],
    output_path: Path | str,
    explicit_final_evaluation: bool,
    candidate_sha256: str | None = None,
    lock_dir: Path | str | None = None,
    source_root: Path | str | None = None,
) -> Path:
    if not explicit_final_evaluation:
        raise PermissionError("final holdout evaluation requires explicit_final_evaluation=True")
    if not verify_holdout_manifest(manifest, source_root=source_root):
        raise ValueError("holdout manifest verification failed")
    if candidate_sha256 is None:
        raise ValueError("final holdout evaluation requires candidate_sha256")
    if not _is_sha256(candidate_sha256):
        raise ValueError("candidate_sha256 must be a SHA-256 digest")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"immutable holdout result already exists: {path}")
    manifest_sha256 = canonical_json_sha256(asdict(manifest))
    lock_root = Path(lock_dir) if lock_dir is not None else path.parent / ".holdout_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{manifest_sha256}_{candidate_sha256}.lock"
    if lock_path.exists():
        raise FileExistsError(f"final holdout already evaluated for manifest/candidate: {lock_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": asdict(manifest),
        "metrics": metrics,
        "final_evaluation": True,
    }
    payload["candidate_sha256"] = candidate_sha256
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lock_path.write_text(
        json.dumps(
            {
                "candidate_sha256": candidate_sha256,
                "manifest_sha256": manifest_sha256,
                "result_path": str(path),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot build official holdout manifest without a Git commit") from exc
    commit = result.stdout.strip()
    if not GIT_COMMIT_RE.fullmatch(commit):
        raise RuntimeError("cannot build official holdout manifest without a Git commit")
    return commit


def _manifest_source_path(path: Path, *, source_root: Path | str | None) -> str:
    resolved_path = path.resolve()
    root = Path.cwd() if source_root is None else Path(source_root)
    try:
        return resolved_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved_path)


def _resolve_manifest_source_path(manifest: HoldoutManifest, *, source_root: Path | str | None) -> Path:
    source_path = Path(manifest.source_path)
    if source_path.is_absolute():
        return source_path
    root = Path.cwd() if source_root is None else Path(source_root)
    return root / source_path


def _candidate_manifest_source_roots(manifest_path: Path) -> Iterator[Path]:
    seen: set[Path] = set()
    candidates = [Path.cwd().resolve(), manifest_path.parent.resolve()]
    candidates.extend(manifest_path.parent.resolve().parents)
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _date_range(path: Path, split_column: str) -> tuple[str, str]:
    values: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if split_column not in (reader.fieldnames or []):
            raise ValueError(f"split column {split_column!r} missing from CSV")
        for row in reader:
            value = row.get(split_column)
            if value:
                values.append(value)
    if not values:
        return "", ""
    return min(values), max(values)
