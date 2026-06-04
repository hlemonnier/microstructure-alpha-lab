from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HoldoutManifest:
    manifest_version: int
    source_path: str
    source_sha256: str
    split_column: str
    holdout_values: tuple[str, ...]
    created_at_utc: str
    notes: str = ""


def build_holdout_manifest(
    source_path: Path | str,
    *,
    split_column: str,
    holdout_values: list[str],
    created_at_utc: str,
    notes: str = "",
) -> HoldoutManifest:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not holdout_values:
        raise ValueError("holdout_values cannot be empty")
    return HoldoutManifest(
        manifest_version=1,
        source_path=str(path),
        source_sha256=sha256_file(path),
        split_column=split_column,
        holdout_values=tuple(sorted(set(holdout_values))),
        created_at_utc=created_at_utc,
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
    return HoldoutManifest(**payload)


def verify_holdout_manifest(manifest: HoldoutManifest) -> bool:
    return sha256_file(Path(manifest.source_path)) == manifest.source_sha256


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
