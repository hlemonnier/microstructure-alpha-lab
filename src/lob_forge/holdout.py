from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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
    purged_label_rows: int = 0


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
    if manifest.manifest_version != 1 or not manifest.split_column or not manifest.holdout_values:
        return False
    if manifest.git_commit in FORBIDDEN_GIT_SENTINELS:
        return False
    if not GIT_COMMIT_RE.fullmatch(manifest.git_commit):
        return False
    if manifest.candidate_sha256 and not _is_sha256(manifest.candidate_sha256):
        return False
    if not _is_sha256(manifest.dataset_fingerprint) or not _is_sha256(manifest.source_sha256):
        return False
    source_path = _resolve_manifest_source_path(manifest, source_root=source_root)
    if not source_path.is_file() or sha256_file(source_path) != manifest.source_sha256 or manifest.source_sha256 != manifest.dataset_fingerprint:
        return False
    with source_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if manifest.split_column not in (reader.fieldnames or []):
            return False
        present = {row.get(manifest.split_column) for row in reader}
    return set(manifest.holdout_values).issubset(present)


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
        rows = list(reader)
    return _partition_development_rows(rows, manifest)[0]


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
    if source.resolve() == output.resolve():
        raise ValueError("development output cannot overwrite the holdout source")
    assert_holdout_source_matches(source, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = reader.fieldnames or []
        if manifest.split_column not in fieldnames:
            raise ValueError(f"split column {manifest.split_column!r} missing from CSV")
        rows = list(reader)
    kept, purged = _partition_development_rows(rows, manifest)
    source_rows = len(rows)
    development_rows = len(kept)
    excluded_rows = source_rows - development_rows
    if source_rows == 0:
        raise ValueError("feature CSV contains no rows")
    if excluded_rows == 0:
        raise ValueError("holdout manifest excluded zero rows from the feature CSV")
    if development_rows == 0:
        raise ValueError("holdout manifest excluded all rows; no development data remains")
    with output.open("w", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return DevelopmentCsvResult(
        path=output,
        source_rows=source_rows,
        development_rows=development_rows,
        excluded_holdout_rows=excluded_rows,
        purged_label_rows=purged,
    )


def _partition_development_rows(
    rows: list[dict[str, str]], manifest: HoldoutManifest,
) -> tuple[list[dict[str, str]], int]:
    holdout_values = set(manifest.holdout_values)
    selected = [row for row in rows if row.get(manifest.split_column) in holdout_values]
    if not selected:
        raise ValueError("holdout manifest selected zero rows from the feature CSV")
    # Generic L2/event CSVs have no forward labels; their sequence builder must
    # protect the entire feature/label footprint separately after replay.
    has_forward_endpoint = any("future_event_time" in row for row in rows)
    intervals: list[tuple[int, int]] = []
    if has_forward_endpoint:
        for value in holdout_values:
            bucket = [row for row in selected if row.get(manifest.split_column) == value]
            if manifest.split_column == "source_date" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                day_start = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
                intervals.append((int(day_start.timestamp() * 1000), int((day_start + timedelta(days=1)).timestamp() * 1000) - 1))
            else:
                times = [_required_timestamp(row, "event_time") for row in bucket]
                intervals.append((min(times), max(times)))
    kept = []
    purged = 0
    for row in rows:
        if row.get(manifest.split_column) in holdout_values:
            continue
        if has_forward_endpoint:
            start = _required_timestamp(row, "event_time")
            end = _required_timestamp(row, "future_event_time")
            if end < start:
                raise ValueError("label endpoint precedes decision time")
            if any(start <= stop and end >= begin for begin, stop in intervals):
                purged += 1
                continue
        kept.append(row)
    return kept, purged


def _required_timestamp(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "")
    if not raw:
        raise ValueError(f"holdout label purging requires {key}")
    value = float(raw)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{key} must be a finite integral timestamp")
    return int(raw) if re.fullmatch(r"[+-]?\d+", raw) else int(value)


def write_final_holdout_result(
    *,
    manifest: HoldoutManifest,
    metrics: dict[str, object],
    output_path: Path | str,
    explicit_final_evaluation: bool,
    candidate_sha256: str | None = None,
    lock_dir: Path | str | None = None,
    source_root: Path | str | None = None,
    reservation: dict[str, object] | None = None,
    candidate_path: Path | str | None = None,
) -> Path:
    if not explicit_final_evaluation:
        raise PermissionError("final holdout evaluation requires explicit_final_evaluation=True")
    if not verify_holdout_manifest(manifest, source_root=source_root):
        raise ValueError("holdout manifest verification failed")
    if candidate_sha256 is None:
        raise ValueError("final holdout evaluation requires candidate_sha256")
    if not _is_sha256(candidate_sha256):
        raise ValueError("candidate_sha256 must be a SHA-256 digest")
    if not manifest.candidate_sha256:
        raise ValueError("final holdout manifest must pre-register candidate_sha256")
    if manifest.candidate_sha256 != candidate_sha256:
        raise ValueError("candidate_sha256 does not match holdout manifest candidate_sha256")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"immutable holdout result already exists: {path}")
    _assert_finite_json(metrics)
    candidate_file = Path(candidate_path).resolve() if candidate_path is not None else None
    if candidate_file is not None:
        candidate = json.loads(candidate_file.read_text())
        if canonical_json_sha256(candidate) != candidate_sha256:
            raise ValueError("candidate file content does not match frozen candidate hash")
        if metrics.get("candidate") != candidate or metrics.get("candidate_sha256") != candidate_sha256:
            raise ValueError("metrics must contain the exact frozen candidate and candidate_sha256")
    pre_read_reserved = reservation is not None
    if reservation is None:
        # Compatibility for low-level writers. Such results cannot pass the
        # evidence gate: only a reservation obtained before evaluation can.
        reservation = reserve_final_holdout_evaluation(
            manifest=manifest, candidate_sha256=candidate_sha256,
            explicit_final_evaluation=explicit_final_evaluation, source_root=source_root,
        )
    ledger_path = _holdout_ledger_path(manifest, source_root=source_root)
    expected = _reservation_identity(manifest, candidate_sha256, source_root=source_root)
    if any(reservation.get(key) != value for key, value in expected.items()):
        raise ValueError("holdout reservation does not match manifest, source and candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_version": 2,
        "manifest": asdict(manifest),
        "metrics": metrics,
        "final_evaluation": True,
        "source_root": str(Path(source_root or Path.cwd()).resolve()),
        "reservation": reservation,
        "pre_read_reserved": pre_read_reserved,
        "candidate_path": str(candidate_file) if candidate_file is not None else "",
        "candidate_file_sha256": sha256_file(candidate_file) if candidate_file is not None else "",
    }
    payload["candidate_sha256"] = candidate_sha256
    # lock_dir is intentionally not authoritative: changing an output or lock
    # directory must never permit another evaluation of the same observations.
    with _locked_holdout_ledger(ledger_path):
        ledger = _read_holdout_ledger(ledger_path)
        record = _reservation_record(ledger, reservation)
        if record.get("status") != "reserved":
            raise FileExistsError(f"final holdout reservation already completed: {ledger_path}")
        with path.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        record.update(status="completed", result_path=str(path.resolve()), result_sha256=sha256_file(path))
        _write_json_atomic(ledger_path, ledger)
    return path


def reserve_final_holdout_evaluation(
    *, manifest: HoldoutManifest, candidate_sha256: str,
    explicit_final_evaluation: bool, source_root: Path | str | None = None,
) -> dict[str, object]:
    """Consume the declared observations before reading targets or predicting.

    A failed/interrupted evaluation stays consumed. Row ranges, not mutable
    manifest annotations or candidate names, identify overlap on a bound data
    source. This enforces local workflow integrity, not secrecy of local files.
    """
    if not explicit_final_evaluation:
        raise PermissionError("final holdout evaluation requires explicit_final_evaluation=True")
    if not verify_holdout_manifest(manifest, source_root=source_root):
        raise ValueError("holdout manifest verification failed")
    if not _is_sha256(candidate_sha256) or candidate_sha256 != manifest.candidate_sha256:
        raise ValueError("candidate_sha256 does not match pre-registered holdout candidate")
    source = _resolve_manifest_source_path(manifest, source_root=source_root)
    semantics_errors = _built_feature_evidence_errors(source, source_root=source_root)
    if semantics_errors:
        raise ValueError("; ".join(semantics_errors))
    token = _reservation_identity(manifest, candidate_sha256, source_root=source_root)
    indices = []
    with source.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row.get(manifest.split_column) in set(manifest.holdout_values):
                indices.append(index)
    if not indices:
        raise ValueError("holdout contains no observations")
    ranges: list[list[int]] = []
    for index in indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    token.update(reservation_id=uuid.uuid4().hex, row_ranges=ranges)
    ledger_path = _holdout_ledger_path(manifest, source_root=source_root)
    with _locked_holdout_ledger(ledger_path):
        ledger = _read_holdout_ledger(ledger_path)
        for record in ledger["reservations"]:
            if _ranges_overlap(ranges, record["row_ranges"]):
                raise FileExistsError(f"final holdout data already reserved/evaluated: {ledger_path}")
        ledger["reservations"].append({**token, "status": "reserved"})
        _write_json_atomic(ledger_path, ledger)
    return token


def verify_final_holdout_result(path: Path | str) -> tuple[bool, tuple[str, ...]]:
    """Verify source, frozen candidate, pre-read reservation and result digest."""
    result_path = Path(path)
    errors = []
    try:
        payload = json.loads(result_path.read_text())
        _assert_finite_json(payload)
        manifest_data = dict(payload["manifest"])
        manifest_data["holdout_values"] = tuple(manifest_data["holdout_values"])
        manifest_data["date_range"] = tuple(manifest_data.get("date_range", ("", "")))
        manifest = HoldoutManifest(**manifest_data)
        root = payload["source_root"]
        if payload.get("result_version") != 2 or payload.get("pre_read_reserved") is not True:
            errors.append("missing pre-evaluation data reservation")
        if payload.get("final_evaluation") is not True or not verify_holdout_manifest(manifest, source_root=root):
            errors.append("source/manifest verification failed")
        errors.extend(_built_feature_evidence_errors(_resolve_manifest_source_path(manifest, source_root=root), source_root=root))
        candidate_path = Path(payload["candidate_path"])
        if not candidate_path.is_file() or sha256_file(candidate_path) != payload["candidate_file_sha256"]:
            errors.append("candidate file hash mismatch")
        else:
            candidate = json.loads(candidate_path.read_text())
            candidate_hash = canonical_json_sha256(candidate)
            if candidate_hash != payload["candidate_sha256"] or candidate_hash != manifest.candidate_sha256:
                errors.append("candidate content hash mismatch")
            if payload["metrics"].get("candidate") != candidate or payload["metrics"].get("candidate_sha256") != candidate_hash:
                errors.append("metrics candidate binding mismatch")
            for file_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),):
                if file_key in candidate and (not Path(candidate[file_key]).is_file()
                                             or sha256_file(Path(candidate[file_key])) != candidate.get(hash_key)):
                    errors.append(f"candidate {file_key} hash mismatch")
        reservation = payload["reservation"]
        expected = _reservation_identity(manifest, payload["candidate_sha256"], source_root=root)
        if any(reservation.get(key) != value for key, value in expected.items()):
            errors.append("reservation identity mismatch")
        ledger = _read_holdout_ledger(_holdout_ledger_path(manifest, source_root=root))
        record = _reservation_record(ledger, reservation)
        if record.get("status") != "completed" or record.get("result_path") != str(result_path.resolve()):
            errors.append("evaluation ledger is not completed for this result")
        if record.get("result_sha256") != sha256_file(result_path):
            errors.append("result digest does not match evaluation ledger")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"invalid final holdout evidence: {exc}")
    return not errors, tuple(errors)


def _built_feature_evidence_errors(source: Path, *, source_root: Path | str | None) -> list[str]:
    """Recognize our generated schema without imposing it on external CSVs."""
    # Local imports avoid the experiments -> holdout import cycle.
    from lob_forge.features import FEATURE_SEMANTICS_VERSION
    from lob_forge.experiments import (
        COMBINED_FEATURE_MANIFEST_VERSION, _feature_builder_source_sha256, feature_build_is_current,
    )

    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        built_schema = {"bucket_start_ms", "quote_updates_in_bucket", "quote_ofi"}.issubset(columns)
        if not built_schema and "feature_semantics_version" not in columns:
            return []
        if any(row.get("feature_semantics_version") != FEATURE_SEMANTICS_VERSION for row in reader):
            return ["generated feature source has missing or obsolete feature semantics; rebuild features"]

    def valid_daily(path: Path) -> bool:
        marker = path.with_suffix(path.suffix + ".done")
        if not marker.is_file():
            return False
        data = json.loads(marker.read_text())
        return feature_build_is_current(path, marker, build_config=data["build_config"], input_hashes=data["input_hashes"])

    try:
        if valid_daily(source):
            return []
        combined = Path(f"{source}.manifest.json")
        if not combined.is_file():
            return ["generated feature source lacks a current build marker or combined manifest"]
        payload = json.loads(combined.read_text())
        root = Path(source_root or Path.cwd())

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else root / path

        current = (
            payload.get("manifest_version") == COMBINED_FEATURE_MANIFEST_VERSION
            and payload.get("builder_source_sha256") == _feature_builder_source_sha256()
            and resolve(payload["output_path"]).resolve() == source.resolve()
            and payload.get("output_sha256") == sha256_file(source)
            and bool(payload.get("daily_features"))
            and all(sha256_file(resolve(item["path"])) == item["sha256"] and valid_daily(resolve(item["path"]))
                    for item in payload["daily_features"])
        )
        return [] if current else ["generated feature build provenance is obsolete or mismatched"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return [f"invalid generated feature build provenance: {exc}"]


def _reservation_identity(manifest: HoldoutManifest, candidate_sha256: str, *, source_root: Path | str | None) -> dict[str, object]:
    return {
        "manifest_sha256": canonical_json_sha256(asdict(manifest)),
        "candidate_sha256": candidate_sha256,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "ledger_path": str(_holdout_ledger_path(manifest, source_root=source_root).resolve()),
    }


def _holdout_ledger_path(manifest: HoldoutManifest, *, source_root: Path | str | None) -> Path:
    return _default_final_holdout_lock_dir(manifest, source_root=source_root) / f"{manifest.dataset_fingerprint}.json"


@contextmanager
def _locked_holdout_ledger(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = path.with_suffix(".guard")
    guard.mkdir()  # Atomic exclusion; a crashed writer fails closed.
    try:
        yield
    finally:
        guard.rmdir()


def _read_holdout_ledger(path: Path) -> dict:
    if not path.exists():
        return {"ledger_version": 1, "reservations": []}
    payload = json.loads(path.read_text())
    if payload.get("ledger_version") != 1 or not isinstance(payload.get("reservations"), list):
        raise ValueError("invalid holdout consumption ledger")
    return payload


def _reservation_record(ledger: dict, token: dict) -> dict:
    matches = [record for record in ledger["reservations"] if record.get("reservation_id") == token.get("reservation_id")]
    if len(matches) != 1 or any(matches[0].get(key) != value for key, value in token.items()):
        raise ValueError("holdout reservation is missing or does not match ledger")
    return matches[0]


def _ranges_overlap(left: list, right: list) -> bool:
    return any(a <= d and c <= b for a, b in left for c, d in right)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("final holdout evidence must contain finite numbers")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_json(item)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _default_final_holdout_lock_dir(manifest: HoldoutManifest, *, source_root: Path | str | None) -> Path:
    source_path = _resolve_manifest_source_path(manifest, source_root=source_root).resolve()
    # Anchor a study at its declared containing root (or Git checkout), so a
    # copy of the same dataset in another subdirectory cannot reset exposure.
    for parent in source_path.parents:
        if (parent / ".git").exists():
            return parent / ".final_holdout_locks"
    if source_root is not None:
        root = Path(source_root).resolve()
        try:
            source_path.relative_to(root)
        except ValueError:
            pass  # An absolute external source is not owned by this root.
        else:
            return root / ".final_holdout_locks"
    return source_path.parent / ".final_holdout_locks"


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
