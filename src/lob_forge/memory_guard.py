from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_CSV_LOAD_MULTIPLIER = 8.0
DEFAULT_FEATURE_BUILD_MULTIPLIER = 12.0
PROCESS_MEMORY_LIMIT_ENV = "LOB_FORGE_MAX_PROCESS_MEMORY_GB"


def physical_memory_gb() -> float | None:
    """Return total physical RAM in GB when the platform exposes it."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return pages * page_size / 1_000_000_000
    except (AttributeError, OSError, ValueError):
        pass

    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2)
        bytes_value = int(output.strip())
        if bytes_value > 0:
            return bytes_value / 1_000_000_000
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def estimate_csv_load_memory_gb(
    path: Path | str,
    *,
    multiplier: float = DEFAULT_CSV_LOAD_MULTIPLIER,
) -> float:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    size_bytes = Path(path).stat().st_size
    return size_bytes * multiplier / 1_000_000_000


def assert_csv_load_budget(
    path: Path | str,
    *,
    max_memory_gb: float,
    multiplier: float = DEFAULT_CSV_LOAD_MULTIPLIER,
) -> None:
    if max_memory_gb <= 0:
        return
    estimate_gb = estimate_csv_load_memory_gb(path, multiplier=multiplier)
    if estimate_gb > max_memory_gb:
        raise ValueError(
            "estimated CSV load memory exceeds budget: "
            f"path={Path(path)} file_size_gb={Path(path).stat().st_size / 1_000_000_000:.2f} "
            f"multiplier={multiplier:.2f} estimated_gb={estimate_gb:.2f} "
            f"max_memory_gb={max_memory_gb:.2f}. "
            "Use a smaller profile, increase --max-load-memory-gb on a high-RAM machine, "
            "or set it to 0 to disable this guard."
        )


def estimate_feature_build_memory_gb(
    paths: Iterable[Path | str | None],
    *,
    multiplier: float = DEFAULT_FEATURE_BUILD_MULTIPLIER,
) -> float:
    """Conservative preflight estimate for the daily feature-builder inputs."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    size_bytes = 0
    for path_value in paths:
        if path_value is None:
            continue
        size_bytes += Path(path_value).stat().st_size
    return size_bytes * multiplier / 1_000_000_000


def assert_feature_build_budget(
    paths: Iterable[Path | str | None],
    *,
    max_memory_gb: float,
    multiplier: float = DEFAULT_FEATURE_BUILD_MULTIPLIER,
) -> None:
    if max_memory_gb <= 0:
        return
    materialized_paths = [Path(path) for path in paths if path is not None]
    estimate_gb = estimate_feature_build_memory_gb(materialized_paths, multiplier=multiplier)
    if estimate_gb > max_memory_gb:
        file_size_gb = sum(path.stat().st_size for path in materialized_paths) / 1_000_000_000
        joined_paths = ",".join(str(path) for path in materialized_paths)
        raise ValueError(
            "estimated feature-build memory exceeds budget: "
            f"paths={joined_paths} compressed_file_size_gb={file_size_gb:.2f} "
            f"multiplier={multiplier:.2f} estimated_gb={estimate_gb:.2f} "
            f"max_memory_gb={max_memory_gb:.2f}. "
            "Use STUDY_PROFILE=laptop_tiny, lower MAX_QUOTE_BUCKETS, disable WITH_BOOK_DEPTH, "
            "or run the profile on a high-RAM cloud machine."
        )


def assert_physical_memory(
    *,
    min_memory_gb: float,
    allow_unknown: bool = True,
) -> float | None:
    ram_gb = physical_memory_gb()
    if ram_gb is None:
        if allow_unknown:
            return None
        raise ValueError("could not detect physical memory")
    if ram_gb < min_memory_gb:
        raise ValueError(f"physical RAM {ram_gb:.1f} GB is below required minimum {min_memory_gb:.1f} GB")
    return ram_gb


def process_memory_limit_gb_from_env(env_var: str = PROCESS_MEMORY_LIMIT_ENV) -> float | None:
    raw_limit = os.environ.get(env_var, "").strip()
    if not raw_limit:
        return None
    try:
        limit_gb = float(raw_limit)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a number of GB, got {raw_limit!r}") from exc
    if limit_gb <= 0:
        return None
    return limit_gb


def apply_process_memory_limit(
    max_memory_gb: float | None = None,
    *,
    env_var: str = PROCESS_MEMORY_LIMIT_ENV,
) -> float | None:
    """Apply an address-space cap so a bad local command fails before exhausting swap."""
    if max_memory_gb is None:
        max_memory_gb = process_memory_limit_gb_from_env(env_var)
    if max_memory_gb is None:
        return None
    if max_memory_gb <= 0:
        return None

    try:
        import resource
    except ImportError as exc:  # pragma: no cover - Unix platforms expose this in normal use.
        raise ValueError(f"{env_var} requested a process memory limit, but resource limits are unavailable") from exc

    requested_bytes = int(max_memory_gb * 1_000_000_000)
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
        if not hasattr(resource, limit_name):
            continue
        limit_kind = getattr(resource, limit_name)
        _, hard_limit = resource.getrlimit(limit_kind)
        limit_bytes = requested_bytes
        if hard_limit != resource.RLIM_INFINITY and hard_limit > 0 and hard_limit < limit_bytes:
            limit_bytes = hard_limit
        try:
            resource.setrlimit(limit_kind, (limit_bytes, hard_limit))
        except (OSError, ValueError):
            continue
        return limit_bytes / 1_000_000_000
    return None
