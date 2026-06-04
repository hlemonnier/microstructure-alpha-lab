from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator


S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PUBLIC_BASE_URL = "https://data.binance.vision"
VALID_FREQUENCIES = {"daily", "monthly"}
KLINE_DATASETS = {"klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"}


@dataclass(frozen=True)
class VisionObject:
    key: str
    size: int
    last_modified: str
    etag: str

    @property
    def url(self) -> str:
        return url_for_key(self.key)


def url_for_key(key: str) -> str:
    return f"{PUBLIC_BASE_URL}/{key.lstrip('/')}"


def archive_key(
    *,
    market: str,
    frequency: str,
    dataset: str,
    symbol: str,
    date_value: str,
    interval: str | None = None,
) -> str:
    """Build a Binance Vision archive key for common market datasets."""
    if frequency not in VALID_FREQUENCIES:
        raise ValueError(f"frequency must be one of {sorted(VALID_FREQUENCIES)}")

    market = market.strip("/")
    symbol = symbol.upper()

    if dataset in KLINE_DATASETS:
        if not interval:
            raise ValueError(f"dataset {dataset} requires --interval")
        return (
            f"data/{market}/{frequency}/{dataset}/{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_value}.zip"
        )

    return f"data/{market}/{frequency}/{dataset}/{symbol}/{symbol}-{dataset}-{date_value}.zip"


def dataset_prefix(
    *,
    market: str,
    frequency: str,
    dataset: str,
    symbol: str | None = None,
    interval: str | None = None,
) -> str:
    market = market.strip("/")
    prefix = f"data/{market}/{frequency}/{dataset}/"
    if symbol:
        prefix += f"{symbol.upper()}/"
    if interval:
        prefix += f"{interval}/"
    return prefix


def list_objects(prefix: str, limit: int | None = None) -> list[VisionObject]:
    """List Binance Vision S3 objects under a prefix."""
    objects: list[VisionObject] = []
    marker: str | None = None

    while True:
        params = {"prefix": prefix, "max-keys": "1000"}
        if marker:
            params["marker"] = marker
        url = f"{S3_LIST_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_body = response.read()

        root = ET.fromstring(xml_body)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        contents = root.findall("s3:Contents", ns)
        for item in contents:
            key = _text(item, "Key", ns)
            if key.endswith(".CHECKSUM"):
                continue
            objects.append(
                VisionObject(
                    key=key,
                    size=int(_text(item, "Size", ns)),
                    last_modified=_text(item, "LastModified", ns),
                    etag=_text(item, "ETag", ns).strip('"'),
                )
            )
            if limit is not None and len(objects) >= limit:
                return objects

        truncated = _text(root, "IsTruncated", ns).lower() == "true"
        if not truncated:
            return objects

        next_marker = root.findtext("s3:NextMarker", default="", namespaces=ns)
        if next_marker:
            marker = next_marker
        elif contents:
            marker = _text(contents[-1], "Key", ns)
        else:
            return objects


def download_key(
    key: str,
    root: Path | str = Path("data/raw"),
    *,
    overwrite: bool = False,
    verify_checksum: bool = True,
    checksum_retries: int = 3,
) -> Path:
    """Download a Binance Vision object to root/key and optionally verify SHA-256."""
    root_path = Path(root)
    destination = root_path / key
    destination.parent.mkdir(parents=True, exist_ok=True)

    attempts = max(1, checksum_retries)
    _log_download(f"archive_prepare key={key} verify_checksum={int(verify_checksum)}")
    expected = fetch_checksum(key) if verify_checksum else None
    for attempt in range(1, attempts + 1):
        if not overwrite and destination.exists() and not verify_checksum and _looks_like_bad_zip(destination):
            _log_download(f"archive_existing_bad_zip_removed key={key} path={destination}")
            destination.unlink()
        if overwrite or not destination.exists():
            _log_download(f"archive_download_start attempt={attempt}/{attempts} key={key}")
            _download_url(url_for_key(key), destination)
            _log_download(f"archive_download_done attempt={attempt}/{attempts} key={key} bytes={destination.stat().st_size}")

        if not expected:
            return destination

        _log_download(f"archive_checksum_start attempt={attempt}/{attempts} key={key}")
        actual = sha256_file(destination)
        if actual.lower() == expected.lower():
            _log_download(f"archive_checksum_ok attempt={attempt}/{attempts} key={key}")
            return destination

        if destination.exists():
            destination.unlink()
        if attempt == attempts:
            raise ValueError(
                f"Checksum mismatch for {destination}: expected {expected}, got {actual}"
            )

    return destination


def download_archive(
    *,
    market: str,
    frequency: str,
    dataset: str,
    symbol: str,
    date_value: str,
    interval: str | None = None,
    root: Path | str = Path("data/raw"),
    overwrite: bool = False,
    verify_checksum: bool = True,
) -> Path:
    key = archive_key(
        market=market,
        frequency=frequency,
        dataset=dataset,
        symbol=symbol,
        date_value=date_value,
        interval=interval,
    )
    return download_key(
        key,
        root=root,
        overwrite=overwrite,
        verify_checksum=verify_checksum,
    )


def fetch_checksum(key: str, *, retries: int = 3) -> str | None:
    checksum_url = url_for_key(f"{key}.CHECKSUM")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _log_download(f"checksum_fetch_start attempt={attempt}/{retries} key={key}")
            with urllib.request.urlopen(checksum_url, timeout=30) as response:
                body = response.read().decode("utf-8").strip()
            _log_download(f"checksum_fetch_done attempt={attempt}/{retries} key={key}")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _log_download(f"checksum_missing key={key}")
                return None
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionResetError, OSError) as exc:
            last_error = exc
            _log_download(f"checksum_fetch_retry attempt={attempt}/{retries} key={key} error={exc!r}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    else:
        assert last_error is not None
        raise last_error
    if not body:
        return None
    return body.split()[0]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dates(start: str, end: str) -> Iterator[str]:
    current = _parse_date(start)
    stop = _parse_date(end)
    if stop < current:
        raise ValueError("end date must be >= start date")
    while current <= stop:
        yield current.isoformat()
        current += timedelta(days=1)


def infer_date_from_key(key: str) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{4}-\d{2})\.zip$", key)
    return match.group(1) if match else None


def _download_url(url: str, destination: Path, *, retries: int = 3) -> None:
    if os.environ.get("LOB_FORGE_DOWNLOAD_BACKEND", "urllib") == "curl":
        _download_url_curl(url, destination, retries=retries)
        return

    last_error: Exception | None = None
    temp_destination = destination.with_name(destination.name + ".download")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with temp_destination.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
            temp_destination.replace(destination)
            return
        except urllib.error.HTTPError:
            _remove_if_exists(temp_destination)
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionResetError, OSError) as exc:
            last_error = exc
            _log_download(f"archive_download_retry attempt={attempt}/{retries} url={url} error={exc!r}")
            _remove_if_exists(temp_destination)
            if attempt == retries:
                break
            time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def _download_url_curl(url: str, destination: Path, *, retries: int = 3) -> None:
    temp_destination = destination.with_name(destination.name + ".download")
    _remove_if_exists(temp_destination)
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--show-error",
        "--silent",
        "--connect-timeout",
        os.environ.get("LOB_FORGE_CURL_CONNECT_TIMEOUT", "30"),
        "--speed-limit",
        os.environ.get("LOB_FORGE_CURL_SPEED_LIMIT", "1024"),
        "--speed-time",
        os.environ.get("LOB_FORGE_CURL_SPEED_TIME", "30"),
        "--retry",
        str(retries),
        "--retry-delay",
        "2",
        "--output",
        str(temp_destination),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        temp_destination.replace(destination)
    except (OSError, subprocess.CalledProcessError):
        _remove_if_exists(temp_destination)
        raise


def _text(element: ET.Element, tag: str, ns: dict[str, str]) -> str:
    value = element.findtext(f"s3:{tag}", namespaces=ns)
    if value is None:
        raise ValueError(f"Missing S3 XML tag: {tag}")
    return value


def _log_download(message: str) -> None:
    if os.environ.get("LOB_FORGE_DOWNLOAD_VERBOSE", "0") == "1":
        print(message, file=sys.stderr, flush=True)


def _looks_like_bad_zip(path: Path) -> bool:
    if path.suffix.lower() != ".zip":
        return False
    return not zipfile.is_zipfile(path)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
