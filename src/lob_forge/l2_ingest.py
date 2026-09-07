from __future__ import annotations

import csv
import json
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from lob_forge.binance_vision import iter_dates, sha256_file
from lob_forge.data_sources import (
    L2FileValidation,
    NormalizedL2Row,
    get_source,
    iter_bybit_l2_csv,
    iter_bybit_orderbook_data_zip,
    iter_okx_l2_csv,
    validate_bybit_orderbook_data_zip,
    validate_l2_csv,
)
from lob_forge.l2_replay import AtomicOrderBookReplayer, iter_l2_events, l2_event_key
from lob_forge.l2_storage import (
    normalized_l2_csv_path,
    normalized_l2_parquet_path,
    write_normalized_l2_csv,
    write_normalized_l2_parquet,
)


SUPPORTED_HISTORICAL_L2_SOURCES = {"okx", "bybit"}
OKX_DOWNLOAD_LINK_ENDPOINT = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"
OKX_ORDER_BOOK_DEPTH_MODULES = {"400": "4", "order_book_400": "4", "5000": "5", "order_book_5000": "5"}
BYBIT_LIST_FILES_ENDPOINT = "https://api2.bybit.com/quote/public/support/download/list-files"


@dataclass(frozen=True)
class HistoricalL2ManifestEntry:
    source_id: str
    symbol: str
    session_date: str
    dataset: str
    market: str
    source_page_url: str
    direct_url: str = ""
    local_path: str = ""
    status: str = "url_required"
    sha256: str = ""
    bytes: int = 0
    rows_checked: int = 0
    normalized_path: str = ""
    notes: str = ""


@dataclass(frozen=True)
class HistoricalL2ImportResult:
    source_id: str
    symbol: str
    session_date: str
    input_path: Path
    output_path: Path
    storage_format: str
    rows_written: int
    validation: L2FileValidation


def build_historical_l2_manifest(
    *,
    source_id: str,
    symbols: Iterable[str],
    start: str,
    end: str,
    dataset: str = "order_book_l2",
    market: str = "swap",
) -> list[HistoricalL2ManifestEntry]:
    source = _require_supported_source(source_id)
    page_url = source.canonical_urls[0]
    entries: list[HistoricalL2ManifestEntry] = []
    for symbol in symbols:
        normalized_symbol = symbol.strip()
        if not normalized_symbol:
            continue
        for session_date in iter_dates(start, end):
            entries.append(
                HistoricalL2ManifestEntry(
                    source_id=source.source_id,
                    symbol=normalized_symbol,
                    session_date=session_date,
                    dataset=dataset,
                    market=market,
                    source_page_url=page_url,
                    notes="Fill direct_url after selecting the exact exchange-provided historical file.",
                )
            )
    if not entries:
        raise ValueError("manifest needs at least one symbol/date entry")
    return entries


def write_historical_l2_manifest(entries: Iterable[HistoricalL2ManifestEntry], path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(HistoricalL2ManifestEntry.__dataclass_fields__)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))
    return output_path


def read_historical_l2_manifest(path: Path | str) -> list[HistoricalL2ManifestEntry]:
    with Path(path).open(newline="") as handle:
        return [_historical_l2_manifest_entry_from_row(row) for row in csv.DictReader(handle)]


def _historical_l2_manifest_entry_from_row(row: Mapping[str, str]) -> HistoricalL2ManifestEntry:
    return HistoricalL2ManifestEntry(
        source_id=row.get("source_id", ""),
        symbol=row.get("symbol", ""),
        session_date=row.get("session_date", ""),
        dataset=row.get("dataset", ""),
        market=row.get("market", ""),
        source_page_url=row.get("source_page_url", ""),
        direct_url=row.get("direct_url", ""),
        local_path=row.get("local_path", ""),
        status=row.get("status", "url_required"),
        sha256=row.get("sha256", ""),
        bytes=int(row.get("bytes") or 0),
        rows_checked=int(row.get("rows_checked") or 0),
        normalized_path=row.get("normalized_path", ""),
        notes=row.get("notes", ""),
    )


def download_historical_l2_manifest(
    manifest_path: Path | str,
    *,
    raw_root: Path | str,
    overwrite: bool = False,
) -> list[HistoricalL2ManifestEntry]:
    entries = read_historical_l2_manifest(manifest_path)
    updated: list[HistoricalL2ManifestEntry] = []
    for entry in entries:
        _require_supported_source(entry.source_id)
        if not entry.direct_url:
            updated.append(entry)
            continue
        local_path = Path(entry.local_path) if entry.local_path else _raw_manifest_path(raw_root, entry)
        if overwrite or not local_path.exists():
            _download_or_copy(entry.direct_url, local_path)
        updated.append(
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "local_path": str(local_path),
                    "status": "downloaded",
                    "sha256": sha256_file(local_path),
                    "bytes": local_path.stat().st_size,
                }
            )
        )
    write_historical_l2_manifest(updated, manifest_path)
    return updated


def resolve_okx_historical_l2_manifest(
    manifest_path: Path | str,
    *,
    depth: str = "400",
    overwrite: bool = False,
    throttle_seconds: float = 1.0,
    timeout_seconds: float = 30.0,
    fetcher: Callable[[HistoricalL2ManifestEntry, str, float], dict] | None = None,
) -> list[HistoricalL2ManifestEntry]:
    """Resolve OKX historical-data manifest rows through OKX's public download-link endpoint.

    The public page can return no details for a valid request, and it rate-limits
    quickly. This function records those states in the manifest instead of
    pretending that a missing URL is a completed acquisition.
    """
    module = _okx_order_book_module(depth)
    entries = read_historical_l2_manifest(manifest_path)
    updated: list[HistoricalL2ManifestEntry] = []
    fetch_download_details = fetcher or fetch_okx_historical_download_details

    for entry in entries:
        if entry.source_id != "okx":
            updated.append(entry)
            continue
        if entry.direct_url and not overwrite:
            updated.append(entry)
            continue

        try:
            data = fetch_download_details(entry, module, timeout_seconds)
        except Exception as exc:
            updated.append(
                HistoricalL2ManifestEntry(
                    **{
                        **asdict(entry),
                        "status": "resolve_failed",
                        "notes": _append_note(entry.notes, f"OKX download-link lookup failed: {exc!r}"),
                    }
                )
            )
        else:
            resolved_entries = _entries_from_okx_download_data(entry, data, depth=depth)
            updated.extend(resolved_entries)

        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    write_historical_l2_manifest(updated, manifest_path)
    return updated


def resolve_bybit_historical_l2_manifest(
    manifest_path: Path | str,
    *,
    overwrite: bool = False,
    throttle_seconds: float = 1.0,
    timeout_seconds: float = 30.0,
    fetcher: Callable[[HistoricalL2ManifestEntry, float], dict] | None = None,
) -> list[HistoricalL2ManifestEntry]:
    """Resolve Bybit historical orderBook manifest rows through Bybit's public history-data API."""
    entries = read_historical_l2_manifest(manifest_path)
    updated: list[HistoricalL2ManifestEntry] = []
    fetch_download_details = fetcher or fetch_bybit_historical_download_details

    for entry in entries:
        if entry.source_id != "bybit":
            updated.append(entry)
            continue
        if entry.direct_url and not overwrite:
            updated.append(entry)
            continue

        try:
            data = fetch_download_details(entry, timeout_seconds)
        except Exception as exc:
            updated.append(
                HistoricalL2ManifestEntry(
                    **{
                        **asdict(entry),
                        "status": "resolve_failed",
                        "notes": _append_note(entry.notes, f"Bybit list-files lookup failed: {exc!r}"),
                    }
                )
            )
        else:
            updated.extend(_entries_from_bybit_download_data(entry, data))

        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    write_historical_l2_manifest(updated, manifest_path)
    return updated


def fetch_okx_historical_download_details(
    entry: HistoricalL2ManifestEntry,
    module: str,
    timeout_seconds: float = 30.0,
) -> dict:
    payload = build_okx_download_link_payload(entry, module=module)
    request = urllib.request.Request(
        OKX_DOWNLOAD_LINK_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://www.okx.com",
            "Referer": "https://www.okx.com/en-gb/historical-data",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    if str(body.get("code")) != "0":
        raise ValueError(f"OKX download-link returned code={body.get('code')} msg={body.get('msg')}")
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("OKX download-link response missing data object")
    return data


def fetch_bybit_historical_download_details(
    entry: HistoricalL2ManifestEntry,
    timeout_seconds: float = 30.0,
) -> dict:
    query = build_bybit_list_files_query(entry)
    url = BYBIT_LIST_FILES_ENDPOINT + "?" + urllib.parse.urlencode(query, doseq=True)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Referer": "https://www.bybit.com/derivatives/en/history-data",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    if int(body.get("ret_code", -1)) != 0:
        raise ValueError(f"Bybit list-files returned ret_code={body.get('ret_code')} msg={body.get('ret_msg')}")
    data = body.get("result")
    if not isinstance(data, dict):
        raise ValueError("Bybit list-files response missing result object")
    return data


def build_okx_download_link_payload(entry: HistoricalL2ManifestEntry, *, module: str) -> dict:
    inst_type = _okx_inst_type(entry.market)
    if inst_type == "SPOT":
        inst_query = {"instIdList": [entry.symbol]}
    else:
        inst_query = {"instFamilyList": [_okx_inst_family(entry.symbol)]}
    begin_ms, end_ms = _utc_day_ms_range(entry.session_date)
    return {
        "module": module,
        "instType": inst_type,
        "instQueryParam": inst_query,
        "dateQuery": {
            "dateAggrType": "daily",
            "begin": str(begin_ms),
            "end": str(end_ms),
        },
    }


def build_bybit_list_files_query(entry: HistoricalL2ManifestEntry) -> dict:
    return {
        "bizType": _bybit_biz_type(entry.market),
        "productId": _bybit_product_id(entry.dataset),
        "symbols": _bybit_symbol(entry.symbol),
        "interval": "daily",
        "startDay": entry.session_date,
        "endDay": entry.session_date,
    }


def import_historical_l2_file(
    path: Path | str,
    *,
    source_id: str,
    symbol: str,
    session_date: str,
    output_root: Path | str,
    storage_format: str = "csv",
    max_validation_rows: int = 1000,
    max_import_rows: int | None = None,
    require_sequence: bool | None = None,
) -> HistoricalL2ImportResult:
    _require_supported_source(source_id)
    if storage_format not in {"csv", "parquet"}:
        raise ValueError("storage_format must be csv or parquet")
    if max_import_rows is not None and max_import_rows <= 0:
        raise ValueError("max_import_rows must be positive when provided")

    input_path = Path(path)
    validation = _validate_source_file(
        source_id=source_id,
        path=input_path,
        max_rows=max_validation_rows,
        require_sequence=require_sequence,
    )
    if not validation.ok:
        missing = ", ".join(validation.missing_columns)
        row_errors = "; ".join(
            f"row {row.row_index}: " + ", ".join(f"{issue.field}={issue.message}" for issue in row.errors)
            for row in validation.row_errors[:5]
        )
        raise ValueError(f"historical L2 validation failed missing=[{missing}] errors=[{row_errors}]")

    rows = _iter_source_rows(source_id, input_path, symbol=symbol)
    if max_import_rows is not None:
        rows = _limit_rows(rows, max_import_rows)
    # Validate every consumed complete message, not only the schema sample above.
    def validated_rows():
        replayer = AtomicOrderBookReplayer()
        for event in iter_l2_events(rows):
            if any(row.symbol != symbol or row.venue != source_id for row in event):
                raise ValueError("historical L2 instrument does not match requested venue/symbol")
            replayer.apply_event(event)
            yield from event

    counted_rows = _counting_rows(validated_rows())
    if storage_format == "csv":
        output_path = normalized_l2_csv_path(output_root, venue=source_id, symbol=symbol, session_date=session_date)
        temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
        try:
            write_normalized_l2_csv(counted_rows, temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        output_path = normalized_l2_parquet_path(output_root, venue=source_id, symbol=symbol, session_date=session_date)
        temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
        try:
            write_normalized_l2_parquet(counted_rows, temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return HistoricalL2ImportResult(
        source_id=source_id,
        symbol=symbol,
        session_date=session_date,
        input_path=input_path,
        output_path=output_path,
        storage_format=storage_format,
        rows_written=getattr(counted_rows, "count", 0),
        validation=replace(validation, rows_checked=getattr(counted_rows, "count", 0)),
    )


def import_historical_l2_manifest(
    manifest_path: Path | str,
    *,
    output_root: Path | str,
    storage_format: str = "csv",
    max_validation_rows: int = 1000,
    max_import_rows: int | None = None,
    require_sequence: bool | None = None,
) -> list[HistoricalL2ImportResult]:
    entries = read_historical_l2_manifest(manifest_path)
    results: list[HistoricalL2ImportResult] = []
    updated: list[HistoricalL2ManifestEntry] = []
    for entry in entries:
        if not entry.local_path:
            updated.append(entry)
            continue
        result = import_historical_l2_file(
            entry.local_path,
            source_id=entry.source_id,
            symbol=entry.symbol,
            session_date=entry.session_date,
            output_root=output_root,
            storage_format=storage_format,
            max_validation_rows=max_validation_rows,
            max_import_rows=max_import_rows,
            require_sequence=require_sequence,
        )
        results.append(result)
        updated.append(
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "status": "normalized",
                    "rows_checked": result.validation.rows_checked,
                    "normalized_path": str(result.output_path),
                    "notes": _append_note(entry.notes, _normalized_note(result, max_import_rows=max_import_rows)),
                }
            )
        )
    write_historical_l2_manifest(updated, manifest_path)
    return results


class _counting_rows:
    def __init__(self, rows: Iterable[NormalizedL2Row]) -> None:
        self.rows = iter(rows)
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self) -> NormalizedL2Row:
        row = next(self.rows)
        self.count += 1
        return row


def _limit_rows(rows: Iterable[NormalizedL2Row], max_rows: int) -> Iterable[NormalizedL2Row]:
    for event in iter_l2_events(rows, max_rows=max_rows):
        yield from event


def _normalized_event_key(row: NormalizedL2Row) -> tuple[str, int | None, int | None, int | None]:
    return l2_event_key(row)


def _iter_source_rows(source_id: str, path: Path, *, symbol: str) -> Iterable[NormalizedL2Row]:
    if source_id == "okx":
        return iter_okx_l2_csv(path, default_symbol=symbol)
    if source_id == "bybit":
        if _is_bybit_orderbook_data_zip(path):
            return iter_bybit_orderbook_data_zip(path, default_symbol=symbol)
        return iter_bybit_l2_csv(path, default_symbol=symbol)
    raise ValueError(f"unsupported historical L2 source: {source_id}")


def _validate_source_file(
    *,
    source_id: str,
    path: Path,
    max_rows: int,
    require_sequence: bool | None,
) -> L2FileValidation:
    if source_id == "bybit" and _is_bybit_orderbook_data_zip(path):
        return validate_bybit_orderbook_data_zip(
            path,
            max_rows=max_rows,
            require_sequence=True if require_sequence is None else require_sequence,
        )
    return validate_l2_csv(
        path,
        source_id=source_id,
        max_rows=max_rows,
        require_sequence=require_sequence,
    )


def _require_supported_source(source_id: str):
    source = get_source(source_id)
    if source.source_id not in SUPPORTED_HISTORICAL_L2_SOURCES:
        raise ValueError(f"historical L2 import supports {sorted(SUPPORTED_HISTORICAL_L2_SOURCES)}")
    return source


def _entries_from_okx_download_data(
    entry: HistoricalL2ManifestEntry,
    data: dict,
    *,
    depth: str,
) -> list[HistoricalL2ManifestEntry]:
    details = data.get("details") or []
    note_parts = [
        f"OKX depth={depth}",
        f"export_time={data.get('exportTime', '')}",
        f"total_size_mb={data.get('totalSizeMB', '')}",
    ]
    if not details:
        return [
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "status": "no_url_found",
                    "direct_url": "",
                    "notes": _append_note(entry.notes, "; ".join(note_parts + ["details=0"])),
                }
            )
        ]

    resolved: list[HistoricalL2ManifestEntry] = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict):
            continue
        direct_url = _detail_url(detail)
        detail_symbol = str(detail.get("instId") or detail.get("instFamily") or entry.symbol)
        status = "url_resolved" if direct_url else "url_missing"
        notes = _append_note(
            entry.notes,
            "; ".join(
                note_parts
                + [
                    f"detail_index={index}",
                    f"detail_symbol={detail_symbol}",
                    f"detail_size_mb={detail.get('fileSizeMB', detail.get('sizeMB', ''))}",
                ]
            ),
        )
        resolved.append(
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "symbol": entry.symbol,
                    "direct_url": direct_url,
                    "local_path": "" if index else entry.local_path,
                    "status": status,
                    "sha256": "",
                    "bytes": 0,
                    "rows_checked": 0,
                    "normalized_path": "",
                    "notes": notes,
                }
            )
        )
    if not resolved:
        return [
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "status": "url_missing",
                    "direct_url": "",
                    "notes": _append_note(entry.notes, "; ".join(note_parts + ["details_without_urls=1"])),
                }
            )
        ]
    return resolved


def _entries_from_bybit_download_data(
    entry: HistoricalL2ManifestEntry,
    data: dict,
) -> list[HistoricalL2ManifestEntry]:
    details = data.get("list") or []
    note_parts = [
        "Bybit product=orderbook",
        f"biz_type={_bybit_biz_type(entry.market)}",
    ]
    if not details:
        return [
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "status": "no_url_found",
                    "direct_url": "",
                    "notes": _append_note(entry.notes, "; ".join(note_parts + ["list=0"])),
                }
            )
        ]

    resolved: list[HistoricalL2ManifestEntry] = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict):
            continue
        direct_url = _detail_url(detail)
        status = "url_resolved" if direct_url else "url_missing"
        notes = _append_note(
            entry.notes,
            "; ".join(
                note_parts
                + [
                    f"detail_index={index}",
                    f"filename={detail.get('filename', '')}",
                    f"remote_size_bytes={detail.get('size', '')}",
                    f"period={detail.get('period', '')}",
                ]
            ),
        )
        resolved.append(
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "symbol": str(detail.get("symbol") or entry.symbol),
                    "direct_url": direct_url,
                    "local_path": "" if index else entry.local_path,
                    "status": status,
                    "sha256": "",
                    "bytes": 0,
                    "rows_checked": 0,
                    "normalized_path": "",
                    "notes": notes,
                }
            )
        )
    if not resolved:
        return [
            HistoricalL2ManifestEntry(
                **{
                    **asdict(entry),
                    "status": "url_missing",
                    "direct_url": "",
                    "notes": _append_note(entry.notes, "; ".join(note_parts + ["list_without_urls=1"])),
                }
            )
        ]
    return resolved


def _detail_url(detail: dict) -> str:
    for key in ("url", "downloadUrl", "downloadURL", "fileUrl", "fileURL"):
        value = detail.get(key)
        if value:
            return str(value)
    return ""


def _okx_order_book_module(depth: str) -> str:
    normalized = str(depth).strip().lower()
    try:
        return OKX_ORDER_BOOK_DEPTH_MODULES[normalized]
    except KeyError as exc:
        raise ValueError("OKX order-book depth must be 400 or 5000") from exc


def _okx_inst_type(market: str) -> str:
    normalized = market.strip().upper()
    aliases = {"PERP": "SWAP", "PERPETUAL": "SWAP"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"SPOT", "SWAP", "FUTURES", "OPTION"}:
        raise ValueError("OKX market must be spot, swap, futures, or option")
    return normalized


def _okx_inst_family(symbol: str) -> str:
    parts = symbol.strip().upper().split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return symbol.strip().upper()


def _bybit_biz_type(market: str) -> str:
    normalized = market.strip().lower()
    aliases = {
        "linear": "contract",
        "inverse": "contract",
        "swap": "contract",
        "perp": "contract",
        "perpetual": "contract",
        "futures": "contract",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"spot", "contract", "option"}:
        raise ValueError("Bybit market must be spot, contract, option, linear, inverse, swap, perp, or futures")
    return normalized


def _bybit_product_id(dataset: str) -> str:
    normalized = dataset.strip().lower().replace("-", "_")
    if normalized in {"orderbook", "order_book", "order_book_l2", "l2", "ob"}:
        return "orderbook"
    raise ValueError("Bybit historical resolver currently supports only orderbook datasets")


def _bybit_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("-", "")


def _is_bybit_orderbook_data_zip(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".data.zip") or ("_ob" in name and name.endswith(".zip"))


def _normalized_note(result: HistoricalL2ImportResult, *, max_import_rows: int | None) -> str:
    if max_import_rows is None:
        return f"normalized rows_written={result.rows_written}"
    return f"normalized rows_written={result.rows_written} max_import_rows={max_import_rows}"


def _utc_day_ms_range(session_date: str) -> tuple[int, int]:
    date_value = datetime.strptime(session_date, "%Y-%m-%d").date()
    begin = datetime.combine(date_value, datetime_time.min, tzinfo=timezone.utc)
    end = datetime.combine(date_value, datetime_time.max, tzinfo=timezone.utc)
    return int(begin.timestamp() * 1000), int(end.timestamp() * 1000)


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    return f"{existing} | {note}"


def _raw_manifest_path(raw_root: Path | str, entry: HistoricalL2ManifestEntry) -> Path:
    parsed = urllib.parse.urlparse(entry.direct_url)
    filename = Path(parsed.path).name or f"{entry.source_id}-{entry.symbol}-{entry.session_date}.csv"
    safe_symbol = entry.symbol.replace("/", "-").replace(":", "-")
    return Path(raw_root) / "historical_l2" / entry.source_id / safe_symbol / entry.session_date / filename


def _download_or_copy(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    temp_path = output_path.with_name(output_path.name + ".download")
    if parsed.scheme in {"", "file"}:
        source_path = Path(urllib.request.url2pathname(parsed.path)) if parsed.scheme == "file" else Path(url)
        shutil.copyfile(source_path, temp_path)
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    temp_path.replace(output_path)
