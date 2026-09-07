import csv
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone

from lob_forge.binance_vision import archive_key, sha256_file, url_for_key
from lob_forge.features import AGG_TRADE_COLUMNS, BOOK_TICKER_COLUMNS, iter_quote_events


def _preparation():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_performance_pilot.py"
    spec = importlib.util.spec_from_file_location("_prepare_performance_pilot_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _quotes(path: Path, events: list[tuple[int, int]]) -> None:
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(BOOK_TICKER_COLUMNS)
    for timestamp, update_id in events:
        writer.writerow([update_id, 100.0, 2.0, 100.1, 3.0, timestamp, timestamp])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source.csv", text.getvalue())


def test_pilot_prefix_preserves_all_events_and_excludes_end(tmp_path: Path) -> None:
    preparation = _preparation()
    source = tmp_path / "source.zip"
    _quotes(source, [(999, 1), (1000, 2), (1500, 3), (1500, 4), (2999, 5), (3000, 6)])
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"
    result = preparation.extract_prefix(source, first, dataset="bookTicker", start_ms=1000, end_ms=3000)
    preparation.extract_prefix(source, second, dataset="bookTicker", start_ms=1000, end_ms=3000)
    assert [q.update_id for q in iter_quote_events(first)] == [2, 3, 4, 5]
    assert result["rows"] == 4
    assert result["first_excluded_event_time_ms"] == 3000
    assert result["first_event_time_ms"] == 1000
    assert result["last_event_time_ms"] == 2999
    assert first.read_bytes() == second.read_bytes()


def test_pilot_prefix_rejects_duplicate_ids_and_incomplete_coverage(tmp_path: Path) -> None:
    preparation = _preparation()
    source = tmp_path / "source.zip"
    destination = tmp_path / "prefix.zip"
    for events in [[(1000, 1), (1000, 1), (3000, 2)], [(1000, 1), (2000, 2)]]:
        _quotes(source, events)
        try:
            preparation.extract_prefix(source, destination, dataset="bookTicker", start_ms=1000, end_ms=3000)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid prefix must be rejected")
        assert not destination.exists()
        assert not destination.with_name("prefix.zip.partial").exists()


def test_pilot_plan_rejects_reserved_dates_duplicates_and_oversized_files() -> None:
    preparation = _preparation()
    entries = []
    for symbol in preparation.SYMBOLS:
        for day in preparation.DATES:
            for dataset in preparation.DATASETS:
                key = archive_key(market="futures/um", frequency="daily", dataset=dataset,
                                  symbol=symbol, date_value=day)
                entries.append({"symbol": symbol, "session_date": day, "dataset": dataset,
                                "key": key, "url": url_for_key(key), "checksum_url": url_for_key(key + ".CHECKSUM"),
                                "status": 200, "bytes": 100})
    valid = {"archives": entries, "total_compressed_bytes": 3200}
    assert len(preparation.validated_archives(valid)) == 32
    invalid = [
        {"archives": [*entries[:-1], entries[0]], "total_compressed_bytes": 3200},
        {"archives": [{**entries[0], "session_date": "2023-05-24"}, *entries[1:]], "total_compressed_bytes": 3200},
        {"archives": [{**entries[0], "bytes": 2_000_000_000}, *entries[1:]], "total_compressed_bytes": 2_000_003_100},
    ]
    for plan in invalid:
        try:
            preparation.validated_archives(plan)
        except ValueError:
            pass
        else:
            raise AssertionError("unexpected/oversized scope must be rejected")


def test_pilot_acquisition_rejects_preregistration_changes(tmp_path: Path) -> None:
    preparation = _preparation()
    protocol = tmp_path / "protocol.json"
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n")
    payload = {
        "data": {"symbols": list(preparation.SYMBOLS), "start_date": "2023-05-16", "end_date": "2023-05-23",
                 "slice_start_utc": "00:00:00", "slice_end_utc": "02:00:30",
                 "max_compressed_download_bytes": 2_000_000_000, "max_parallel_downloads": 2,
                 "require_official_sha256": True},
        "features": {"bucket_ms": 1000, "horizon_ms": 5000, "execution_latency_ms": 100, "with_book_depth": False},
    }
    protocol.write_text(json.dumps(payload))
    output = tmp_path / "output"
    original = preparation.bind_acquisition(output, protocol, plan)
    assert preparation.bind_acquisition(output, protocol, plan) == original
    payload["models"] = ["unregistered_change"]
    protocol.write_text(json.dumps(payload))
    try:
        preparation.bind_acquisition(output, protocol, plan)
    except ValueError as error:
        assert "changed after acquisition" in str(error)
    else:
        raise AssertionError("acquisition must not silently follow a changed registration")


def test_pilot_session_uses_hashed_cache_and_rebuilds_tampered_features(tmp_path: Path) -> None:
    preparation = _preparation()
    symbol, day = "BTCUSDT", "2023-05-16"
    start = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    output = output_root / symbol / day
    output.mkdir(parents=True)
    entries = []
    for dataset in preparation.DATASETS:
        key = archive_key(market="futures/um", frequency="daily", dataset=dataset, symbol=symbol, date_value=day)
        path = raw_root / key
        path.parent.mkdir(parents=True)
        if dataset == "bookTicker":
            _quotes(path, [(start + 1000 * i, i + 1) for i in range(15)] + [(start + 7_230_000, 16)])
        else:
            text = io.StringIO()
            writer = csv.writer(text)
            writer.writerow(AGG_TRADE_COLUMNS)
            for i, timestamp in enumerate([start + 500, start + 7_230_000]):
                writer.writerow([i + 1, 100.1, 1.0, i + 1, i + 1, timestamp, "false"])
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("source.csv", text.getvalue())
        entry = {"symbol": symbol, "session_date": day, "dataset": dataset, "key": key,
                 "url": url_for_key(key), "checksum_url": url_for_key(key + ".CHECKSUM"),
                 "status": 200, "bytes": path.stat().st_size}
        record = {"key": key, "url": entry["url"], "checksum_url": entry["checksum_url"],
                  "bytes": entry["bytes"], "sha256": sha256_file(path),
                  "provider_checksum_sha256": sha256_file(path), "local_path": str(path.resolve())}
        (output / f"{dataset}.source.json").write_text(json.dumps(record))
        entries.append(entry)
    result = preparation.prepare_session(entries, output_root=output_root, raw_root=raw_root,
                                         identity={"fixture_only": True, "prefix_start_utc": "00:00:00"}, code_hash="test-code")
    feature = Path(result["features_path"])
    expected_hash = sha256_file(feature)
    assert result["feature_rows"] > 0
    before = feature.stat().st_mtime_ns
    preparation.prepare_session(entries, output_root=output_root, raw_root=raw_root,
                                identity={"fixture_only": True, "prefix_start_utc": "00:00:00"}, code_hash="test-code")
    assert feature.stat().st_mtime_ns == before
    feature.write_text("tampered\n")
    preparation.prepare_session(entries, output_root=output_root, raw_root=raw_root,
                                identity={"fixture_only": True, "prefix_start_utc": "00:00:00"}, code_hash="test-code")
    assert sha256_file(feature) == expected_hash
