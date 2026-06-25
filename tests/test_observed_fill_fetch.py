import base64
import hashlib
import hmac
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lob_forge.cli import main as cli_main
from lob_forge.observed_fill_fetch import (
    fetch_observed_fill_export,
    format_observed_fill_fetch_report,
)


def test_fetch_bybit_demo_executions_signs_and_writes_raw_json(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str], float]] = []

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> dict:
        calls.append((url, dict(headers), timeout))
        return {"result": {"list": [{"orderLinkId": "d1", "execPrice": "100", "execQty": "0.1"}]}}

    output = tmp_path / "raw_bybit.json"
    report = fetch_observed_fill_export(
        provider="bybit",
        output_path=output,
        symbol="BTCUSDT",
        start_time_ms=1700000000000,
        end_time_ms=1700000100000,
        limit=1,
        env={"BYBIT_DEMO_API_KEY": "key", "BYBIT_DEMO_API_SECRET": "secret"},
        http_get_json=fake_get,
        now_ms=1711420489915,
    )

    url, headers, timeout = calls[0]
    query = urlparse(url).query
    expected_signature = hmac.new(
        b"secret",
        f"1711420489915key5000{query}".encode(),
        hashlib.sha256,
    ).hexdigest()

    assert report.provider == "bybit"
    assert report.source_id == "bybit_demo_fills"
    assert report.rows == 1
    assert report.endpoint == "/v5/execution/list"
    assert timeout == 30.0
    assert url.startswith("https://api-demo.bybit.com/v5/execution/list?")
    assert parse_qs(query)["category"] == ["linear"]
    assert parse_qs(query)["symbol"] == ["BTCUSDT"]
    assert headers["X-BAPI-API-KEY"] == "key"
    assert headers["X-BAPI-SIGN"] == expected_signature
    assert json.loads(output.read_text())["result"]["list"][0]["orderLinkId"] == "d1"


def test_fetch_okx_demo_fills_history_uses_simulated_header(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str], float]] = []

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> dict:
        calls.append((url, dict(headers), timeout))
        return {"data": [{"clOrdId": "d1", "fillPx": "100", "fillSz": "0.1"}]}

    output = tmp_path / "raw_okx.json"
    report = fetch_observed_fill_export(
        provider="okx",
        output_path=output,
        symbol="BTC-USDT-SWAP",
        start_time_ms=1700000000000,
        end_time_ms=1700000100000,
        limit=1,
        env={
            "OKX_DEMO_API_KEY": "key",
            "OKX_DEMO_API_SECRET": "secret",
            "OKX_DEMO_API_PASSPHRASE": "passphrase",
        },
        http_get_json=fake_get,
        now_ms=1711420489915,
    )

    url, headers, _timeout = calls[0]
    parsed = urlparse(url)
    request_path = parsed.path + "?" + parsed.query
    expected_signature = base64.b64encode(
        hmac.new(
            b"secret",
            (headers["OK-ACCESS-TIMESTAMP"] + "GET" + request_path).encode(),
            hashlib.sha256,
        ).digest()
    ).decode()

    assert report.provider == "okx"
    assert report.source_id == "okx_demo_fills"
    assert report.rows == 1
    assert url.startswith("https://eea.okx.com/api/v5/trade/fills-history?")
    assert parse_qs(parsed.query)["instId"] == ["BTC-USDT-SWAP"]
    assert parse_qs(parsed.query)["instType"] == ["SWAP"]
    assert headers["x-simulated-trading"] == "1"
    assert headers["OK-ACCESS-SIGN"] == expected_signature
    assert json.loads(output.read_text())["data"][0]["clOrdId"] == "d1"


def test_fetch_observed_fills_requires_explicit_credentials(tmp_path: Path) -> None:
    try:
        fetch_observed_fill_export(
            provider="bybit",
            output_path=tmp_path / "raw.json",
            env={},
            http_get_json=lambda _url, _headers, _timeout: {},
        )
    except ValueError as exc:
        assert "BYBIT_DEMO_API_KEY" in str(exc)
        assert "BYBIT_DEMO_API_SECRET" in str(exc)
    else:
        raise AssertionError("expected missing credential failure")


def test_fetch_observed_fills_report_formats_csv(tmp_path: Path) -> None:
    report = fetch_observed_fill_export(
        provider="okx",
        output_path=tmp_path / "raw.json",
        env={
            "OKX_DEMO_API_KEY": "key",
            "OKX_DEMO_API_SECRET": "secret",
            "OKX_DEMO_API_PASSPHRASE": "passphrase",
        },
        http_get_json=lambda _url, _headers, _timeout: {"data": []},
    )

    csv_text = format_observed_fill_fetch_report(report, output_format="csv")

    assert "provider,source_id,output_path,rows,endpoint" in csv_text
    assert "okx,okx_demo_fills" in csv_text


def test_cli_fetch_observed_fills_rejects_unknown_provider() -> None:
    try:
        cli_main(["fetch-observed-fills", "--provider", "alpaca", "--output", "raw.json"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse provider rejection")
