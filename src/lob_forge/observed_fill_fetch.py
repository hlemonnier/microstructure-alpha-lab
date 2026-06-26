from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_OBSERVED_FILL_FETCH_PROVIDERS = ("bybit", "okx", "binance")

BYBIT_DEMO_BASE_URL = "https://api-demo.bybit.com"
OKX_DEMO_BASE_URL = "https://eea.okx.com"
BINANCE_USDM_TESTNET_BASE_URL = "https://demo-fapi.binance.com"

HttpGetJson = Callable[[str, Mapping[str, str], float], Any]


@dataclass(frozen=True)
class ObservedFillFetchReport:
    provider: str
    output_path: Path
    rows: int
    endpoint: str
    source_id: str


def fetch_observed_fill_export(
    *,
    provider: str,
    output_path: Path | str,
    symbol: str | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = 50,
    category: str = "linear",
    inst_type: str = "SWAP",
    cursor: str | None = None,
    base_url: str | None = None,
    recv_window: int = 5000,
    timeout_seconds: float = 30.0,
    env: Mapping[str, str] | None = None,
    http_get_json: HttpGetJson | None = None,
    now_ms: int | None = None,
) -> ObservedFillFetchReport:
    provider = provider.lower()
    if provider == "bybit":
        payload, endpoint = fetch_bybit_demo_executions(
            output_symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=limit,
            category=category,
            cursor=cursor,
            base_url=base_url or BYBIT_DEMO_BASE_URL,
            recv_window=recv_window,
            timeout_seconds=timeout_seconds,
            env=env,
            http_get_json=http_get_json,
            now_ms=now_ms,
        )
        source_id = "bybit_demo_fills"
    elif provider == "okx":
        payload, endpoint = fetch_okx_demo_fills_history(
            inst_id=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=limit,
            inst_type=inst_type,
            base_url=base_url or OKX_DEMO_BASE_URL,
            timeout_seconds=timeout_seconds,
            env=env,
            http_get_json=http_get_json,
            now_ms=now_ms,
        )
        source_id = "okx_demo_fills"
    elif provider == "binance":
        payload, endpoint = fetch_binance_usdm_testnet_orders(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=limit,
            base_url=base_url or BINANCE_USDM_TESTNET_BASE_URL,
            recv_window=recv_window,
            timeout_seconds=timeout_seconds,
            env=env,
            http_get_json=http_get_json,
            now_ms=now_ms,
        )
        source_id = "binance_usdm_testnet_orders"
    else:
        raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_OBSERVED_FILL_FETCH_PROVIDERS)}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ObservedFillFetchReport(
        provider=provider,
        output_path=output,
        rows=_count_provider_rows(payload),
        endpoint=endpoint,
        source_id=source_id,
    )


def fetch_bybit_demo_executions(
    *,
    output_symbol: str | None,
    start_time_ms: int | None,
    end_time_ms: int | None,
    limit: int,
    category: str,
    cursor: str | None,
    base_url: str,
    recv_window: int,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    http_get_json: HttpGetJson | None = None,
    now_ms: int | None = None,
) -> tuple[Any, str]:
    if not 1 <= limit <= 100:
        raise ValueError("Bybit execution history limit must be between 1 and 100")
    credentials = _credentials(env, "BYBIT_DEMO_API_KEY", "BYBIT_DEMO_API_SECRET")
    timestamp = str(now_ms if now_ms is not None else int(time.time() * 1000))
    params = {
        "category": category,
        "symbol": output_symbol,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": limit,
        "cursor": cursor,
    }
    query = _query_string(params)
    sign_payload = timestamp + credentials["BYBIT_DEMO_API_KEY"] + str(recv_window) + query
    signature = hmac.new(
        credentials["BYBIT_DEMO_API_SECRET"].encode("utf-8"),
        sign_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "X-BAPI-API-KEY": credentials["BYBIT_DEMO_API_KEY"],
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": str(recv_window),
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json",
    }
    endpoint = "/v5/execution/list"
    url = _url(base_url, endpoint, query)
    return (http_get_json or _http_get_json)(url, headers, timeout_seconds), endpoint


def fetch_okx_demo_fills_history(
    *,
    inst_id: str | None,
    start_time_ms: int | None,
    end_time_ms: int | None,
    limit: int,
    inst_type: str,
    base_url: str,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    http_get_json: HttpGetJson | None = None,
    now_ms: int | None = None,
) -> tuple[Any, str]:
    if not 1 <= limit <= 100:
        raise ValueError("OKX fills-history limit must be between 1 and 100")
    credentials = _credentials(env, "OKX_DEMO_API_KEY", "OKX_DEMO_API_SECRET", "OKX_DEMO_API_PASSPHRASE")
    timestamp = _okx_timestamp(now_ms)
    params = {
        "instType": inst_type,
        "instId": inst_id,
        "begin": start_time_ms,
        "end": end_time_ms,
        "limit": limit,
    }
    query = _query_string(params)
    endpoint = "/api/v5/trade/fills-history"
    request_path = endpoint + (f"?{query}" if query else "")
    sign_payload = timestamp + "GET" + request_path
    signature = base64.b64encode(
        hmac.new(
            credentials["OKX_DEMO_API_SECRET"].encode("utf-8"),
            sign_payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")
    headers = {
        "OK-ACCESS-KEY": credentials["OKX_DEMO_API_KEY"],
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": credentials["OKX_DEMO_API_PASSPHRASE"],
        "x-simulated-trading": "1",
        "Content-Type": "application/json",
    }
    return (http_get_json or _http_get_json)(_url(base_url, endpoint, query), headers, timeout_seconds), endpoint


def fetch_binance_usdm_testnet_orders(
    *,
    symbol: str | None,
    start_time_ms: int | None,
    end_time_ms: int | None,
    limit: int,
    base_url: str,
    recv_window: int,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    http_get_json: HttpGetJson | None = None,
    now_ms: int | None = None,
) -> tuple[Any, str]:
    if not symbol:
        raise ValueError("Binance USD-M futures testnet order export requires --symbol")
    if not 1 <= limit <= 1000:
        raise ValueError("Binance allOrders limit must be between 1 and 1000")
    credentials = _credentials(env, "BINANCE_USDM_TESTNET_API_KEY", "BINANCE_USDM_TESTNET_API_SECRET")
    params = {
        "symbol": symbol,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": limit,
        "recvWindow": recv_window,
        "timestamp": now_ms if now_ms is not None else int(time.time() * 1000),
    }
    unsigned_query = _query_string(params)
    signature = hmac.new(
        credentials["BINANCE_USDM_TESTNET_API_SECRET"].encode("utf-8"),
        unsigned_query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    query = f"{unsigned_query}&signature={signature}"
    headers = {
        "X-MBX-APIKEY": credentials["BINANCE_USDM_TESTNET_API_KEY"],
        "Content-Type": "application/json",
    }
    endpoint = "/fapi/v1/allOrders"
    return (http_get_json or _http_get_json)(_url(base_url, endpoint, query), headers, timeout_seconds), endpoint


def format_observed_fill_fetch_report(report: ObservedFillFetchReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = ["provider", "source_id", "output_path", "rows", "endpoint"]
        values = [
            report.provider,
            report.source_id,
            str(report.output_path),
            str(report.rows),
            report.endpoint,
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    return "\n".join(
        [
            f"provider={report.provider}",
            f"source_id={report.source_id}",
            f"output_path={report.output_path}",
            f"rows={report.rows}",
            f"endpoint={report.endpoint}",
        ]
    )


def _credentials(env: Mapping[str, str] | None, *names: str) -> dict[str, str]:
    source = os.environ if env is None else env
    missing = [name for name in names if not source.get(name)]
    if missing:
        raise ValueError("missing required environment variables: " + ", ".join(missing))
    return {name: source[name] for name in names}


def _query_string(params: Mapping[str, Any]) -> str:
    values = [(key, str(value)) for key, value in sorted(params.items()) if value is not None and value != ""]
    return urllib.parse.urlencode(values)


def _url(base_url: str, endpoint: str, query: str) -> str:
    return base_url.rstrip("/") + endpoint + (f"?{query}" if query else "")


def _okx_timestamp(now_ms: int | None) -> str:
    now = datetime.fromtimestamp((now_ms if now_ms is not None else int(time.time() * 1000)) / 1000, tz=timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _http_get_json(url: str, headers: Mapping[str, str], timeout_seconds: float) -> Any:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _count_provider_rows(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("list"), list):
            return len(result["list"])
        data = payload.get("data")
        if isinstance(data, list):
            return len(data)
        rows = payload.get("list")
        if isinstance(rows, list):
            return len(rows)
        return 1
    return 0
