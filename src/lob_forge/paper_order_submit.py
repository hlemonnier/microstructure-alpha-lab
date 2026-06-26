from __future__ import annotations

import csv
import json
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lob_forge.observed_fill_fetch import (
    BINANCE_USDM_TESTNET_BASE_URL,
    BYBIT_DEMO_BASE_URL,
    OKX_DEMO_BASE_URL,
    _credentials,
    _okx_timestamp,
    _query_string,
    _url,
)
from lob_forge.paper_orders import SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS, PaperOrderInstruction

HttpPostJson = Callable[[str, Mapping[str, str], Optional[str], float], Any]


@dataclass(frozen=True)
class PaperOrderSubmissionReport:
    provider: str
    output_path: Path
    dry_run: bool
    input_rows: int
    order_rows: int
    submitted_rows: int
    base_url: str


def submit_paper_order_plan(
    *,
    provider: str,
    plan_path: Path | str,
    output_path: Path | str,
    execute: bool = False,
    limit: int = 0,
    recv_window: int = 5000,
    timeout_seconds: float = 30.0,
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
    http_post_json: HttpPostJson | None = None,
    now_ms: int | None = None,
) -> PaperOrderSubmissionReport:
    provider = provider.lower()
    if provider not in SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS)}")
    instructions = _read_plan(plan_path)
    selected = instructions[:limit] if limit > 0 else instructions
    for instruction in selected:
        _validate_instruction(instruction, provider=provider)
    resolved_base_url = base_url or _default_base_url(provider)
    rows = [
        _submit_or_preview(
            instruction,
            provider=provider,
            base_url=resolved_base_url,
            execute=execute,
            recv_window=recv_window,
            timeout_seconds=timeout_seconds,
            env=env,
            http_post_json=http_post_json,
            now_ms=now_ms,
        )
        for instruction in selected
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    return PaperOrderSubmissionReport(
        provider=provider,
        output_path=output,
        dry_run=not execute,
        input_rows=len(instructions),
        order_rows=len(selected),
        submitted_rows=len(rows) if execute else 0,
        base_url=resolved_base_url,
    )


def format_paper_order_submission_report(
    report: PaperOrderSubmissionReport,
    *,
    output_format: str = "text",
) -> str:
    if output_format == "csv":
        fields = ["provider", "output_path", "dry_run", "input_rows", "order_rows", "submitted_rows", "base_url"]
        values = [
            report.provider,
            str(report.output_path),
            str(int(report.dry_run)),
            str(report.input_rows),
            str(report.order_rows),
            str(report.submitted_rows),
            report.base_url,
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    return "\n".join(
        [
            f"provider={report.provider}",
            f"output_path={report.output_path}",
            f"dry_run={int(report.dry_run)}",
            f"input_rows={report.input_rows}",
            f"order_rows={report.order_rows}",
            f"submitted_rows={report.submitted_rows}",
            f"base_url={report.base_url}",
        ]
    )


def _submit_or_preview(
    instruction: PaperOrderInstruction,
    *,
    provider: str,
    base_url: str,
    execute: bool,
    recv_window: int,
    timeout_seconds: float,
    env: Mapping[str, str] | None,
    http_post_json: HttpPostJson | None,
    now_ms: int | None,
) -> dict[str, Any]:
    if not execute:
        return {
            "provider": provider,
            "decision_id": instruction.decision_id,
            "endpoint": instruction.endpoint,
            "symbol": instruction.symbol,
            "dry_run": True,
            "payload": instruction.payload,
        }
    if provider == "bybit":
        response = _post_bybit_order(
            instruction,
            base_url=base_url,
            recv_window=recv_window,
            timeout_seconds=timeout_seconds,
            env=env,
            http_post_json=http_post_json,
            now_ms=now_ms,
        )
    elif provider == "okx":
        response = _post_okx_order(
            instruction,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            env=env,
            http_post_json=http_post_json,
            now_ms=now_ms,
        )
    elif provider == "binance":
        response = _post_binance_order(
            instruction,
            base_url=base_url,
            recv_window=recv_window,
            timeout_seconds=timeout_seconds,
            env=env,
            http_post_json=http_post_json,
            now_ms=now_ms,
        )
    else:
        raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS)}")
    return {
        "provider": provider,
        "decision_id": instruction.decision_id,
        "endpoint": instruction.endpoint,
        "symbol": instruction.symbol,
        "dry_run": False,
        "payload": instruction.payload,
        "response": response,
    }


def _post_bybit_order(
    instruction: PaperOrderInstruction,
    *,
    base_url: str,
    recv_window: int,
    timeout_seconds: float,
    env: Mapping[str, str] | None,
    http_post_json: HttpPostJson | None,
    now_ms: int | None,
) -> Any:
    credentials = _credentials(env, "BYBIT_DEMO_API_KEY", "BYBIT_DEMO_API_SECRET")
    timestamp = str(now_ms if now_ms is not None else int(time.time() * 1000))
    body = _json_body(instruction.payload)
    signature_payload = timestamp + credentials["BYBIT_DEMO_API_KEY"] + str(recv_window) + body
    signature = _hmac_sha256_hex(credentials["BYBIT_DEMO_API_SECRET"], signature_payload)
    headers = {
        "X-BAPI-API-KEY": credentials["BYBIT_DEMO_API_KEY"],
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": str(recv_window),
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json",
    }
    return (http_post_json or _http_post_json)(_url(base_url, instruction.endpoint, ""), headers, body, timeout_seconds)


def _post_okx_order(
    instruction: PaperOrderInstruction,
    *,
    base_url: str,
    timeout_seconds: float,
    env: Mapping[str, str] | None,
    http_post_json: HttpPostJson | None,
    now_ms: int | None,
) -> Any:
    credentials = _credentials(env, "OKX_DEMO_API_KEY", "OKX_DEMO_API_SECRET", "OKX_DEMO_API_PASSPHRASE")
    timestamp = _okx_timestamp(now_ms)
    body = _json_body(instruction.payload)
    signature_payload = timestamp + "POST" + instruction.endpoint + body
    signature = _hmac_sha256_base64(credentials["OKX_DEMO_API_SECRET"], signature_payload)
    headers = {
        "OK-ACCESS-KEY": credentials["OKX_DEMO_API_KEY"],
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": credentials["OKX_DEMO_API_PASSPHRASE"],
        "x-simulated-trading": "1",
        "Content-Type": "application/json",
    }
    return (http_post_json or _http_post_json)(_url(base_url, instruction.endpoint, ""), headers, body, timeout_seconds)


def _post_binance_order(
    instruction: PaperOrderInstruction,
    *,
    base_url: str,
    recv_window: int,
    timeout_seconds: float,
    env: Mapping[str, str] | None,
    http_post_json: HttpPostJson | None,
    now_ms: int | None,
) -> Any:
    credentials = _credentials(env, "BINANCE_USDM_TESTNET_API_KEY", "BINANCE_USDM_TESTNET_API_SECRET")
    params = {
        **instruction.payload,
        "recvWindow": recv_window,
        "timestamp": now_ms if now_ms is not None else int(time.time() * 1000),
    }
    unsigned_query = _query_string(params)
    signature = _hmac_sha256_hex(credentials["BINANCE_USDM_TESTNET_API_SECRET"], unsigned_query)
    query = f"{unsigned_query}&signature={signature}"
    headers = {
        "X-MBX-APIKEY": credentials["BINANCE_USDM_TESTNET_API_KEY"],
        "Content-Type": "application/json",
    }
    return (http_post_json or _http_post_json)(
        _url(base_url, instruction.endpoint, query), headers, None, timeout_seconds
    )


def _read_plan(plan_path: Path | str) -> list[PaperOrderInstruction]:
    path = Path(plan_path)
    if not path.exists():
        raise ValueError(f"missing paper order plan: {path}")
    if path.suffix.lower() == ".csv":
        return _read_plan_csv(path)
    return _read_plan_jsonl(path)


def _read_plan_jsonl(path: Path) -> list[PaperOrderInstruction]:
    instructions: list[PaperOrderInstruction] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            instructions.append(_instruction_from_mapping(row))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL order plan row {line_number} in {path}: {exc}") from exc
    return instructions


def _read_plan_csv(path: Path) -> list[PaperOrderInstruction]:
    instructions: list[PaperOrderInstruction] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            try:
                payload = json.loads(row.get("payload_json", ""))
                instructions.append(
                    PaperOrderInstruction(
                        provider=row["provider"],
                        decision_id=row["decision_id"],
                        method=row["method"],
                        endpoint=row["endpoint"],
                        symbol=row["symbol"],
                        payload=payload,
                        notes=row.get("notes", ""),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CSV order plan row {line_number} in {path}: {exc}") from exc
    return instructions


def _instruction_from_mapping(row: Mapping[str, Any]) -> PaperOrderInstruction:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return PaperOrderInstruction(
        provider=str(row["provider"]),
        decision_id=str(row["decision_id"]),
        method=str(row["method"]),
        endpoint=str(row["endpoint"]),
        symbol=str(row["symbol"]),
        payload=payload,
        notes=str(row.get("notes", "")),
    )


def _validate_instruction(instruction: PaperOrderInstruction, *, provider: str) -> None:
    expected_endpoint = {
        "bybit": "/v5/order/create",
        "okx": "/api/v5/trade/order",
        "binance": "/fapi/v1/order",
    }[provider]
    if instruction.provider != provider:
        raise ValueError(
            f"plan row {instruction.decision_id} provider={instruction.provider}; expected provider={provider}"
        )
    if instruction.method.upper() != "POST":
        raise ValueError(f"plan row {instruction.decision_id} method must be POST")
    if instruction.endpoint != expected_endpoint:
        raise ValueError(
            f"plan row {instruction.decision_id} endpoint={instruction.endpoint}; expected {expected_endpoint}"
        )


def _default_base_url(provider: str) -> str:
    if provider == "bybit":
        return BYBIT_DEMO_BASE_URL
    if provider == "okx":
        return OKX_DEMO_BASE_URL
    if provider == "binance":
        return BINANCE_USDM_TESTNET_BASE_URL
    raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS)}")


def _json_body(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _hmac_sha256_hex(secret: str, payload: str) -> str:
    import hashlib
    import hmac

    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _hmac_sha256_base64(secret: str, payload: str) -> str:
    import base64
    import hashlib
    import hmac

    return base64.b64encode(hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()).decode(
        "ascii"
    )


def _http_post_json(url: str, headers: Mapping[str, str], body: str | None, timeout_seconds: float) -> Any:
    data = body.encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))
