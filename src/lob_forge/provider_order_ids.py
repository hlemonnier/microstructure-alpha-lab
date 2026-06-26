from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROVIDER_CLIENT_ORDER_ID_FIELDS = {
    "bybit": "orderLinkId",
    "okx": "clOrdId",
    "binance": "newClientOrderId",
}


@dataclass(frozen=True)
class ProviderClientOrderIdRule:
    max_length: int
    pattern: re.Pattern[str]
    description: str


PROVIDER_CLIENT_ORDER_ID_RULES = {
    "bybit": ProviderClientOrderIdRule(
        max_length=36,
        pattern=re.compile(r"^[A-Za-z0-9_-]+$"),
        description="letters, numbers, dashes, and underscores",
    ),
    "okx": ProviderClientOrderIdRule(
        max_length=32,
        pattern=re.compile(r"^[A-Za-z0-9]+$"),
        description="case-sensitive alphanumerics",
    ),
    "binance": ProviderClientOrderIdRule(
        max_length=36,
        pattern=re.compile(r"^[.A-Z:/a-z0-9_-]+$"),
        description="letters, numbers, '.', ':', '/', underscores, and dashes",
    ),
}


def client_order_id_for_decision(provider: str, decision_id: str) -> str:
    """Return a deterministic provider-safe client order ID for a shadow decision."""
    if not decision_id:
        raise ValueError("decision_id must be non-empty")
    provider = provider.lower()
    try:
        return validate_provider_client_order_id(provider, decision_id)
    except ValueError:
        digest = hashlib.sha256(f"{provider}:{decision_id}".encode("utf-8")).hexdigest()
        return validate_provider_client_order_id(provider, "P" + digest[:31])


def provider_client_order_id_field(provider: str) -> str:
    provider = provider.lower()
    try:
        return PROVIDER_CLIENT_ORDER_ID_FIELDS[provider]
    except KeyError as exc:
        raise ValueError(f"unsupported provider for client order id: {provider}") from exc


def provider_payload_client_order_id(payload: Mapping[str, object], *, provider: str) -> str:
    return str(payload.get(provider_client_order_id_field(provider), ""))


def validate_provider_client_order_id(provider: str, client_order_id: str) -> str:
    provider = provider.lower()
    try:
        rule = PROVIDER_CLIENT_ORDER_ID_RULES[provider]
    except KeyError as exc:
        raise ValueError(f"unsupported provider for client order id: {provider}") from exc
    if not client_order_id:
        raise ValueError(f"{provider} client_order_id must be non-empty")
    if len(client_order_id) > rule.max_length:
        raise ValueError(
            f"{provider} client_order_id={client_order_id!r} is {len(client_order_id)} characters; "
            f"maximum is {rule.max_length}"
        )
    if not rule.pattern.fullmatch(client_order_id):
        raise ValueError(f"{provider} client_order_id={client_order_id!r} must contain only {rule.description}")
    return client_order_id


def read_order_plan_client_id_map(
    path: Path | str,
    *,
    provider: str | None = None,
) -> dict[str, str]:
    """Read a provider order plan as client_order_id -> decision_id."""
    rows = _read_order_plan_rows(Path(path))
    client_to_decision: dict[str, str] = {}
    for row_number, row in rows:
        row_provider = str(row.get("provider") or provider or "").lower()
        if provider is not None and row_provider != provider.lower():
            raise ValueError(
                f"order plan row {row_number} provider={row_provider!r}; expected provider={provider.lower()!r}"
            )
        decision_id = str(row.get("decision_id") or "")
        if not decision_id:
            raise ValueError(f"order plan row {row_number} missing decision_id")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"order plan row {row_number} payload must be an object")
        client_order_id = str(
            row.get("client_order_id") or provider_payload_client_order_id(payload, provider=row_provider)
        )
        validate_provider_client_order_id(row_provider, client_order_id)
        existing = client_to_decision.get(client_order_id)
        if existing is not None and existing != decision_id:
            raise ValueError(f"client_order_id={client_order_id!r} maps to both {existing!r} and {decision_id!r}")
        client_to_decision[client_order_id] = decision_id
    return client_to_decision


def _read_order_plan_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise ValueError(f"missing order plan: {path}")
    if path.suffix.lower() == ".csv":
        return _read_order_plan_csv_rows(path)
    return _read_order_plan_jsonl_rows(path)


def _read_order_plan_jsonl_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"order plan row {line_number} in {path} must be an object")
        rows.append((line_number, dict(row)))
    return rows


def _read_order_plan_csv_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            payload = json.loads(row.get("payload_json", ""))
            rows.append(
                (
                    line_number,
                    {
                        "provider": row.get("provider", ""),
                        "decision_id": row.get("decision_id", ""),
                        "client_order_id": row.get("client_order_id", ""),
                        "payload": payload,
                    },
                )
            )
    return rows
