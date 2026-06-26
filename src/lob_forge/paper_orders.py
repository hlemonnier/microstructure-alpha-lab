from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lob_forge.live_validation import ShadowDecision, read_shadow_decisions


SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS = ("bybit", "okx", "binance")


@dataclass(frozen=True)
class PaperOrderInstruction:
    provider: str
    decision_id: str
    method: str
    endpoint: str
    symbol: str
    payload: dict[str, object]
    notes: str


@dataclass(frozen=True)
class PaperOrderPlanReport:
    provider: str
    output_path: Path
    output_format: str
    input_rows: int
    order_rows: int
    skipped_flat_rows: int
    skipped_size_rows: int


def build_paper_order_plan(
    shadow_path: Path | str,
    *,
    provider: str,
    limit: int = 0,
    category: str = "linear",
    td_mode: str = "cross",
    time_in_force: str = "GTC",
    symbol_override: str | None = None,
) -> list[PaperOrderInstruction]:
    provider = provider.lower()
    if provider not in SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS)}")
    decisions = read_shadow_decisions(shadow_path)
    if limit > 0:
        decisions = decisions[:limit]
    return [
        _instruction_for_decision(
            decision,
            provider=provider,
            category=category,
            td_mode=td_mode,
            time_in_force=time_in_force,
            symbol_override=symbol_override,
        )
        for decision in decisions
        if decision.predicted_side != 0 and decision.intended_size > 0.0
    ]


def write_paper_order_plan(
    *,
    shadow_path: Path | str,
    output_path: Path | str,
    provider: str,
    limit: int = 0,
    category: str = "linear",
    td_mode: str = "cross",
    time_in_force: str = "GTC",
    symbol_override: str | None = None,
    output_format: str = "jsonl",
) -> PaperOrderPlanReport:
    provider = provider.lower()
    if output_format not in {"csv", "jsonl"}:
        raise ValueError("output_format must be csv or jsonl")
    decisions = read_shadow_decisions(shadow_path)
    selected = decisions[:limit] if limit > 0 else decisions
    instructions = [
        _instruction_for_decision(
            decision,
            provider=provider,
            category=category,
            td_mode=td_mode,
            time_in_force=time_in_force,
            symbol_override=symbol_override,
        )
        for decision in selected
        if decision.predicted_side != 0 and decision.intended_size > 0.0
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        output.write_text(format_paper_order_instructions_csv(instructions) + "\n")
    else:
        output.write_text(format_paper_order_instructions_jsonl(instructions) + ("\n" if instructions else ""))
    return PaperOrderPlanReport(
        provider=provider,
        output_path=output,
        output_format=output_format,
        input_rows=len(selected),
        order_rows=len(instructions),
        skipped_flat_rows=sum(1 for decision in selected if decision.predicted_side == 0),
        skipped_size_rows=sum(
            1 for decision in selected if decision.predicted_side != 0 and decision.intended_size <= 0.0
        ),
    )


def format_paper_order_plan_report(report: PaperOrderPlanReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "provider",
            "output_path",
            "output_format",
            "input_rows",
            "order_rows",
            "skipped_flat_rows",
            "skipped_size_rows",
        ]
        values = [
            report.provider,
            str(report.output_path),
            report.output_format,
            str(report.input_rows),
            str(report.order_rows),
            str(report.skipped_flat_rows),
            str(report.skipped_size_rows),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    return "\n".join(
        [
            f"provider={report.provider}",
            f"output_path={report.output_path}",
            f"output_format={report.output_format}",
            f"input_rows={report.input_rows}",
            f"order_rows={report.order_rows}",
            f"skipped_flat_rows={report.skipped_flat_rows}",
            f"skipped_size_rows={report.skipped_size_rows}",
        ]
    )


def format_paper_order_instructions_jsonl(instructions: list[PaperOrderInstruction]) -> str:
    return "\n".join(json.dumps(asdict(instruction), sort_keys=True) for instruction in instructions)


def format_paper_order_instructions_csv(instructions: list[PaperOrderInstruction]) -> str:
    fields = ["provider", "decision_id", "method", "endpoint", "symbol", "payload_json", "notes"]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for instruction in instructions:
        writer.writerow(
            {
                "provider": instruction.provider,
                "decision_id": instruction.decision_id,
                "method": instruction.method,
                "endpoint": instruction.endpoint,
                "symbol": instruction.symbol,
                "payload_json": json.dumps(instruction.payload, sort_keys=True, separators=(",", ":")),
                "notes": instruction.notes,
            }
        )
    return handle.getvalue().strip("\r\n")


def _instruction_for_decision(
    decision: ShadowDecision,
    *,
    provider: str,
    category: str,
    td_mode: str,
    time_in_force: str,
    symbol_override: str | None,
) -> PaperOrderInstruction:
    if decision.order_type not in {"paper_taker", "paper_limit"}:
        raise ValueError(f"unsupported shadow order_type for {decision.decision_id}: {decision.order_type}")
    if decision.order_type == "paper_limit" and decision.intended_price <= 0.0:
        raise ValueError(f"limit shadow decision {decision.decision_id} requires a positive intended_price")
    symbol = _provider_symbol(provider, decision.symbol, symbol_override)
    if provider == "bybit":
        return _bybit_instruction(decision, symbol=symbol, category=category, time_in_force=time_in_force)
    if provider == "okx":
        return _okx_instruction(decision, symbol=symbol, td_mode=td_mode)
    if provider == "binance":
        return _binance_instruction(decision, symbol=symbol, time_in_force=time_in_force)
    raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_PAPER_ORDER_PLAN_PROVIDERS)}")


def _bybit_instruction(
    decision: ShadowDecision,
    *,
    symbol: str,
    category: str,
    time_in_force: str,
) -> PaperOrderInstruction:
    payload: dict[str, object] = {
        "category": category,
        "symbol": symbol,
        "side": "Buy" if decision.predicted_side > 0 else "Sell",
        "orderType": "Market" if decision.order_type == "paper_taker" else "Limit",
        "qty": _number(decision.intended_size),
        "orderLinkId": decision.decision_id,
    }
    if decision.order_type == "paper_limit":
        payload["price"] = _number(decision.intended_price)
        payload["timeInForce"] = time_in_force
    return PaperOrderInstruction(
        provider="bybit",
        decision_id=decision.decision_id,
        method="POST",
        endpoint="/v5/order/create",
        symbol=symbol,
        payload=payload,
        notes="demo trading: use orderLinkId=decision_id; fetch fills from /v5/execution/list",
    )


def _okx_instruction(decision: ShadowDecision, *, symbol: str, td_mode: str) -> PaperOrderInstruction:
    payload: dict[str, object] = {
        "instId": symbol,
        "tdMode": td_mode,
        "side": "buy" if decision.predicted_side > 0 else "sell",
        "ordType": "market" if decision.order_type == "paper_taker" else "limit",
        "sz": _number(decision.intended_size),
        "clOrdId": decision.decision_id,
    }
    if decision.order_type == "paper_limit":
        payload["px"] = _number(decision.intended_price)
    return PaperOrderInstruction(
        provider="okx",
        decision_id=decision.decision_id,
        method="POST",
        endpoint="/api/v5/trade/order",
        symbol=symbol,
        payload=payload,
        notes="demo trading: include x-simulated-trading: 1; use clOrdId=decision_id",
    )


def _binance_instruction(decision: ShadowDecision, *, symbol: str, time_in_force: str) -> PaperOrderInstruction:
    payload: dict[str, object] = {
        "symbol": symbol,
        "side": "BUY" if decision.predicted_side > 0 else "SELL",
        "type": "MARKET" if decision.order_type == "paper_taker" else "LIMIT",
        "quantity": _number(decision.intended_size),
        "newClientOrderId": decision.decision_id,
    }
    if decision.order_type == "paper_limit":
        payload["price"] = _number(decision.intended_price)
        payload["timeInForce"] = time_in_force
    return PaperOrderInstruction(
        provider="binance",
        decision_id=decision.decision_id,
        method="POST",
        endpoint="/fapi/v1/order",
        symbol=symbol,
        payload=payload,
        notes="USD-M futures testnet: sign payload and use newClientOrderId=decision_id",
    )


def _provider_symbol(provider: str, symbol: str, symbol_override: str | None) -> str:
    if symbol_override:
        return symbol_override
    if provider == "okx":
        return _okx_swap_symbol(symbol)
    return symbol


def _okx_swap_symbol(symbol: str) -> str:
    if "-" in symbol:
        return symbol
    upper = symbol.upper()
    if upper.endswith("USDT") and len(upper) > len("USDT"):
        return f"{upper[:-4]}-USDT-SWAP"
    return symbol


def _number(value: float) -> str:
    return format(value, ".12g")
