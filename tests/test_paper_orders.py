from __future__ import annotations

import csv
import json
from pathlib import Path

from lob_forge.cli import main as cli_main
from lob_forge.live_validation import ShadowDecision, append_shadow_decision
from lob_forge.paper_orders import build_paper_order_plan, write_paper_order_plan
from lob_forge.provider_order_ids import client_order_id_for_decision


def test_paper_order_plan_builds_bybit_demo_payloads_from_shadow_decisions(tmp_path: Path) -> None:
    shadow = _shadow_file(tmp_path)

    instructions = build_paper_order_plan(shadow, provider="bybit")

    assert len(instructions) == 2
    assert instructions[0].endpoint == "/v5/order/create"
    assert instructions[0].client_order_id == "d1"
    assert instructions[0].payload == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": "0.01",
        "orderLinkId": "d1",
        "price": "65000.5",
        "timeInForce": "GTC",
    }
    assert instructions[1].payload == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Sell",
        "orderType": "Market",
        "qty": "0.02",
        "orderLinkId": "d2",
    }


def test_paper_order_plan_writes_jsonl_and_skips_non_orders(tmp_path: Path) -> None:
    shadow = _shadow_file(tmp_path)
    output = tmp_path / "plan.jsonl"

    report = write_paper_order_plan(shadow_path=shadow, output_path=output, provider="okx", output_format="jsonl")

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert report.input_rows == 4
    assert report.order_rows == 2
    assert report.skipped_flat_rows == 1
    assert report.skipped_size_rows == 1
    assert lines[0]["endpoint"] == "/api/v5/trade/order"
    assert lines[0]["symbol"] == "BTC-USDT-SWAP"
    assert lines[0]["client_order_id"] == "d1"
    assert lines[0]["payload"]["instId"] == "BTC-USDT-SWAP"
    assert lines[0]["payload"]["clOrdId"] == "d1"
    assert lines[0]["payload"]["ordType"] == "limit"
    assert lines[0]["payload"]["px"] == "65000.5"
    assert lines[1]["payload"]["side"] == "sell"
    assert "px" not in lines[1]["payload"]


def test_paper_order_plan_cli_writes_binance_csv(tmp_path: Path) -> None:
    shadow = _shadow_file(tmp_path)
    output = tmp_path / "plan.csv"

    code = cli_main(
        [
            "paper-order-plan",
            "--shadow",
            str(shadow),
            "--provider",
            "binance",
            "--output",
            str(output),
            "--plan-format",
            "csv",
        ]
    )

    rows = list(csv.DictReader(output.read_text().splitlines()))
    first_payload = json.loads(rows[0]["payload_json"])
    assert code == 0
    assert rows[0]["endpoint"] == "/fapi/v1/order"
    assert rows[0]["client_order_id"] == "d1"
    assert first_payload["newClientOrderId"] == "d1"
    assert first_payload["type"] == "LIMIT"
    assert first_payload["timeInForce"] == "GTC"


def test_paper_order_plan_allows_explicit_symbol_override(tmp_path: Path) -> None:
    shadow = _shadow_file(tmp_path)

    instructions = build_paper_order_plan(
        shadow,
        provider="okx",
        limit=1,
        symbol_override="ETH-USDT-SWAP",
    )

    assert len(instructions) == 1
    assert instructions[0].symbol == "ETH-USDT-SWAP"
    assert instructions[0].payload["instId"] == "ETH-USDT-SWAP"


def test_paper_order_plan_maps_long_decision_ids_to_provider_safe_client_ids(tmp_path: Path) -> None:
    long_decision_id = "BTCUSDT-2023-05-17-f1-1684313188997-0"
    shadow = tmp_path / "shadow.csv"
    output = tmp_path / "plan.jsonl"
    append_shadow_decision(
        shadow,
        ShadowDecision(
            decision_id=long_decision_id,
            timestamp_ms=1,
            venue="okx",
            symbol="BTCUSDT",
            model_name="edge",
            predicted_side=1,
            predicted_edge_bps=1.0,
            order_type="paper_limit",
            intended_price=65000.5,
            intended_size=0.01,
        ),
    )

    write_paper_order_plan(shadow_path=shadow, output_path=output, provider="okx")

    row = json.loads(output.read_text().splitlines()[0])
    expected_client_order_id = client_order_id_for_decision("okx", long_decision_id)
    assert row["decision_id"] == long_decision_id
    assert row["client_order_id"] == expected_client_order_id
    assert row["payload"]["clOrdId"] == expected_client_order_id
    assert expected_client_order_id != long_decision_id
    assert len(expected_client_order_id) == 32
    assert expected_client_order_id.isalnum()


def test_paper_order_plan_rejects_limit_order_without_price(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow.csv"
    append_shadow_decision(
        shadow,
        ShadowDecision(
            decision_id="bad",
            timestamp_ms=1,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="edge",
            predicted_side=1,
            predicted_edge_bps=1.0,
            order_type="paper_limit",
            intended_price=0.0,
            intended_size=0.01,
        ),
    )

    try:
        build_paper_order_plan(shadow, provider="bybit")
    except ValueError as exc:
        assert "requires a positive intended_price" in str(exc)
    else:
        raise AssertionError("expected limit decisions without a price to fail")


def _shadow_file(tmp_path: Path) -> Path:
    shadow = tmp_path / "shadow.csv"
    for decision in (
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="edge",
            predicted_side=1,
            predicted_edge_bps=1.2,
            order_type="paper_limit",
            intended_price=65000.5,
            intended_size=0.01,
        ),
        ShadowDecision(
            decision_id="d2",
            timestamp_ms=2,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="edge",
            predicted_side=-1,
            predicted_edge_bps=0.9,
            order_type="paper_taker",
            intended_price=65001.0,
            intended_size=0.02,
        ),
        ShadowDecision(
            decision_id="flat",
            timestamp_ms=3,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="edge",
            predicted_side=0,
            predicted_edge_bps=0.0,
            order_type="paper_taker",
            intended_price=65001.0,
            intended_size=0.02,
        ),
        ShadowDecision(
            decision_id="zero",
            timestamp_ms=4,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="edge",
            predicted_side=1,
            predicted_edge_bps=0.5,
            order_type="paper_taker",
            intended_price=65001.0,
            intended_size=0.0,
        ),
    ):
        append_shadow_decision(shadow, decision)
    return shadow
