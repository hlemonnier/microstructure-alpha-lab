from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lob_forge.live_validation import ShadowDecision, append_shadow_decision
from lob_forge.paper_orders import write_paper_order_plan
from lob_forge.paper_order_submit import (
    format_paper_order_submission_report,
    submit_paper_order_plan,
)


def test_submit_paper_orders_defaults_to_dry_run_without_credentials(tmp_path: Path) -> None:
    plan = tmp_path / "bybit_order_plan.jsonl"
    output = tmp_path / "submitted.jsonl"
    write_paper_order_plan(
        shadow_path=_shadow_file(tmp_path),
        output_path=plan,
        provider="bybit",
        limit=1,
    )

    report = submit_paper_order_plan(
        provider="bybit",
        plan_path=plan,
        output_path=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report.dry_run is True
    assert report.input_rows == 1
    assert report.order_rows == 1
    assert report.submitted_rows == 0
    assert rows[0]["dry_run"] is True
    assert rows[0]["decision_id"] == "d1"
    assert rows[0]["endpoint"] == "/v5/order/create"


def test_submit_bybit_demo_order_signs_body_and_writes_response(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str], str | None, float]] = []
    plan = tmp_path / "bybit_order_plan.jsonl"
    output = tmp_path / "submitted.jsonl"
    write_paper_order_plan(
        shadow_path=_shadow_file(tmp_path),
        output_path=plan,
        provider="bybit",
        limit=1,
    )

    def fake_post(url: str, headers: dict[str, str], body: str | None, timeout: float) -> dict:
        calls.append((url, dict(headers), body, timeout))
        return {"retCode": 0, "result": {"orderLinkId": "d1", "orderId": "oid"}}

    report = submit_paper_order_plan(
        provider="bybit",
        plan_path=plan,
        output_path=output,
        execute=True,
        env={"BYBIT_DEMO_API_KEY": "key", "BYBIT_DEMO_API_SECRET": "secret"},
        http_post_json=fake_post,
        now_ms=1711420489915,
    )

    url, headers, body, timeout = calls[0]
    expected_signature = hmac.new(
        b"secret",
        f"1711420489915key5000{body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    response_row = json.loads(output.read_text().splitlines()[0])
    assert report.dry_run is False
    assert report.submitted_rows == 1
    assert url == "https://api-demo.bybit.com/v5/order/create"
    assert headers["X-BAPI-API-KEY"] == "key"
    assert headers["X-BAPI-SIGN"] == expected_signature
    assert timeout == 30.0
    assert json.loads(body or "{}")["orderLinkId"] == "d1"
    assert response_row["response"]["result"]["orderId"] == "oid"


def test_submit_okx_demo_order_uses_simulated_header(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str], str | None, float]] = []
    plan = tmp_path / "okx_order_plan.jsonl"
    output = tmp_path / "submitted.jsonl"
    write_paper_order_plan(
        shadow_path=_shadow_file(tmp_path),
        output_path=plan,
        provider="okx",
        limit=1,
    )

    def fake_post(url: str, headers: dict[str, str], body: str | None, timeout: float) -> dict:
        calls.append((url, dict(headers), body, timeout))
        return {"code": "0", "data": [{"clOrdId": "d1", "ordId": "oid"}]}

    submit_paper_order_plan(
        provider="okx",
        plan_path=plan,
        output_path=output,
        execute=True,
        env={
            "OKX_DEMO_API_KEY": "key",
            "OKX_DEMO_API_SECRET": "secret",
            "OKX_DEMO_API_PASSPHRASE": "passphrase",
        },
        http_post_json=fake_post,
        now_ms=1711420489915,
    )

    url, headers, body, _timeout = calls[0]
    expected_signature = base64.b64encode(
        hmac.new(
            b"secret",
            (headers["OK-ACCESS-TIMESTAMP"] + "POST" + "/api/v5/trade/order" + (body or "")).encode(),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")
    assert url == "https://eea.okx.com/api/v5/trade/order"
    assert headers["x-simulated-trading"] == "1"
    assert headers["OK-ACCESS-SIGN"] == expected_signature
    assert json.loads(body or "{}")["clOrdId"] == "d1"


def test_submit_binance_usdm_testnet_order_signs_query(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str], str | None, float]] = []
    plan = tmp_path / "binance_plan.csv"
    output = tmp_path / "submitted.jsonl"
    write_paper_order_plan(
        shadow_path=_shadow_file(tmp_path),
        output_path=plan,
        provider="binance",
        limit=1,
        output_format="csv",
    )

    def fake_post(url: str, headers: dict[str, str], body: str | None, timeout: float) -> dict:
        calls.append((url, dict(headers), body, timeout))
        return {"clientOrderId": "d1", "orderId": 123}

    report = submit_paper_order_plan(
        provider="binance",
        plan_path=plan,
        output_path=output,
        execute=True,
        env={
            "BINANCE_USDM_TESTNET_API_KEY": "key",
            "BINANCE_USDM_TESTNET_API_SECRET": "secret",
        },
        http_post_json=fake_post,
        now_ms=1711420489915,
    )

    url, headers, body, _timeout = calls[0]
    parsed = urlparse(url)
    query_values = parse_qs(parsed.query)
    unsigned_query = parsed.query.rsplit("&signature=", 1)[0]
    expected_signature = hmac.new(b"secret", unsigned_query.encode(), hashlib.sha256).hexdigest()
    assert report.submitted_rows == 1
    assert url.startswith("https://demo-fapi.binance.com/fapi/v1/order?")
    assert query_values["newClientOrderId"] == ["d1"]
    assert query_values["timestamp"] == ["1711420489915"]
    assert query_values["signature"] == [expected_signature]
    assert headers["X-MBX-APIKEY"] == "key"
    assert body is None


def test_submit_paper_orders_rejects_wrong_provider_plan(tmp_path: Path) -> None:
    plan = tmp_path / "bybit_order_plan.jsonl"
    output = tmp_path / "submitted.jsonl"
    write_paper_order_plan(
        shadow_path=_shadow_file(tmp_path),
        output_path=plan,
        provider="bybit",
        limit=1,
    )

    try:
        submit_paper_order_plan(provider="okx", plan_path=plan, output_path=output)
    except ValueError as exc:
        assert "expected provider=okx" in str(exc)
    else:
        raise AssertionError("expected provider mismatch failure")


def test_submit_paper_orders_report_formats_csv(tmp_path: Path) -> None:
    plan = tmp_path / "bybit_order_plan.jsonl"
    output = tmp_path / "submitted.jsonl"
    write_paper_order_plan(
        shadow_path=_shadow_file(tmp_path),
        output_path=plan,
        provider="bybit",
        limit=1,
    )
    report = submit_paper_order_plan(provider="bybit", plan_path=plan, output_path=output)

    csv_text = format_paper_order_submission_report(report, output_format="csv")

    assert "provider,output_path,dry_run,input_rows,order_rows,submitted_rows,base_url" in csv_text
    assert "bybit" in csv_text


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
    ):
        append_shadow_decision(shadow, decision)
    return shadow
