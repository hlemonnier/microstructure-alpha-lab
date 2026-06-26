import json
from contextlib import redirect_stdout
from io import StringIO

from lob_forge.cli import main as cli_main
from lob_forge.local_api_sources import (
    L2_PRETRAINING_GATE,
    PAPER_FILL_GAP,
    REAL_SHADOW_FILL_GATE,
    SEQUENCE_MODEL_GATE,
    TRUE_L2_SMOKE_GAP,
    format_local_api_sources_csv,
    list_local_api_sources,
)


def test_paper_fill_sources_prioritize_crypto_demo_venues() -> None:
    sources = list_local_api_sources(data_gap=PAPER_FILL_GAP)

    assert [source.source_id for source in sources[:3]] == [
        "bybit_demo_fills",
        "okx_demo_fills",
        "binance_usdm_futures_testnet_fills",
    ]
    assert sources[0].id_field == "orderLinkId"
    assert "execPrice" in sources[0].fill_fields
    assert "fetch-observed-fills --provider bybit" in sources[0].local_commands
    assert "submit-paper-orders --provider bybit --execute" in sources[0].local_commands
    assert "normalize-observed-fills --provider bybit" in sources[0].local_commands
    assert REAL_SHADOW_FILL_GATE in sources[0].evidence_gates
    assert "raw_bybit_executions.json" in sources[0].target_artifacts[0]
    assert all(source.requires_api_key for source in sources)


def test_true_l2_smoke_sources_are_public_downloads_or_freemium_samples() -> None:
    sources = list_local_api_sources(data_gap=TRUE_L2_SMOKE_GAP)
    source_ids = [source.source_id for source in sources]

    assert source_ids == [
        "okx_public_historical_l2",
        "bybit_public_historical_l2",
        "coinbase_public_level2",
        "tardis_free_csv_samples",
        "crypto_lake_free_samples",
    ]
    assert not any(source.requires_account for source in sources)
    assert sources[0].local_commands[0] == "l2-manifest --source okx"
    assert "live-l2-capture --venue coinbase" in sources[2].local_commands[0]
    assert "first-day-of-month" in sources[3].limitations[0]
    assert L2_PRETRAINING_GATE in sources[-1].evidence_gates


def test_local_api_sources_csv_includes_credentials_and_commands() -> None:
    csv_text = format_local_api_sources_csv(list_local_api_sources(data_gap=PAPER_FILL_GAP))

    assert "BYBIT_DEMO_API_KEY BYBIT_DEMO_API_SECRET" in csv_text
    assert "BINANCE_USDM_TESTNET_API_KEY BINANCE_USDM_TESTNET_API_SECRET" in csv_text
    assert "paper-order-plan --provider okx [--symbol-override BTC-USDT-SWAP]" in csv_text
    assert "submit-paper-orders --provider okx --execute" in csv_text
    assert "fetch-observed-fills --provider okx" in csv_text
    assert "fetch-observed-fills --provider binance --symbol BTCUSDT" in csv_text
    assert "normalize-observed-fills --provider okx" in csv_text
    assert "minimum_local_proof" in csv_text


def test_local_api_sources_filter_by_evidence_gate() -> None:
    fill_sources = list_local_api_sources(evidence_gate=REAL_SHADOW_FILL_GATE)
    l2_sources = list_local_api_sources(evidence_gate=SEQUENCE_MODEL_GATE)

    assert [source.source_id for source in fill_sources[:2]] == ["bybit_demo_fills", "okx_demo_fills"]
    assert "binance_usdm_futures_testnet_fills" in {source.source_id for source in fill_sources}
    assert "binance_public_archives" not in {source.source_id for source in fill_sources}
    assert {source.data_gap for source in l2_sources} == {TRUE_L2_SMOKE_GAP}
    assert "coinbase_public_level2" in {source.source_id for source in l2_sources}


def test_cli_free_api_sources_json_filters_by_gap() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout):
        exit_code = cli_main(["free-api-sources", "--gap", PAPER_FILL_GAP, "--format", "json"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert {row["data_gap"] for row in payload} == {PAPER_FILL_GAP}
    assert payload[0]["source_id"] == "bybit_demo_fills"
    assert "target_artifacts" in payload[0]


def test_cli_free_api_sources_json_filters_by_evidence_gate() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout):
        exit_code = cli_main(["free-api-sources", "--evidence-gate", SEQUENCE_MODEL_GATE, "--format", "json"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert {row["data_gap"] for row in payload} == {TRUE_L2_SMOKE_GAP}
    assert payload[0]["source_id"] == "okx_public_historical_l2"
    assert SEQUENCE_MODEL_GATE in payload[0]["evidence_gates"]
