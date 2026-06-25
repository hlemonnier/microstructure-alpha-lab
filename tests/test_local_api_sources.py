import json
from contextlib import redirect_stdout
from io import StringIO

from lob_forge.cli import main as cli_main
from lob_forge.local_api_sources import (
    PAPER_FILL_GAP,
    TRUE_L2_SMOKE_GAP,
    format_local_api_sources_csv,
    list_local_api_sources,
)


def test_paper_fill_sources_prioritize_crypto_demo_venues() -> None:
    sources = list_local_api_sources(data_gap=PAPER_FILL_GAP)

    assert [source.source_id for source in sources[:2]] == ["bybit_demo_fills", "okx_demo_fills"]
    assert sources[0].id_field == "orderLinkId"
    assert "execPrice" in sources[0].fill_fields
    assert "normalize-observed-fills --provider bybit" in sources[0].local_commands
    assert all(source.requires_api_key for source in sources)


def test_true_l2_smoke_sources_are_public_downloads_or_freemium_samples() -> None:
    sources = list_local_api_sources(data_gap=TRUE_L2_SMOKE_GAP)
    source_ids = [source.source_id for source in sources]

    assert source_ids == [
        "okx_public_historical_l2",
        "bybit_public_historical_l2",
        "tardis_free_csv_samples",
    ]
    assert not any(source.requires_account for source in sources)
    assert sources[0].local_commands[0] == "l2-manifest --source okx"
    assert "first-day-of-month" in sources[-1].limitations[0]


def test_local_api_sources_csv_includes_credentials_and_commands() -> None:
    csv_text = format_local_api_sources_csv(list_local_api_sources(data_gap=PAPER_FILL_GAP))

    assert "BYBIT_DEMO_API_KEY BYBIT_DEMO_API_SECRET" in csv_text
    assert "normalize-observed-fills --provider okx" in csv_text


def test_cli_free_api_sources_json_filters_by_gap() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout):
        exit_code = cli_main(["free-api-sources", "--gap", PAPER_FILL_GAP, "--format", "json"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert {row["data_gap"] for row in payload} == {PAPER_FILL_GAP}
    assert payload[0]["source_id"] == "bybit_demo_fills"
