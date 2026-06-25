from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from typing import Iterable


PAPER_FILL_GAP = "paper_live_fill_validation"
TRUE_L2_SMOKE_GAP = "true_l2_laptop_smoke"
QUOTE_TRADE_GAP = "quote_trade_depth_band_research"

REAL_SHADOW_FILL_GATE = "real_shadow_fill_validation"
PAPER_LIVE_FILL_GATE = "paper_live_fill_validation"
CAPPED_60DAY_GATE = "capped_60day_btc_eth"
KELLY_VARIANCE_GATE = "kelly_variance_stability"
SEQUENCE_MODEL_GATE = "sequence_transformer_tcn_experiments"
L2_PRETRAINING_GATE = "self_supervised_l2_pretraining"

LOCAL_EVIDENCE_GATES = (
    REAL_SHADOW_FILL_GATE,
    PAPER_LIVE_FILL_GATE,
    CAPPED_60DAY_GATE,
    KELLY_VARIANCE_GATE,
    SEQUENCE_MODEL_GATE,
    L2_PRETRAINING_GATE,
)


@dataclass(frozen=True)
class LocalApiSource:
    source_id: str
    name: str
    priority: int
    data_gap: str
    access_tier: str
    requires_account: bool
    requires_api_key: bool
    credential_env: tuple[str, ...]
    best_for: str
    not_for: str
    id_field: str
    fill_fields: tuple[str, ...]
    endpoints: tuple[str, ...]
    local_commands: tuple[str, ...]
    docs_urls: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_gates: tuple[str, ...] = ()
    target_artifacts: tuple[str, ...] = ()
    minimum_local_proof: str = ""


LOCAL_API_SOURCES: tuple[LocalApiSource, ...] = (
    LocalApiSource(
        source_id="bybit_demo_fills",
        name="Bybit Demo Trading",
        priority=1,
        data_gap=PAPER_FILL_GAP,
        access_tier="free_demo_account",
        requires_account=True,
        requires_api_key=True,
        credential_env=("BYBIT_DEMO_API_KEY", "BYBIT_DEMO_API_SECRET"),
        best_for="primary BTCUSDT/ETHUSDT crypto demo-fill observations with client order IDs",
        not_for="proof of live queue priority or hidden-liquidity behavior",
        id_field="orderLinkId",
        fill_fields=("execPrice", "execQty", "execPnl", "isMaker", "execType"),
        endpoints=("/v5/order/create", "/v5/execution/list", "wss://stream-demo.bybit.com/v5/private"),
        local_commands=(
            "fetch-observed-fills --provider bybit",
            "normalize-observed-fills --provider bybit",
        ),
        docs_urls=(
            "https://bybit-exchange.github.io/docs/v5/demo",
            "https://bybit-exchange.github.io/docs/v5/order/execution",
        ),
        limitations=("demo orders are retained for 7 days", "demo trading is not a live execution-quality claim"),
        evidence_gates=(REAL_SHADOW_FILL_GATE, PAPER_LIVE_FILL_GATE),
        target_artifacts=(
            "results/shadow_validation/raw_bybit_executions.json",
            "results/shadow_validation/observed_fills.csv",
            "results/shadow_validation/shadow_decisions_observed.csv",
        ),
        minimum_local_proof="non-empty normalized Bybit fills matched back to shadow decision_id/orderLinkId",
    ),
    LocalApiSource(
        source_id="okx_demo_fills",
        name="OKX Demo Trading",
        priority=2,
        data_gap=PAPER_FILL_GAP,
        access_tier="free_demo_account",
        requires_account=True,
        requires_api_key=True,
        credential_env=("OKX_DEMO_API_KEY", "OKX_DEMO_API_SECRET", "OKX_DEMO_API_PASSPHRASE"),
        best_for="cross-venue crypto demo-fill observations on instruments such as BTC-USDT-SWAP",
        not_for="single-venue proof that Bybit-specific passive assumptions generalize",
        id_field="clOrdId",
        fill_fields=("fillPx", "fillSz", "fillPnl", "execType"),
        endpoints=(
            "https://eea.okx.com",
            "wss://wseeapap.okx.com:8443/ws/v5/private",
            "/api/v5/trade/order",
            "/api/v5/trade/fills",
            "/api/v5/trade/fills-history",
            "demo private WebSocket order channel",
        ),
        local_commands=(
            "fetch-observed-fills --provider okx",
            "normalize-observed-fills --provider okx",
        ),
        docs_urls=("https://my.okx.com/docs-v5/en/",),
        limitations=(
            "requires x-simulated-trading: 1 with demo API credentials",
            "transaction-detail history is windowed",
        ),
        evidence_gates=(REAL_SHADOW_FILL_GATE, PAPER_LIVE_FILL_GATE),
        target_artifacts=(
            "results/shadow_validation/raw_okx_fills.json",
            "results/shadow_validation/observed_fills.csv",
            "results/shadow_validation/shadow_decisions_observed.csv",
        ),
        minimum_local_proof="non-empty normalized OKX fills matched back to shadow decision_id/clOrdId",
    ),
    LocalApiSource(
        source_id="binance_spot_testnet_fills",
        name="Binance Spot Testnet",
        priority=4,
        data_gap=PAPER_FILL_GAP,
        access_tier="free_testnet_account",
        requires_account=True,
        requires_api_key=True,
        credential_env=("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET"),
        best_for="spot-only order-event plumbing and importer regression coverage",
        not_for="derivatives, funding, or deep passive queue evidence",
        id_field="newClientOrderId",
        fill_fields=("executionReport.L", "executionReport.l", "fills[].price", "fills[].qty"),
        endpoints=("/api/v3/order", "user-data-stream executionReport"),
        local_commands=("normalize-observed-fills --provider binance",),
        docs_urls=(
            "https://developers.binance.com/docs/binance-spot-api-docs/testnet/user-data-stream",
            "https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/trading-endpoints",
        ),
        limitations=("spot testnet only", "not a futures L2 queue validation source"),
        evidence_gates=(REAL_SHADOW_FILL_GATE, PAPER_LIVE_FILL_GATE),
        target_artifacts=(
            "results/shadow_validation/raw_binance_execution_reports.json",
            "results/shadow_validation/observed_fills.csv",
        ),
        minimum_local_proof="spot testnet executionReport/FULL order rows normalize into canonical observed fills",
    ),
    LocalApiSource(
        source_id="binance_usdm_futures_testnet_fills",
        name="Binance USD-M Futures Testnet",
        priority=3,
        data_gap=PAPER_FILL_GAP,
        access_tier="free_testnet_account",
        requires_account=True,
        requires_api_key=True,
        credential_env=("BINANCE_FUTURES_TESTNET_API_KEY", "BINANCE_FUTURES_TESTNET_API_SECRET"),
        best_for="BTCUSDT/ETHUSDT futures testnet order-event plumbing with client order IDs",
        not_for="proof of live production queue priority, impact, or adverse-selection behavior",
        id_field="newClientOrderId",
        fill_fields=("ORDER_TRADE_UPDATE.o.L", "ORDER_TRADE_UPDATE.o.l", "ORDER_TRADE_UPDATE.o.rp"),
        endpoints=(
            "https://demo-fapi.binance.com",
            "wss://demo-fstream.binance.com",
            "/fapi/v1/order",
            "ORDER_TRADE_UPDATE user data stream",
        ),
        local_commands=("normalize-observed-fills --provider binance",),
        docs_urls=(
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info",
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api",
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/user-data-streams",
        ),
        limitations=("testnet matching is not production queue evidence", "requires futures testnet API credentials"),
        evidence_gates=(REAL_SHADOW_FILL_GATE, PAPER_LIVE_FILL_GATE),
        target_artifacts=(
            "results/shadow_validation/raw_binance_futures_order_trade_updates.jsonl",
            "results/shadow_validation/observed_fills.csv",
        ),
        minimum_local_proof="USD-M futures ORDER_TRADE_UPDATE rows normalize and match decision_id/newClientOrderId",
    ),
    LocalApiSource(
        source_id="alpaca_paper_fills",
        name="Alpaca Paper",
        priority=5,
        data_gap=PAPER_FILL_GAP,
        access_tier="free_paper_account",
        requires_account=True,
        requires_api_key=True,
        credential_env=("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
        best_for="generic order lifecycle, terminal no-fill rows, and importer sanity checks",
        not_for="crypto exchange L2 queue-position validation",
        id_field="client_order_id",
        fill_fields=("filled_avg_price", "filled_qty", "trade_updates.price", "trade_updates.qty"),
        endpoints=("/v2/orders", "trade_updates websocket"),
        local_commands=("normalize-observed-fills --provider alpaca",),
        docs_urls=("https://docs.alpaca.markets/us/docs/paper-trading",),
        limitations=("paper simulator omits queue position, market impact, information leakage, and latency slippage",),
        evidence_gates=(REAL_SHADOW_FILL_GATE, PAPER_LIVE_FILL_GATE),
        target_artifacts=(
            "results/shadow_validation/raw_alpaca_trade_updates.json",
            "results/shadow_validation/observed_fills.csv",
        ),
        minimum_local_proof="paper trade updates normalize and merge, but only as low-weight order-lifecycle evidence",
    ),
    LocalApiSource(
        source_id="okx_public_historical_l2",
        name="OKX Historical Market Data",
        priority=6,
        data_gap=TRUE_L2_SMOKE_GAP,
        access_tier="free_public_download",
        requires_account=False,
        requires_api_key=False,
        credential_env=(),
        best_for="bounded true-L2 laptop smoke imports and schema/replay validation",
        not_for="unattended full 60-90 day matrix on a 16 GB laptop",
        id_field="direct_url",
        fill_fields=(),
        endpoints=("historical-data download-link",),
        local_commands=(
            "l2-manifest --source okx",
            "l2-resolve-okx",
            "l2-download-manifest",
            "l2-import-manifest --max-import-rows",
        ),
        docs_urls=("https://www.okx.com/en-us/historical-data",),
        limitations=("download-link discovery is rate-limited and can legitimately return no file for a row",),
        evidence_gates=(SEQUENCE_MODEL_GATE, L2_PRETRAINING_GATE),
        target_artifacts=("data/normalized_l2/okx/BTC-USDT-SWAP/2023-05-16.csv",),
        minimum_local_proof="row-capped normalized OKX L2 CSV passes l2-import-manifest and l2-validate",
    ),
    LocalApiSource(
        source_id="bybit_public_historical_l2",
        name="Bybit Historical orderBook",
        priority=7,
        data_gap=TRUE_L2_SMOKE_GAP,
        access_tier="free_public_download",
        requires_account=False,
        requires_api_key=False,
        credential_env=(),
        best_for="Bybit true-L2 orderBook archives and row-capped normalized imports",
        not_for="filling the paper/live observed-fill gate without a demo/private API session",
        id_field="direct_url",
        fill_fields=(),
        endpoints=("history-data list-files", "/v5/market/orderbook"),
        local_commands=(
            "l2-manifest --source bybit",
            "l2-resolve-bybit",
            "l2-download-manifest",
            "l2-import-manifest --max-import-rows",
        ),
        docs_urls=(
            "https://www.bybit.com/en/derivative-activity/history-data/",
            "https://bybit-exchange.github.io/docs/v5/market/orderbook",
        ),
        limitations=("historical file availability varies by symbol/date",),
        evidence_gates=(SEQUENCE_MODEL_GATE, L2_PRETRAINING_GATE),
        target_artifacts=("data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv",),
        minimum_local_proof="row-capped normalized Bybit orderBook CSV preserves seq/u/cts and passes l2-validate",
    ),
    LocalApiSource(
        source_id="coinbase_public_level2",
        name="Coinbase Public Level2 WebSocket",
        priority=8,
        data_gap=TRUE_L2_SMOKE_GAP,
        access_tier="free_public_websocket",
        requires_account=False,
        requires_api_key=False,
        credential_env=(),
        best_for="no-key live L2 smoke capture when historical archives are unavailable",
        not_for="historical 60-90 day model evidence or exchange-specific Bybit/OKX fill validation",
        id_field="sequence_num",
        fill_fields=(),
        endpoints=("wss://advanced-trade-ws.coinbase.com level2",),
        local_commands=(
            "live-l2-capture --venue coinbase --symbol BTC-USD",
            "l2-validate --source coinbase",
        ),
        docs_urls=(
            "https://docs.cdp.coinbase.com/exchange/websocket-feed/overview",
            "https://docs.cdp.coinbase.com/exchange/websocket-feed/channels",
        ),
        limitations=("live-only for this project; no free historical backfill claim",),
        evidence_gates=(SEQUENCE_MODEL_GATE, L2_PRETRAINING_GATE),
        target_artifacts=("data/live_l2/coinbase/BTC-USD/session.csv",),
        minimum_local_proof="captured normalized level2 rows pass l2-validate with monitored sequence gaps",
    ),
    LocalApiSource(
        source_id="tardis_free_csv_samples",
        name="Tardis.dev Free CSV Samples",
        priority=9,
        data_gap=TRUE_L2_SMOKE_GAP,
        access_tier="freemium_no_key_monthly_samples",
        requires_account=False,
        requires_api_key=False,
        credential_env=(),
        best_for="cross-vendor L2 schema checks and first-day-of-month sample validation",
        not_for="arbitrary 60-90 day historical coverage without paid API access",
        id_field="download_url",
        fill_fields=(),
        endpoints=("downloadable CSV files",),
        local_commands=("l2-validate --source tardis",),
        docs_urls=(
            "https://docs.tardis.dev/downloadable-csv-files/overview",
            "https://docs.tardis.dev/historical-data-details/bybit",
        ),
        limitations=("only first-day-of-month historical CSV samples are available without an API key",),
        evidence_gates=(SEQUENCE_MODEL_GATE, L2_PRETRAINING_GATE),
        target_artifacts=("data/normalized_l2/tardis/{exchange}/{symbol}/{sample-date}.csv",),
        minimum_local_proof="sample incremental_book_L2 rows validate as schema/replay sanity data; not arbitrary history",
    ),
    LocalApiSource(
        source_id="crypto_lake_free_samples",
        name="Crypto Lake Free Data",
        priority=10,
        data_gap=TRUE_L2_SMOKE_GAP,
        access_tier="freemium_python_api",
        requires_account=False,
        requires_api_key=False,
        credential_env=(),
        best_for="extra 20-level book/trade sample corpus for L2 parser and tensor sanity checks",
        not_for="main replay-grade claim until source schema and sampling limits are inspected",
        id_field="lakeapi_query",
        fill_fields=(),
        endpoints=("lakeapi free data",),
        local_commands=("l2-validate --source crypto_lake",),
        docs_urls=("https://crypto-lake.com/free-data/", "https://crypto-lake.com/data/"),
        limitations=("free coverage is a sample corpus; inspect dates/symbols before treating it as study coverage",),
        evidence_gates=(SEQUENCE_MODEL_GATE, L2_PRETRAINING_GATE),
        target_artifacts=("data/normalized_l2/crypto_lake/{exchange}/{symbol}/{sample-date}.csv",),
        minimum_local_proof="book_delta_v2 or compatible rows normalize and pass l2-validate on a bounded sample",
    ),
    LocalApiSource(
        source_id="binance_public_archives",
        name="Binance Public Data Archives",
        priority=11,
        data_gap=QUOTE_TRADE_GAP,
        access_tier="free_public_archive",
        requires_account=False,
        requires_api_key=False,
        credential_env=(),
        best_for="quote/trade/depth-band feature research with deterministic daily/monthly ZIPs",
        not_for="historical full level-by-level L2 replay",
        id_field="archive_key",
        fill_fields=(),
        endpoints=("https://data.binance.vision/data/",),
        local_commands=("download-range", "build-range"),
        docs_urls=("https://github.com/binance/binance-public-data", "https://data.binance.vision/"),
        limitations=("bookDepth archives are aggregate percentage-depth bands, not full price-level L2",),
        evidence_gates=(CAPPED_60DAY_GATE, KELLY_VARIANCE_GATE),
        target_artifacts=(
            "results/expected_edge_local16_20230516_20230714/",
            "results/kelly_candidate_search/",
        ),
        minimum_local_proof="bounded local16 quote/trade/depth-band study artifacts pass audit and variance gates",
    ),
)


def list_local_api_sources(
    *,
    data_gap: str | None = None,
    evidence_gate: str | None = None,
) -> list[LocalApiSource]:
    sources = [
        source
        for source in LOCAL_API_SOURCES
        if (data_gap is None or source.data_gap == data_gap)
        and (evidence_gate is None or evidence_gate in source.evidence_gates)
    ]
    return sorted(sources, key=lambda source: source.priority)


def format_local_api_sources_csv(sources: Iterable[LocalApiSource]) -> str:
    fields = [
        "priority",
        "source_id",
        "name",
        "data_gap",
        "access_tier",
        "requires_account",
        "requires_api_key",
        "credential_env",
        "id_field",
        "best_for",
        "not_for",
        "local_commands",
        "docs_urls",
        "limitations",
        "evidence_gates",
        "target_artifacts",
        "minimum_local_proof",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for source in sources:
        writer.writerow(
            {
                "priority": source.priority,
                "source_id": source.source_id,
                "name": source.name,
                "data_gap": source.data_gap,
                "access_tier": source.access_tier,
                "requires_account": int(source.requires_account),
                "requires_api_key": int(source.requires_api_key),
                "credential_env": " ".join(source.credential_env),
                "id_field": source.id_field,
                "best_for": source.best_for,
                "not_for": source.not_for,
                "local_commands": " | ".join(source.local_commands),
                "docs_urls": " ".join(source.docs_urls),
                "limitations": " | ".join(source.limitations),
                "evidence_gates": " ".join(source.evidence_gates),
                "target_artifacts": " | ".join(source.target_artifacts),
                "minimum_local_proof": source.minimum_local_proof,
            }
        )
    return handle.getvalue().strip("\r\n")


def format_local_api_sources_json(sources: Iterable[LocalApiSource]) -> str:
    return json.dumps([asdict(source) for source in sources], indent=2, sort_keys=True)


def format_local_api_sources_markdown(sources: Iterable[LocalApiSource]) -> str:
    lines = [
        "| Priority | Source | Gap | Evidence Gate | Access | API Key | Use | Target Artifact | Main Limitation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in sources:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(source.priority),
                    source.name,
                    source.data_gap,
                    ", ".join(source.evidence_gates),
                    source.access_tier,
                    "yes" if source.requires_api_key else "no",
                    source.best_for,
                    source.target_artifacts[0] if source.target_artifacts else "",
                    source.limitations[0] if source.limitations else "",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
