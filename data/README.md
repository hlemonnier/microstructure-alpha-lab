# External Market Data

This repository does not include downloaded market data. The `data/` tree is
local, reproducible input state and is ignored by Git except for this file.
Research commands that require empirical data will fail or remain unavailable
until the relevant inputs are downloaded again.

## Laptop-safe reconstruction

Install the research environment first as described in the root README. Then
download and build a bounded Binance USD-M futures sample:

```bash
make laptop-smoke
```

This uses the repository's Binance Vision downloader, stores archives below
`data/raw`, and verifies Binance-published SHA-256 checksums by default. The
authoritative public archive is:

- <https://data.binance.vision/data/>

For a one-day Bybit historical true-L2 smoke sample with row-capped
normalization:

```bash
bash scripts/run_bybit_l2_smoke.sh
```

The manifest-driven download and import flow is documented in
[`docs/historical_l2_sources.md`](../docs/historical_l2_sources.md). Other
supported historical or live sources, including OKX, Binance, Coinbase,
Tardis.dev, Crypto Lake, and FI-2010, are described in
[`docs/l2_data_sources.md`](../docs/l2_data_sources.md).

## Full research dataset

The five-symbol, 60-day study is intentionally not the default laptop path.
Review [`docs/full_study_cloud_run.md`](../docs/full_study_cloud_run.md) before
accepting its disk, network, memory, and runtime cost. The explicit cloud
workflow is:

```bash
CONFIRM_HEAVY=1 STUDY_PROFILE=cloud_full \
  bash scripts/run_60day_expected_edge_study.sh
```

Generated raw, normalized, processed, result, and artifact files must remain
uncommitted. Preserve their manifests and provenance sidecars with any study
output that is promoted as evidence.
