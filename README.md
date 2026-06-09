# Microstructure Alpha Lab

Research stack for testing short-horizon crypto microstructure signals under realistic validation, cost, latency, and execution assumptions.

The project starts from Binance Vision USD-M futures archives and treats the core question as an empirical one:

> Do quote, trade-flow, and depth-band features contain a signal that still has value after walk-forward validation, fees, spread, slippage, and latency?

This is not a trading bot. It is a reproducible research workflow for market-data ingestion, feature engineering, baseline modeling, expected-edge analysis, L2 replay checks, passive-fill diagnostics, and cloud-scale validation.

## What Is Inside

- Binance Vision archive download, schema inspection, and checksum-aware data handling.
- Quote/trade/depth-band feature generation for BTCUSDT and ETHUSDT USD-M futures.
- Baseline statistical models, calibration checks, and walk-forward validation.
- Fee, latency, capacity, regime, and passive execution diagnostics.
- Historical/live L2 adapters for deeper order-book replay experiments.
- Modal cloud runner for the full high-RAM expected-edge study.

## Quick Start

Create a virtual environment, install the package, then run the direct test suite:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,research,cloud]"
bash scripts/run_tests.sh
```

Expected local verification:

```text
passed 160 direct test functions
```

## Local Runs

Safe laptop smoke run:

```bash
make laptop-smoke
```

Verify compact result artifacts:

```bash
make verify-results
```

Check which evidence gates still need external data or cloud compute:

```bash
make external-readiness
make verify-evidence-gates
```

The full study is intentionally not a 16 GB laptop workload. Local profiles are for smoke tests and capped research only.

## Full Cloud Study

Authenticate Modal, dry-run the plan, then launch the full run:

```bash
.venv/bin/modal setup
make external-readiness
MODE=plan make modal-study
MODE=run make modal-study
```

Optional sequence-model experiments after the main run:

```bash
MODE=sequence make modal-study
```

Download Modal results:

```bash
.venv/bin/modal volume get \
  microstructure-alpha-lab-expected-edge \
  /results/expected_edge_60day_20230516_20230714 \
  ./expected_edge_results
```

Verify the downloaded study:

```bash
MIN_AUDIT_FOLD_COUNT=20 \
bash scripts/verify_expected_edge_study.sh expected_edge_results
```

## Documentation

- [Case study](docs/microstructure_case_study.md)
- [Research plan](docs/research_plan.md)
- [Data source reality](docs/data_source_reality.md)
- [Pipeline notes](docs/pipeline.md)
- [Expected-edge methodology](docs/expected_edge.md)
- [L2 data sources](docs/l2_data_sources.md)
- [L2 replay](docs/l2_replay.md)
- [Passive fill diagnostics](docs/passive_fill_diagnostics.md)
- [Cloud runbook](docs/full_study_cloud_run.md)
- [Modal checklist](docs/modal_full_run.md)
- [Implementation checklist](docs/implementation_todo.md)

## Repository Hygiene

Generated market data, results, archives, local virtual environments, and private notes are intentionally ignored. The repository is source, tests, scripts, and documentation only.
