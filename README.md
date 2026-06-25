# Microstructure Alpha Lab

Reproducible crypto microstructure research stack for the question:

> Does public crypto market microstructure data contain short-horizon predictive information that remains executable after causal timing, fees, latency, slippage, available liquidity, inventory constraints, adverse selection, and model-selection bias?

The current evidence should be read conservatively. The repository supports causal feature generation, validation-only model selection, frozen holdout manifests, stateful execution simulation, dependence-aware statistics, true-L2 replay checks, gated neural smoke paths with checkpoints/prediction exports, and a small C++ replay component. It does not claim executable alpha on the available local artifacts.

## What Is Implemented

- Causal event-time feature construction with explicit decision, entry, and exit timestamps.
- Chronological and purged walk-forward protocols with runtime guards against selecting by test metrics.
- Mandatory holdout manifests for CLI research/evidence commands; development runs materialize a content-hash-checked CSV with declared holdout rows physically removed.
- Stateful execution simulation with cash, inventory, equity, maker/taker fills, partial liquidity, fees, latency, expiry, position/leverage limits, ledgers, and a kill switch.
- Dependence-aware inference helpers: HAC/Newey-West, day-level bootstrap, moving/stationary block bootstrap, Sharpe-like and break-even-cost intervals.
- True-L2 schema, replay validation, normalized tensor path, TCN/Transformer CPU smoke tests with class weighting, early stopping, checkpoints, calibration metrics, prediction exports, stateful economic smoke fields, and a compact C++ L2 replay equivalence test.

## Current Evidence

Tracked docs and smoke artifacts support the cautious conclusion that local Binance quote/trade/depth-band samples show preliminary pre-cost predictability, but ordinary taker costs and adverse-selection assumptions dominate simple strategies. Full multi-month, multi-asset confirmatory runs and large repeated-seed crypto L2 neural experiments remain external-data/cloud-compute work.

The reduced E2E fixture declares the Jan 3 holdout before selection, runs walk-forward only on Jan 1-2 development rows, records the full threshold search family in `artifacts/reduced_e2e/experiment_registry.jsonl`, and feeds the validation-selected rule into the stateful simulator.

## Verify Locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,research]"
bash scripts/run_tests.sh
```

For a pinned full research environment, use `requirements-research.txt` instead of the range-based optional
dependencies:

```bash
.venv/bin/python -m pip install -r requirements-research.txt
```

Current fast verification in this checkout:

```text
passed 244 direct test functions
```

Reduced fixture-only artifacts:

```bash
.venv/bin/python scripts/run_reduced_e2e.py
```

This writes `artifacts/reduced_e2e/result_manifest.json` plus stateful simulation ledgers and smoke-result CSVs.

Optional heavier checks:

```bash
make external-readiness
MODE=plan make modal-study
MIN_AUDIT_FOLD_COUNT=20 bash scripts/verify_expected_edge_study.sh expected_edge_results
```

## Key Documents

- [Research note](docs/research_note.md)
- [Reproducibility](docs/reproducibility.md)
- [Traceability](IMPLEMENTATION_TRACEABILITY.md)
- [Data source reality](docs/data_source_reality.md)
- [Local free/freemium API source plan](docs/local_free_api_sources.md)
- [L2 replay](docs/l2_replay.md)
- [Paper/demo fill sources](docs/paper_demo_fill_sources.md)
- [Expected-edge methodology](docs/expected_edge.md)
- [Cloud runbook](docs/full_study_cloud_run.md)

## Repository Hygiene

Generated market data, results, artifacts, archives, local virtual environments, and private notes are ignored. The tracked repository is source, tests, scripts, fixtures, and documentation for independently reproducing the research workflow.
