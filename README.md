# Microstructure Alpha Lab

Reproducible crypto microstructure research stack for the question:

> Does public crypto market microstructure data contain short-horizon predictive information that remains executable after causal timing, fees, latency, slippage, available liquidity, inventory constraints, adverse selection, and model-selection bias?

The current evidence should be read conservatively. The repository supports causal feature generation, validation-only model selection, frozen holdout manifests, stateful execution simulation, dependence-aware statistics, true-L2 replay checks, gated neural smoke paths with manifest-filtered development L2 runs, repeated-seed and ablation dry-runs, checkpoints/prediction exports, frozen neural sequence candidates, one-way neural final-holdout evaluation, and a small C++ replay component. It does not claim executable alpha on the available local artifacts.

## What Is Implemented

- Causal event-time feature construction with explicit decision, entry, and exit timestamps.
- Chronological and purged walk-forward protocols with runtime guards against selecting by test metrics.
- Mandatory holdout manifests for CLI research/evidence commands; development runs materialize a content-hash-checked CSV with declared holdout rows physically removed.
- Stateful execution simulation with cash, signed inventory, equity, causal maker/taker fills, post-only crossing rejection, partial liquidity, fees, latency, expiry, position/leverage limits, ledgers, and a kill switch.
- Dependence-aware inference helpers: HAC/Newey-West, day-level bootstrap, moving/stationary block bootstrap, Sharpe-like and break-even-cost intervals.
- Content-addressed serious-study provenance from daily source/config markers through combined-feature manifests, immutable holdout hashes, planned split sizes, result/audit sidecars, and procedure-level multiple-testing records.
- True-L2 schema, replay validation, normalized tensor path, TCN/Transformer CPU smoke tests with required manifest-filtered development data for certifiable runs, repeated seeds, named ablations, class weighting, early stopping, contract-bound checkpoints, calibration metrics, prediction exports, stateful economic smoke fields, frozen sequence-candidate final holdout hooks, and a compact C++ L2 replay equivalence test.

## Current Evidence

The September 2026 model/mathematics audit found defects affecting timing, inference and execution. The implementation has been corrected and expanded with adversarial regressions; see [the finding-by-finding remediation](docs/model_math_remediation.md). Historical performance figures and artifacts predate these corrections and require regeneration before interpretation. Neither positive nor negative executable-alpha conclusions are established by those old results.

Feature manifests use the completed-bucket/raw-OFI contract; neural artifacts use version 3. Legacy expected-edge, Kelly and neural CSVs cannot establish current evidence. A new bounded performance pilot uses checksum-verified historical BTC/ETH quotes and trades; genuine observed trading fills remain unavailable. Masked mean imputation is an untrained reconstruction baseline, not learned pretraining.

The [performance pilot report](docs/research/performance_pilot_20260907.md) compares the original expected-payoff ridge, added market-state features, and fixed gradient boosting under the same raw-quote replay. Its three distinct test dates support descriptive research, with no empirical promotion. The [reproduction instructions](docs/research/performance_pilot_reproduction.md) preserve the frozen protocols, source hashes, checkpoints and complete fee/latency comparisons. The [research path](docs/research/performance_research_path.md) connects those comparisons to the next mechanism to test.

The reduced E2E fixture declares the Jan 3 holdout before selection, runs walk-forward only on Jan 1-2 development rows, records the full threshold search family in `artifacts/reduced_e2e/experiment_registry.jsonl`, and feeds the validation-selected rule into the stateful simulator. Full expected-edge registries now fail closed unless every result/audit pair has a sidecar binding it to the exact plan, verified feature inputs, holdout manifest, code fingerprint, and planned split sizes. Statistical correction uses one p-value per validation-selection procedure; it does not duplicate an artifact p-value across threshold candidates.

## Market Data (Not Included)

Downloaded market data is intentionally excluded from Git. A fresh clone may
therefore contain only [`data/README.md`](data/README.md); this is expected. The
source code, tests, and synthetic fixtures remain available, but empirical
studies must reacquire their external inputs before running.

Use `make laptop-smoke` for a bounded Binance sample or
`bash scripts/run_bybit_l2_smoke.sh` for a one-day, row-capped Bybit true-L2
sample. The data README documents the sources, storage layout, checksum
behavior, and deliberate high-resource workflow.

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

Verification commands and the scope of the new regressions are recorded in [the remediation report](docs/model_math_remediation.md). Research tests execute actual small CPU training when the pinned optional dependencies are installed.

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
