# Microstructure Alpha Lab

[![CI](https://github.com/hlemonnier/microstructure-alpha-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hlemonnier/microstructure-alpha-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Crypto market microstructure research on a practical question: **does a short-horizon signal survive the cost of executing it?**

The Python toolkit connects causal quote and trade features to chronological model selection, a stateful execution simulator and provenance checks. A small C++ component cross-checks order-book replay. Spread, fees, latency, available liquidity, inventory and adverse selection are part of the research question.

**Research status:** experiments have been paused since September 2026. The repository contains exploratory results and reproducible software checks; it does not establish independently confirmed executable alpha. The [research summary](docs/research/README.md) explains the findings and the [pause record](docs/research/boundary_research_pause_20260909.md) preserves the interrupted experiment.

## Quick start

Python 3.9 or newer is required; Python 3.11 or 3.12 is recommended. The core package has no third-party runtime dependencies.

```bash
git clone https://github.com/hlemonnier/microstructure-alpha-lab.git
cd microstructure-alpha-lab
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/microstructure-alpha-lab --help
make e2e PYTHON=.venv/bin/python
```

The E2E command builds source and wheel distributions, installs the wheel in a fresh environment, runs the synthetic research pipeline, verifies its artifacts and compares Python with C++ replay. It installs pinned build tools from PyPI. It uses committed fixtures, with no market-data downloads, credentials or cloud jobs. A C++17 compiler, `make`, `bash`, `rsync` and `zip` are needed for the complete check.

Each run writes a separate directory under `artifacts/public_e2e/`, containing `evidence.json`, checksums, logs, source packages and synthetic result ledgers. Run it from a clean, committed checkout so the evidence identifies the exact source revision. See [reproducibility](docs/reproducibility.md) for optional model dependencies and advanced workflows.

## Research workflow

1. Check what the source actually records and when each observation becomes available.
2. Build causal features and declare an untouched holdout before model selection.
3. Compare complete procedures on chronological development splits, with purging and validation-only selection.
4. Simulate orders, fills, cash and inventory under explicit execution assumptions.
5. Bind inputs, configuration, checkpoints and outputs to hashes before interpreting results.

| Component | Purpose | Entry point |
|---|---|---|
| Features and timing | Completed buckets, raw flow and explicit decision/entry/exit clocks | [Pipeline](docs/pipeline.md) |
| Selection and holdouts | Chronological splits, frozen candidates and one-way final evaluation | [Methodology](docs/research_note.md) |
| Execution | Maker/taker orders, partial fills, fees, latency and risk limits | [Expected edge](docs/expected_edge.md) |
| Inference | Dependence-aware intervals and procedure-level multiple-testing records | [Statistical and model contracts](docs/model_math_remediation.md) |
| L2 replay | Snapshot/delta validation, gap detection and Python/C++ parity | [Replay contract](docs/l2_replay.md) |

Optional research dependencies support ridge, tree and sequence-model experiments. Neural fixture runs demonstrate pipeline wiring; empirical claims require sufficient replay-grade data and separate validation.

## Data and evidence

Market data, trained checkpoints and generated results are kept outside Git. A fresh clone includes only synthetic CSV fixtures and the [data acquisition guide](data/README.md).

Binance Vision `bookTicker` and trade archives support quote/trade research. Its `bookDepth` files contain percentage-band aggregate depth; genuine L2 replay requires snapshot/delta data from an appropriate source. Read [data source reality](docs/data_source_reality.md) before selecting a dataset or interpreting a model.

Software verification and empirical validation have separate requirements:

| Evidence | Current scope |
|---|---|
| Synthetic E2E | Reproducible checks of holdout filtering, selection, simulation, packaging and replay |
| Historical studies | Exploratory comparisons; cohorts, controls and limitations are stated in each report |
| Independent confirmation | The substantial-gain requirement has not been met |
| Executable alpha | Unestablished; full confirmatory economics and observed-fill validation remain pending |
| Midpoint conversion study | Interrupted before assessment; partial fits have no reported predictive result |

Historical protocols and evidence inventories remain at their original paths to preserve their hashes. Older implementation counts and artifacts are dated records, not the current verification result. `make verify-evidence-gates` is an empirical audit and may fail when required studies are missing; a green CI run does not close those gates.

## Repository guide

```text
src/lob_forge/       Research toolkit and command-line interface
cpp/                C++17 L2 replay reference
examples/fixtures/  Small synthetic inputs
scripts/            Verification, data acquisition and study runners
tests/              Existing implementation regression suite
docs/               Methodology and runbooks
docs/research/      Dated protocols, findings and evidence inventories
```

Start with the [documentation index](docs/README.md), [research summary](docs/research/README.md) or [contribution guide](CONTRIBUTING.md). The [implementation traceability](IMPLEMENTATION_TRACEABILITY.md) records the historical remediation work.

## License and citation

The code and repository documentation are available under the [MIT License](LICENSE). External datasets and third-party dependencies retain their own terms. Citation metadata is provided in [CITATION.cff](CITATION.cff).
