# Implementation Traceability

Authoritative scope: local implementation specification supplied outside the repository. Status terms: `implemented`, `implemented-smoke-tested`, `empirically-pending`, or `not-applicable`.

| Requirement | Status | Files | Tests / Evidence | Blocker |
|---|---:|---|---|---|
| A. Causal event-time construction | implemented | `src/lob_forge/features.py`, `src/lob_forge/protocol.py` | `tests/test_features.py`, `tests/test_protocol.py`, `bash scripts/run_tests.sh` | None for local fixture path |
| B. Leakage-free preprocessing and sequence splits | implemented | `src/lob_forge/protocol.py`, `src/lob_forge/logistic.py`, `src/lob_forge/ml_models.py` | `tests/test_protocol.py`, `tests/test_ml_models.py` | None for local path |
| C. Strict model-selection protocol | implemented | `src/lob_forge/protocol.py`, `src/lob_forge/baselines.py`, `src/lob_forge/logistic.py`, `src/lob_forge/edge_model.py`, `src/lob_forge/cli.py` | `tests/test_protocol.py`, `tests/test_baselines.py`; search for CLI `test_*` sort choices | None |
| D. Real untouched calendar holdout | implemented-smoke-tested | `src/lob_forge/holdout.py`, `src/lob_forge/cli.py`, `scripts/run_reduced_e2e.py` | `tests/test_holdout.py`; reduced E2E declares holdout before selection and filters development rows; `final-holdout-rule` consumes a frozen candidate | Final empirical holdout artifact pending a declared full run |
| E. Stateful execution simulator | implemented-smoke-tested | `src/lob_forge/execution_sim.py`, `src/lob_forge/portfolio.py`, `scripts/run_reduced_e2e.py` | `tests/test_execution_sim.py`, `tests/test_portfolio.py`; reduced E2E feeds selected rule signals into simulator ledgers | Full strategy reports must be regenerated from simulator ledgers |
| F. Execution realism, capacity, adverse selection | implemented-smoke-tested | `src/lob_forge/execution_sim.py`, `src/lob_forge/fill_diagnostics.py`, `src/lob_forge/capacity.py`, `src/lob_forge/passive_capacity.py` | `tests/test_execution_sim.py`, `tests/test_fill_diagnostics.py`, `tests/test_capacity.py`, `tests/test_passive_capacity.py` | Paper/live fill validation remains external |
| G. Statistical validity | implemented-smoke-tested | `src/lob_forge/statistics.py`, `src/lob_forge/alpha_factory.py`, `src/lob_forge/experiment_registry.py` | `tests/test_statistics.py`, `tests/test_alpha_factory.py`, `tests/test_experiment_registry.py` | Selection-adjusted exact inference for large search families remains a documented limitation |
| H. Baselines and neural models | implemented-smoke-tested | `src/lob_forge/baselines.py`, `src/lob_forge/logistic.py`, `src/lob_forge/edge_model.py`, `src/lob_forge/ml_models.py`, `artifacts/reduced_e2e/sequence_tcn_smoke.csv`, `artifacts/reduced_e2e/sequence_transformer_smoke.csv` | `tests/test_baselines.py`, `tests/test_logistic.py`, `tests/test_edge_model.py`, `tests/test_ml_models.py`; CPU TCN/Transformer fixture runs with minibatches, early stopping, checkpoints, prediction exports, calibration metrics, and stateful economic smoke fields | Large repeated-seed/GPU empirical runs pending compute and data |
| I. Genuine L2 data path | implemented-smoke-tested | `src/lob_forge/data_sources.py`, `src/lob_forge/l2_ingest.py`, `src/lob_forge/l2_replay.py`, `src/lob_forge/l2_storage.py`, `examples/fixtures/l2_replay_fixture.csv` | `tests/test_data_sources.py`, `tests/test_l2_ingest.py`, `tests/test_l2_replay.py`, `tests/test_ml_models.py` | Multi-month real L2 coverage pending data acquisition |
| J. Research artifacts and reporting | implemented | `README.md`, `docs/research_note.md`, `docs/reproducibility.md`, `artifacts/reduced_e2e/result_manifest.json`, `artifacts/reduced_e2e/experiment_registry.jsonl` | `.venv/bin/python scripts/run_reduced_e2e.py`; manual inspection plus final search checks | Final empirical result table pending heavy runs |
| K. Repository quality and application readiness | implemented-smoke-tested | `.gitignore`, `.github/workflows/ci.yml`, `requirements-ci.txt`, docs | `bash scripts/run_tests.sh`; hygiene searches | Optional pytest/ruff/mypy require installing dev deps |
| L. Focused C++ production signal | implemented-smoke-tested | `cpp/l2_replay.cpp`, `examples/fixtures/l2_replay_fixture.csv` | `tests/test_cpp_l2_replay.py`; compiles when `c++` is available | No Python extension needed for this scope |

## Verification Log

- Baseline direct tests before edits: `passed 160 direct test functions`.
- Current direct tests after implementation: `passed 217 direct test functions`.
- C++ replay equivalence is included in the direct suite when a local C++ compiler is present.
- Reduced E2E fixture artifacts generated under `artifacts/reduced_e2e/`.
- CPU neural fixture smokes generated `sequence_tcn_smoke.csv` and `sequence_transformer_smoke.csv`; sequence runs now expose checkpoint/resume, repeated-seed runner support, prediction CSV export, calibration metrics, confusion matrices, and stateful economic smoke fields.
- Full cloud/full-data empirical runs were not executed in this local turn.

## Empirical Work Still Pending

- Full multi-month, multi-asset confirmatory run.
- Immutable final holdout evaluation artifact.
- Genuine crypto L2 neural experiments over sufficient data.
- Large repeated-seed stochastic model sweeps over genuine L2 data.
- Paper/live fill validation against observed fills.
