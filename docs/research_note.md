# Research Note: Crypto Microstructure Predictability Under Execution Constraints

## 1. Question And Thesis

This project studies a narrow but important question:

> Does public crypto market microstructure data contain short-horizon predictive information that remains executable after causal timing, fees, latency, slippage, available liquidity, inventory constraints, adverse selection, and model-selection bias?

The objective is scientific defensibility, not a forced positive result. A negative result is useful if it is produced by a pipeline that would also be capable of detecting a real signal. The working hypothesis is that public quote/trade/depth data can show preliminary pre-cost predictability at short horizons, but that the edge is fragile and likely sub-basis-point after realistic execution assumptions. The repository therefore treats forecast quality, gross diagnostic edge, stateful executable simulation, and untouched holdout evidence as separate categories.

Current local evidence does not support a claim of executable alpha. The reduced tracked artifacts are synthetic fixtures for pipeline verification only. Older local Binance quote/trade/depth-band studies are documented as preliminary and pre-cost or cost-fragile. Full confirmatory claims remain pending multi-month data, genuine crypto L2 coverage, repeated stochastic runs, and an immutable final holdout artifact.

## 2. Data Scope

The lightweight local path starts from Binance Vision USD-M futures archives:

- `bookTicker` for best bid/ask quotes and executable top-of-book timestamps;
- `aggTrades` or `trades` for signed flow and trade intensity;
- `bookDepth` percentage-band depth files for liquidity context.

The Binance `bookDepth` percentage-band files are not treated as genuine multi-level L2 books. They can support regime features such as broad liquidity imbalance, but they are not suitable for deterministic order-book replay or DeepLOB-style tensor claims.

The true-L2 path uses normalized snapshot/delta rows with:

- `event_type`: snapshot or delta;
- `exchange_timestamp` and optional `local_timestamp`;
- `side`, `price`, `size`;
- `sequence` and/or `update_id`;
- venue and symbol metadata.

The committed fixtures under `examples/fixtures/` are sanitized synthetic data for CI and reduced smoke runs. They are not empirical trading evidence. Real L2 experiments should use the OKX/Bybit acquisition path and must pass sequence-gap, duplicate-update, crossed-book, and replay validation before being used for model claims.

## 3. Causal Timing Contract

Each sample is defined by a decision timestamp. Feature visibility is restricted to events at or before that timestamp. In the main quote/trade CSV builder, the default entry and exit are resolved from the first raw `bookTicker` quote event at or after the latency-adjusted target, while the feature state remains cut off at the retained decision quote. A `--execution-quote-resolution bucket` fallback exists for reproducing older bucket-retained studies. The feature CSV retains decision, quote, entry, and future timestamps so downstream code does not need to infer timing from bucket names.

```text
events <= decision_time       decision_time + latency       decision_time + latency + horizon
        |                                  |                                |
        v                                  v                                v
   feature state                    executable entry                 executable exit
```

Important details:

- same-bucket trade aggregation is causal and only includes trades whose exchange timestamp is at or before the decision timestamp;
- depth features use the latest depth snapshot at or before decision time;
- default entry/exit use raw quote event timestamps, not bucket labels;
- support exists for receive-time sampling where local timestamps are available;
- adversarial tests mutate events after decision time and assert unchanged feature inputs.

This matters because short-horizon microstructure research is highly sensitive to tiny timing mistakes. A model can appear predictive if a bucket is labeled with an early timestamp while containing later quote/trade events. The current feature builder avoids that specific leakage mode and now uses the stricter first-raw-executable quote contract by default.

## 4. Split, Selection, And Holdout Protocol

Model selection is restricted to training and validation data. Runtime guards reject selection metrics beginning with `test_`; test metrics remain report-only. The CLI also removes `test_*` objectives from selection choices. This applies to threshold rules, fee sweeps, logistic paths, ridge expected-edge paths, and shadow-decision export.

Chronological splits support purging, sequence overlap buffers, latency buffers, and embargoes. The `protocol` module includes row-index split helpers and disjoint-partition checks. Sequence-model normalization is fit on the training slice only and applied unchanged to validation/test slices.

Holdouts are declared in a manifest before final evaluation. A manifest records:

- source path and SHA-256 dataset fingerprint;
- split column and holdout values;
- source date range;
- feature and target versions;
- creation timestamp;
- Git commit.

CLI research and evidence commands require a holdout manifest and run on a physically materialized development CSV whose content hash must match the manifest source fingerprint. Final holdout result writing requires an explicit final-evaluation flag and refuses overwrite. No final empirical holdout result is included in the repository because the full declared run has not been executed.

## 5. Model Suite

The benchmark suite is intentionally layered:

- no-skill and always-flat baselines;
- simple imbalance and signed-flow threshold rules;
- logistic/softmax classification;
- ridge expected-edge model;
- optional tree/boosting models when dependencies are available;
- sequence MLP-style smoke path;
- causal TCN with dilated residual blocks and a documented receptive field;
- Transformer with positional encoding and time-delta channel support;
- compact LOB CNN hook for tensor smoke tests.

The compact CNN is not described as a DeepLOB replication. A faithful DeepLOB-style result requires genuine multi-level book tensors at sufficient historical scale. The current repository has the L2 tensor path and smoke tests, but the empirical DeepLOB gate remains blocked by data coverage.

The reduced CPU neural artifacts in `artifacts/reduced_e2e/sequence_tcn_smoke.csv` and `artifacts/reduced_e2e/sequence_transformer_smoke.csv` verify that the Torch training path runs on a tiny clean L2 fixture with a positive sequence purge gap. The sequence experiment path now records minibatch training settings, class weighting, scheduler state, selected device, checkpoint path, prediction-export path, balanced accuracy, Brier/ECE calibration metrics, confusion matrices, and stateful test-set economic smoke fields. Their scores are still not meaningful alpha evidence.

## 6. Execution Model

Forecast metrics alone are not enough. The primary executable PnL path is the stateful simulator in `lob_forge.execution_sim`. It processes a chronological stream of signals and market events and records:

- orders with decision time, submit time, side, requested quantity, order type, status, and rejection reason;
- fills with fill time, price, quantity, fee, liquidity type, available quantity, and partial-fill flag;
- positions with cash, signed inventory, average entry price, mark price, equity, realized PnL, turnover, and kill-switch state.

The simulator supports maker/taker paths, top-of-book and L2-level walking, partial fills, fees, slippage, latency, expiry checks, position limits, leverage limits, rate limiting, passive queue-ahead assumptions, cancel/replace on signal decay, and a kill switch that prevents subsequent orders. Observed-fill calibration remains an external paper/live validation task. Legacy round-trip evaluators remain in the repository as diagnostic forecast tools and should not be used as primary executable PnL evidence.

Execution assumptions must be reported beside results. Since observed edges in this research family are small, any claim stronger than preliminary pre-cost predictability requires survival under fees, spread, latency, adverse selection, queue assumptions, capacity limits, and untouched holdout evaluation.

## 7. Statistical Inference

The statistics layer includes:

- Newey-West/HAC standard errors for serially correlated returns;
- fixed block bootstrap;
- moving block bootstrap;
- stationary block bootstrap;
- day-level bootstrap;
- confidence intervals for mean returns;
- Sharpe-like intervals;
- break-even-cost intervals;
- multiple-testing correction utilities.

Adjacent folds, overlapping label windows, and repeated high-frequency trades are not iid observations. Reports should aggregate by day/fold/regime where possible and should state when exact selection-adjusted inference is infeasible. `audit-results` now declares its `inference_grain` explicitly; the current walk-forward CSV artifacts are fold-summary audits, while the reduced E2E fixture derives its intervals from simulator ledgers. Larger empirical runs must preserve trade/day ledgers beside fold summaries before claiming trade- or day-level inference. The reduced E2E run writes a machine-readable experiment registry with evaluated status and validation/test PnL for each predeclared threshold attempt; larger empirical runs must do the same for every attempted feature, horizon, cost, model class, hyperparameter, and seed.

## 8. Current Artifact Taxonomy

The tracked reduced E2E manifest is `artifacts/reduced_e2e/result_manifest.json`. It separates wiring evidence from empirical evidence.

| Category | Artifact | Status | Interpretation |
|---|---|---:|---|
| Forecast evidence | `artifacts/reduced_e2e/classical_walk_forward.csv` | smoke-tested | Synthetic fixture proves validation-selected walk-forward path over development rows only; not market evidence. |
| Experiment registry | `artifacts/reduced_e2e/experiment_registry.jsonl` | smoke-tested | Records the full predeclared threshold search family with selected/evaluated status and validation/test PnL per attempt. |
| Diagnostic gross edge | legacy local docs/results under ignored `results/` | local-only | Useful historical exploration, but not final reproducible evidence unless regenerated with manifests. |
| Stateful executable simulation | `artifacts/reduced_e2e/stateful_orders.csv`, `stateful_fills.csv`, `stateful_positions.csv` | smoke-tested | Synthetic fixture feeds validation-selected rule signals into the simulator and proves ledger accounting constraints. |
| Untouched holdout | `artifacts/reduced_e2e/holdout_manifest.json` | manifest only | Final empirical holdout evaluation remains pending. |
| Neural smoke | `sequence_tcn_smoke.csv`, `sequence_transformer_smoke.csv` | smoke-tested | Tiny fixture verifies train path, architecture wiring, and nonzero sequence purge. |
| C++ replay | `cpp/l2_replay.cpp`, `tests/test_cpp_l2_replay.py` | equivalence-tested | Narrow production-style component mirrors Python replay diagnostics on fixture. |

The reduced stateful simulation ends with final equity below initial cash on the synthetic fixture. That is acceptable and reinforces the reporting discipline: a working pipeline can produce negative or fragile outcomes.

## 9. Reproduction Commands

Fast verification:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ci.txt
bash scripts/run_tests.sh
.venv/bin/python -m ruff check src tests scripts/run_reduced_e2e.py
.venv/bin/python -m mypy \
  src/lob_forge/protocol.py \
  src/lob_forge/holdout.py \
  src/lob_forge/execution_sim.py \
  src/lob_forge/l2_replay.py \
  src/lob_forge/statistics.py
```

Reduced artifacts:

```bash
PYTHONPATH=src python3 scripts/run_reduced_e2e.py
```

C++ replay:

```bash
bash scripts/build_cpp_l2_replay.sh /tmp/l2_replay
/tmp/l2_replay examples/fixtures/l2_replay_fixture.csv
```

Optional CPU neural smoke, when Torch is installed:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-sequence-experiment \
  --model sequence_tcn \
  --baseline-audit artifacts/reduced_e2e/baseline_audit_fixture.csv \
  --l2 examples/fixtures/l2_sequence_fixture.csv \
  --output artifacts/reduced_e2e/sequence_tcn_smoke.csv \
  --checkpoint-path artifacts/reduced_e2e/sequence_tcn_smoke.pt \
  --predictions-output artifacts/reduced_e2e/sequence_tcn_predictions.csv \
  --device auto --class-weighting balanced \
  --depth 1 --window 3 --label-horizon 1 --epochs 1 \
  --max-rows 100 --max-snapshots 20 --min-fold-count 1 --min-l2-rows 1
```

Full cloud study planning:

```bash
MODE=plan make modal-study
```

## 10. Limitations And Next Work

The remaining blockers are empirical:

- full multi-month, multi-asset confirmatory quote/trade/depth-band run;
- immutable final holdout result artifact;
- genuine crypto multi-level L2 tensor experiments over sufficient data;
- repeated-seed neural sweeps;
- paper/live shadow-fill validation;
- stronger selection-adjusted reporting over the full search family.

The next high-value experiment is not another local threshold tweak. It is a clean, declared, cloud-backed run that produces a manifest with Git commit, data fingerprint, full attempted search space, validation-selected candidates, stateful simulator ledgers, dependence-aware intervals, and a one-way holdout evaluation.
