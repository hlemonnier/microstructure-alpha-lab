# Reproducibility

This project separates fast local verification from heavy empirical runs.

## Environment

Recommended local setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ci.txt
bash scripts/run_tests.sh
```

Pinned CI/dev constraints are in `requirements-ci.txt`. A pinned full research environment is in `requirements-research.txt`. The package dependencies in `pyproject.toml` remain range-based for normal editable installs; use the requirements files when stricter reproducibility is needed.

## Fast Verification

Run:

```bash
bash scripts/run_tests.sh
.venv/bin/python -m ruff check src tests scripts/run_reduced_e2e.py
.venv/bin/python -m mypy \
  src/lob_forge/protocol.py \
  src/lob_forge/holdout.py \
  src/lob_forge/execution_sim.py \
  src/lob_forge/l2_replay.py \
  src/lob_forge/statistics.py
```

This compiles source/tests and runs dependency-light test functions, including causal timing, leakage guards, holdout isolation, simulator invariants, L2 replay validation, neural architecture smoke paths, and C++ replay equivalence when a compiler is available.

## Reduced End-to-End Runs

Fixture-only smoke:

```bash
.venv/bin/python scripts/run_reduced_e2e.py
```

This command first writes `artifacts/reduced_e2e/holdout_manifest.json`, then materializes `artifacts/reduced_e2e/development_feature_fixture.csv` with the declared holdout rows removed, then runs model selection on that development CSV only. It also writes `artifacts/reduced_e2e/l2_sequence_holdout_manifest.json` and `artifacts/reduced_e2e/development_l2_sequence_fixture.csv` so the reduced TCN/Transformer smoke artifacts exercise the same manifest-filtered L2 training path used by serious neural runs. It predeclares the reduced threshold search family and writes `artifacts/reduced_e2e/experiment_registry.jsonl` with evaluated statuses plus validation/test PnL for each attempted threshold rule. The script requires official source provenance so the holdout manifests record a real commit hash. In a Git checkout it reads `git rev-parse HEAD`; in a `git archive` export it reads the expanded `.source-git-commit`; for a generic source ZIP without Git metadata, set `LOB_FORGE_SOURCE_GIT_COMMIT=<40-or-64-char-commit-hash>` explicitly. Invalid or missing archive provenance is rejected rather than written into a manifest.

C++ replay smoke:

```bash
bash scripts/build_cpp_l2_replay.sh /tmp/l2_replay
/tmp/l2_replay examples/fixtures/l2_replay_fixture.csv
```

Laptop-safe historical smoke:

```bash
make laptop-smoke
```

Optional CPU neural smoke when Torch is installed:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli l2-sequence-experiment \
  --model sequence_transformer \
  --baseline-audit artifacts/reduced_e2e/baseline_audit_fixture.csv \
  --l2 examples/fixtures/l2_sequence_fixture.csv \
  --output artifacts/reduced_e2e/sequence_transformer_smoke.csv \
  --holdout-manifest artifacts/reduced_e2e/l2_sequence_holdout_manifest.json \
  --development-l2-output artifacts/reduced_e2e/development_l2_sequence_fixture.csv \
  --checkpoint-path artifacts/reduced_e2e/sequence_transformer_smoke.pt \
  --predictions-output artifacts/reduced_e2e/sequence_transformer_predictions.csv \
  --device auto --class-weighting balanced \
  --depth 1 --window 3 --label-horizon 1 --epochs 1 \
  --max-rows 100 --max-snapshots 20 --min-fold-count 1 --min-l2-rows 1
```

For a serious neural L2 run, do not train against the full source CSV directly. Create a holdout manifest for the L2 source, then pass it through the sequence runner so a filtered development L2 CSV is written and used for training:

```bash
PYTHONPATH=src .venv/bin/python -m lob_forge.cli create-holdout-manifest \
  data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv \
  --output results/holdout_manifests/bybit_l2_sequence_holdout.json \
  --split-column exchange_timestamp \
  --holdout-values <held-out-exchange-timestamp-or-session>

HOLDOUT_MANIFEST_PATH=results/holdout_manifests/bybit_l2_sequence_holdout.json \
SEEDS=7,11,13 DRY_RUN=0 RESUME=0 bash scripts/run_l2_sequence_experiments.sh
```

Plan-only cloud/full study:

```bash
MODE=plan make modal-study
```

Sequence-model smoke after baseline and L2 gates pass:

```bash
MODE=sequence make modal-study
```

For repeated local seeds without launching cloud work:

```bash
SEEDS=7,11,13 DRY_RUN=0 RESUME=0 bash scripts/run_l2_sequence_experiments.sh
```

## Final Holdout

CLI research commands such as `baseline`, `walk-forward`, `calendar-walk-forward`, `conditional-walk-forward`, `logistic-walk-forward`, `edge-walk-forward`, `edge-shadow-decisions`, `eval-rule`, and `regime` require `--holdout-manifest`. The CLI verifies the source content hash and runs against a temporary development CSV with declared holdout rows removed.

Final holdout evaluation must call `write_final_holdout_result(..., explicit_final_evaluation=True, candidate_sha256=...)`; the output path is immutable, and the manifest/candidate pair is locked even if a different output path is later supplied. A serious final command must consume a frozen candidate file rather than accepting tuning/search options, and the holdout manifest must pre-register that candidate's `candidate_sha256` before evaluation.

For the fixed threshold-rule family, the CLI exposes that one-way path:

```bash
.venv/bin/python -m lob_forge.cli create-holdout-manifest <feature_csv> \
  --output <manifest.json> \
  --split-column source_date \
  --holdout-values <final_holdout_dates> \
  --candidate-json <frozen_candidate.json>

.venv/bin/python -m lob_forge.cli final-holdout-rule <feature_csv> \
  --holdout-manifest <manifest.json> \
  --candidate-json <frozen_candidate.json> \
  --output <final_holdout_result.json> \
  --explicit-final-evaluation
```

## Heavy Blockers

The full multi-month, multi-asset study and genuine crypto L2 neural experiments require large historical data and/or cloud compute. The local code path is implemented and smoke-tested, but empirical gates remain pending unless corresponding immutable result artifacts are present.
