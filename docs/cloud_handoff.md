# Cloud Handoff For Expected-Edge Studies

Use this when the 16 GB laptop is not the right machine for the capped 60-day or full 60-90 day expected-edge study.

## Build The Upload Package

```bash
bash scripts/package_cloud_handoff.sh
```

or:

```bash
make cloud-package
```

The package contains only the runnable project surface:

- `README.md`, `Makefile`, `pyproject.toml`, `.gitignore`
- `requirements-ci.txt`, `requirements-research.txt`
- `docs/`
- `examples/`
- `scripts/`
- `src/`
- `tests/`
- `.source-git-commit`

It intentionally excludes local data, result artifacts, virtualenvs, caches, `.git`, `.next`, and `node_modules`.

The cloud bootstrap installs `requirements-research.txt` first and then installs the local package with `--no-deps`. That keeps the serious run on pinned research dependencies instead of resolving the range-based optional dependencies in `pyproject.toml`.

## Upload And Unpack

Upload the generated `dist/microstructure-alpha-lab-cloud-handoff-*.zip` to the cloud instance, then run:

```bash
unzip microstructure-alpha-lab-cloud-handoff-*.zip
cd microstructure-alpha-lab
```

## Plan First

On the cloud machine, generate the run plan before any downloads:

```bash
MODE=plan bash scripts/bootstrap_cloud_expected_edge.sh
```

The default profile is `cloud_full`. The plan writes:

```text
results/expected_edge_60day_20230516_20230714/run_plan.json
```

Continue only if the plan reports `can_start=1` and the RAM/disk/network choice is acceptable.

To inspect already-built daily feature coverage:

```bash
bash scripts/verify_expected_edge_features.sh results/expected_edge_60day_20230516_20230714
```

This reports `complete=1` only when every daily feature's content hash and build configuration match its marker and every combined CSV matches an ordered-input manifest. Bare legacy `.done` files and arbitrary nonempty combined files fail closed.

To run only complete feature jobs, first dry-run:

```bash
bash scripts/run_complete_feature_edge_jobs.sh
```

Then execute deliberately:

```bash
DRY_RUN=0 MAX_JOBS=1 bash scripts/run_complete_feature_edge_jobs.sh
```

## Run And Verify

Start the full run:

```bash
MODE=run bash scripts/bootstrap_cloud_expected_edge.sh
```

When the command finishes, completion is proven only if:

```bash
MIN_AUDIT_FOLD_COUNT=20 bash scripts/verify_expected_edge_study.sh results/expected_edge_60day_20230516_20230714
```

reports:

```text
complete=1
```

For the full profile this requires 80 result CSVs, 80 audit CSVs, 80 result provenance sidecars, procedure-level `pvalues.csv` and `pvalue_corrections.csv`, and passing artifact verification.

## Resume

If interrupted, rerun:

```bash
MODE=run SKIP_INSTALL=1 bash scripts/bootstrap_cloud_expected_edge.sh
```

The study script reuses a daily feature only when its source hashes, output hash, and complete build configuration match. It rewrites each combined CSV and its ordered-input manifest, and skips a result/audit pair only when its sidecar still verifies against the current plan, feature file, holdout manifest, result, and audit.

## Verify Only

```bash
MODE=verify SKIP_INSTALL=1 RUN_TESTS=0 bash scripts/bootstrap_cloud_expected_edge.sh
```

## Sequence Experiments

After the baseline audit and normalized L2 artifacts exist on the cloud machine, dry-run the Transformer/TCN artifact commands:

```bash
bash scripts/run_l2_sequence_experiments.sh
```

Execute them deliberately on a Torch-capable environment:

```bash
HOLDOUT_MANIFEST_PATH=results/holdout_manifests/bybit_l2_sequence_holdout.json \
DRY_RUN=0 bash scripts/run_l2_sequence_experiments.sh
```

Certifiable neural runs require `HOLDOUT_MANIFEST_PATH`. The runner passes `--holdout-manifest`, writes per-model development L2 CSVs under `results/model_experiments/development_l2/`, and records the manifest hash plus excluded-row counts in each result artifact:

```bash
HOLDOUT_MANIFEST_PATH=results/holdout_manifests/bybit_l2_sequence_holdout.json \
DRY_RUN=0 bash scripts/run_l2_sequence_experiments.sh
```

The bootstrap wrapper exposes the same path:

```bash
HOLDOUT_MANIFEST_PATH=results/holdout_manifests/bybit_l2_sequence_holdout.json \
MODE=sequence SKIP_INSTALL=1 RUN_TESTS=0 bash scripts/bootstrap_cloud_expected_edge.sh
```

This writes `results/model_experiments/sequence_transformer_results.csv` and `results/model_experiments/sequence_tcn_results.csv` when `model-readiness-gate` passes. The runner also writes checkpoints and predictions, supports repeated seeds, and records calibration plus flat-at-label-horizon economics. Resume only reuses artifacts whose hashes and full recorded run configuration match; incompatible checkpoints are ignored and retrained. The aggregate evidence gate requires a verified development holdout with excluded rows, matching source/development-L2, holdout, checkpoint, and prediction hashes, the current economic-semantics version, zero residual inventory, readiness/dependency success, and `pipeline_completed=1`. `acceptance_passed` remains reserved for a separate predictive/economic threshold.

## Modal Batch Runner

If you prefer a managed batch runtime instead of an SSH VM, use the Modal runner:

```bash
python3 -m pip install ".[cloud]"
modal setup
make modal-study
MODE=run make modal-study
MODE=sequence make modal-study
```

The Modal app requests 8 CPU cores, 64 GiB RAM, a 128 GiB hard memory limit, 300 GiB ephemeral disk, and a 24-hour timeout. It mounts a persistent Modal Volume named `microstructure-alpha-lab-expected-edge` by default, then symlinks project `data/` and `results/` into that volume before running `scripts/bootstrap_cloud_expected_edge.sh`.

The wrapper prefers `.venv/bin/modal` when it exists, then falls back to `modal` on `PATH`. `make external-readiness` checks both Modal CLI installation and whether Modal token fields are configured; run `modal setup` or `modal token set` if the `modal_auth` check is not ready.

To change the cloud RAM envelope:

```bash
LOB_FORGE_MODAL_MEMORY_REQUEST_GB=96 LOB_FORGE_MODAL_MEMORY_LIMIT_GB=160 MODE=run make modal-study
```

To use a different volume name:

```bash
LOB_FORGE_MODAL_VOLUME=my-lob-forge-study MODE=run make modal-study
```

Download artifacts with the Modal CLI, for example:

```bash
modal volume get microstructure-alpha-lab-expected-edge /results/expected_edge_60day_20230516_20230714 ./expected_edge_results
```

If the function times out or is interrupted, rerun the same command. The study scripts are resumable from the volume-backed `data/` and `results/` directories.
