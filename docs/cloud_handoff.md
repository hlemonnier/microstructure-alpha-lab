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
- `scripts/`
- `src/`
- `tests/`

It intentionally excludes local data, result artifacts, virtualenvs, caches, `.git`, `.next`, and `node_modules`.

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

This reports `complete=1` only when every expected daily feature marker and combined feature CSV exists.

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

For the full profile this requires 80 result CSVs, 80 audit CSVs, `pvalues.csv`, `pvalue_corrections.csv`, and passing artifact verification.

## Resume

If interrupted, rerun:

```bash
MODE=run SKIP_INSTALL=1 bash scripts/bootstrap_cloud_expected_edge.sh
```

The study script skips existing combined feature files and completed result/audit pairs.

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
DRY_RUN=0 bash scripts/run_l2_sequence_experiments.sh
```

For manifest-filtered neural runs, set `HOLDOUT_MANIFEST_PATH` before launching the same runner. It will pass `--holdout-manifest`, write per-model development L2 CSVs under `results/model_experiments/development_l2/`, and record the manifest hash plus excluded-row counts in each result artifact:

```bash
HOLDOUT_MANIFEST_PATH=results/holdout_manifests/bybit_l2_sequence_holdout.json \
DRY_RUN=0 bash scripts/run_l2_sequence_experiments.sh
```

The bootstrap wrapper exposes the same path:

```bash
MODE=sequence SKIP_INSTALL=1 RUN_TESTS=0 bash scripts/bootstrap_cloud_expected_edge.sh
```

This writes `results/model_experiments/sequence_transformer_results.csv` and `results/model_experiments/sequence_tcn_results.csv` when `model-readiness-gate` passes. The runner also writes checkpoints and prediction CSVs under `results/model_experiments/checkpoints/` and `results/model_experiments/predictions/`, supports `SEEDS=7,11,13` repeated-seed orchestration, and records calibration plus stateful economic smoke fields in each artifact. The aggregate evidence gate validates the model name, selected L2 path, readiness state, dependency state, and `pipeline_completed=1` in each artifact. `acceptance_passed` is reserved for an explicit predictive/economic acceptance threshold, not for smoke completion.

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
