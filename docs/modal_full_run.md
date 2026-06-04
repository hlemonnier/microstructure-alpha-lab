# Modal Full Run Checklist

Use Modal for the full expected-edge study. The full profile is intentionally a cloud job, not a 16 GB laptop workload.

## 1. Authenticate Modal

```bash
.venv/bin/modal setup
```

If `.venv/bin/modal` is not present:

```bash
python3 -m pip install ".[cloud]"
modal setup
```

## 2. Check Readiness

```bash
make external-readiness
```

The Modal checks should pass before paying for a run. The full cloud study and paper/live fill checks are expected to stay red until the run finishes and observed fills are imported.

## 3. Dry-Run The Remote Plan

```bash
MODE=plan make modal-study
```

This verifies packaging, remote dependency installation, Modal volume mounting, and the cloud profile plan without launching the full study.

## 4. Run The Full Study

```bash
MODE=run make modal-study
```

Default Modal resources:

- 8 CPU cores
- 64 GiB requested RAM
- 128 GiB hard RAM limit
- 300 GiB ephemeral disk
- persistent Modal volume: `microstructure-alpha-lab-expected-edge`

For more headroom:

```bash
LOB_FORGE_MODAL_MEMORY_REQUEST_GB=96 \
LOB_FORGE_MODAL_MEMORY_LIMIT_GB=160 \
MODE=run make modal-study
```

## 5. Optional Sequence Models

```bash
MODE=sequence make modal-study
```

Run this after the baseline expected-edge study if the baseline and L2 evidence gates are ready.

## 6. Download Results

```bash
.venv/bin/modal volume get \
  microstructure-alpha-lab-expected-edge \
  /results/expected_edge_60day_20230516_20230714 \
  ./expected_edge_results
```

## 7. Verify Locally

```bash
MIN_AUDIT_FOLD_COUNT=20 \
bash scripts/verify_expected_edge_study.sh \
  expected_edge_results
```

Then copy or inspect the result artifacts and rerun:

```bash
make verify-evidence-gates
```
