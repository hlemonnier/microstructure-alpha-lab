# Full 60-Day Study: Cloud Run Path

The old full-study path was too large for a 16 GB laptop because each combined feature CSV was read into memory by the walk-forward evaluator. The study script now uses bounded-memory streaming evaluation, feature-build memory preflights, and a much smaller local default. The full evidence run is still a cloud job because uncapped multi-symbol feature construction is disk, download, runtime, and RAM heavy.

## Recommendation

Use the tiny laptop profile locally:

```bash
bash scripts/run_60day_expected_edge_study.sh
```

That defaults to:

- `STUDY_PROFILE=laptop_tiny`
- BTCUSDT only
- 2 calendar days: 2023-05-16 to 2023-05-17
- 5s horizon
- capped daily quote rows via `MAX_QUOTE_BUCKETS=1200`
- depth disabled with `WITH_BOOK_DEPTH=0`
- feature-build memory guard: `MAX_FEATURE_BUILD_MEMORY_GB=4`
- reduced fee grid: `0 0.1 0.5`

This profile is for proof-of-pipeline only. It should not be used as alpha evidence.

For the safest local command, use:

```bash
make laptop-smoke
```

The laptop profiles set a runtime process cap through `LOB_FORGE_MAX_PROCESS_MEMORY_GB`:

- `laptop_tiny`: 6 GB
- `laptop_quick`: 10 GB
- `local16_60day`: 8 GB
- `cloud_full`: uncapped by default

The cap is applied by `lob_forge.cli` before heavy feature-build and walk-forward commands run. It is a last-resort guard against the 16 GB laptop swapping into an 80 GB virtual-memory death spiral, but macOS can reject process memory limits. The primary local safety gates are therefore small profile sizes, feature-build preflights, depth disabled on the heavy local profile, and explicit confirmation for the heavy local run. Disable it with `LOB_FORGE_MAX_PROCESS_MEMORY_GB=0` only on a deliberate high-RAM machine.

For a larger local smoke run, use:

```bash
STUDY_PROFILE=laptop_quick bash scripts/run_60day_expected_edge_study.sh
```

That uses BTCUSDT/ETHUSDT, 7 calendar days, 3600 capped quote buckets per day, and depth features.

If you want a larger but still capped laptop run, inspect the plan first:

```bash
PLAN_ONLY=1 STUDY_PROFILE=local16_60day bash scripts/run_60day_expected_edge_study.sh
```

Then run it only if you deliberately accept the local cost:

```bash
CONFIRM_LOCAL16=1 STUDY_PROFILE=local16_60day bash scripts/run_60day_expected_edge_study.sh
```

That keeps BTCUSDT/ETHUSDT and a 5s horizon, disables depth by default, caps each daily feature file at `MAX_QUOTE_BUCKETS=3600`, and uses an 8 GB process cap, 8 GB CSV-load guard, and 6 GB feature-build guard. On a 16 GB laptop this is still the upper bound, not the default path.

Use the full profile only when you are deliberately starting a large data job:

```bash
CONFIRM_HEAVY=1 STUDY_PROFILE=cloud_full bash scripts/run_60day_expected_edge_study.sh
```

Before starting any non-quick run, generate the run plan without downloads or edge evaluation:

```bash
PLAN_ONLY=1 STUDY_PROFILE=local16_60day bash scripts/run_60day_expected_edge_study.sh
CONFIRM_HEAVY=1 STUDY_PROFILE=cloud_full bash scripts/plan_expected_edge_study.sh
```

The plan writes `run_plan.json` inside the result directory and reports days, archive count, feature jobs, edge-evaluation jobs, capped/uncapped row status, depth usage, the predeclared holdout split/values, RAM gate, and whether the current machine can start the profile.

Check feature coverage separately:

```bash
bash scripts/verify_expected_edge_features.sh results/expected_edge_local16_20230516_20230714
bash scripts/verify_expected_edge_features.sh results/expected_edge_60day_20230516_20230714
```

Feature coverage is complete only when this verifier reports `complete=1`. It verifies each daily output hash and exact build configuration, then verifies every combined CSV against its ordered daily-input manifest. Legacy marker existence alone is not completion evidence.

To evaluate only feature jobs that are already complete, dry-run first:

```bash
bash scripts/run_complete_feature_edge_jobs.sh
```

Execute only after reviewing the selected jobs:

```bash
DRY_RUN=0 MAX_JOBS=1 bash scripts/run_complete_feature_edge_jobs.sh
```

This is a partial-progress tool. It does not satisfy the full study gate unless the feature verifier and study verifier both report `complete=1`.

The full profile runs:

- BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- 5s and 10s horizons
- full fee grid: `0 0.05 0.1 0.25 0.5 1 2 5`
- uncapped daily rows

The script refuses the full profile unless `CONFIRM_HEAVY=1` is set. The capped `local16_60day` profile also refuses non-plan execution unless `CONFIRM_LOCAL16=1` is set. The full profile defaults to a 64 GB RAM preflight, a 48 GB CSV-load ceiling if streaming is disabled, and a 48 GB feature-build memory ceiling before parsing daily ZIP files. It uses `EDGE_STREAMING=1` by default, which keeps only the current train/validation/test window in memory during `edge-walk-forward`. Direct CLI calls also stream by default; use `--no-stream` or `EDGE_STREAMING=0` only for tiny debug CSVs or high-RAM cloud runs. If you disable streaming, the CSV load-memory guard still applies.

## Instance Size

Minimum practical local target for capped profiles:

- 16 GB RAM
- 150 GB disk
- 8+ vCPU

Minimum practical cloud target for the full profile:

- 64 GB RAM
- 250 GB disk
- 8+ vCPU

Recommended cloud target for speed and headroom:

- 64-128 GB RAM
- 250+ GB disk
- 12+ vCPU

The current code does not need GPU acceleration for the expected-edge study. A GPU instance is still convenient if it comes with enough RAM, fast local disk, and better network throughput.

## Platform Shortlist

Verify current inventory and pricing before renting because cloud RAM pricing changes quickly. As of the 2026-06-03 review, the practical choices are:

- RunPod: best first try for a cheap one-off research run if availability is good. Check the pod deployment console for current GPU pricing, RAM, CPU, and volume options. RunPod bills pods by compute and storage, bills by the second, has no ingress/egress fees in its pod docs, and warns that pods are not long-term storage.
  <https://docs.runpod.io/pods/pricing>
- Lambda Cloud: cleaner managed GPU-cloud path if you want a simpler high-RAM machine. Current public pricing lists several 1x GPU instances with 100+ GiB RAM and large local SSD options.
  <https://lambda.ai/pricing>
- AWS EC2 memory-optimized: best CPU-only route if you want a conventional VM. R7i is a good class for this workload because AWS presents it as a memory-optimized family. Add enough EBS storage.
  <https://aws.amazon.com/ec2/instance-types/r7i/>
- Modal: good if you want serverless batch execution instead of managing a VM. Modal prices CPU, memory, and volumes separately per second; this is attractive for resumable jobs but requires wrapping the study as a Modal job.
  <https://modal.com/pricing>
- Hetzner Cloud or a Hetzner dedicated server: strong price/performance if you want a simple EU machine and can tolerate less managed workflow polish than AWS. Use dedicated resources for sustained research calculations.
  <https://www.hetzner.com/cloud/>
- Vast.ai: cheapest-marketplace route, but check RAM, disk, bandwidth, reliability, persistence, and storage charges carefully before renting.
  <https://docs.vast.ai/guides/reference/billing>
- Colab/Colab Pro: avoid for the full unattended study. Google's Colab FAQ says usage limits, idle timeout, maximum VM lifetime, GPU types, and other factors vary over time. That is fine for notebooks, not for a deterministic 60-day archive run.
  <https://research.google.com/colaboratory/faq.html>

My call: use Modal first if you want the least operational friction because the repo already has `scripts/modal_expected_edge_job.py` and `scripts/run_modal_expected_edge.sh`. Use RunPod first for a cheaper one-off VM if you are comfortable managing a pod and persistent volume. Use Lambda if you want a cleaner managed GPU VM with enough RAM, AWS R7i if you specifically want a CPU memory-optimized VM, and Hetzner if you want the cheapest conventional EU machine. Avoid paying for H100/B200 class hardware unless it is the only available way to get RAM and disk; the current expected-edge study is not GPU-bound.

## Cloud Command Sequence

For a source-only upload package, see [cloud_handoff.md](cloud_handoff.md). The short path is:

```bash
bash scripts/package_cloud_handoff.sh
```

Upload `dist/microstructure-alpha-lab-cloud-handoff-*.zip` to the cloud machine, then run:

```bash
unzip microstructure-alpha-lab-cloud-handoff-*.zip
cd microstructure-alpha-lab
MODE=plan bash scripts/bootstrap_cloud_expected_edge.sh
MODE=run SKIP_INSTALL=1 bash scripts/bootstrap_cloud_expected_edge.sh
MIN_AUDIT_FOLD_COUNT=20 bash scripts/verify_expected_edge_study.sh results/expected_edge_60day_20230516_20230714
```

For Modal, the equivalent managed batch route is:

```bash
python3 -m pip install ".[cloud]"
modal setup
make modal-study
MODE=run make modal-study
HOLDOUT_MANIFEST_PATH=results/holdout_manifests/bybit_l2_sequence_holdout.json \
MODE=sequence make modal-study
```

This uses a Modal Volume for resumable `data/` and `results/` storage. The Modal job defaults to 8 CPU cores, 64 GiB requested RAM, a 128 GiB hard memory limit, 300 GiB ephemeral disk, and a 24-hour timeout. Override with `LOB_FORGE_MODAL_CPU_CORES`, `LOB_FORGE_MODAL_MEMORY_REQUEST_GB`, `LOB_FORGE_MODAL_MEMORY_LIMIT_GB`, or `LOB_FORGE_MODAL_EPHEMERAL_DISK_GB` before invoking the wrapper. The `sequence` mode requires and forwards the explicit holdout-manifest path, then runs `scripts/run_l2_sequence_experiments.sh` to write Transformer/TCN artifacts after the baseline and L2 gates pass. See [cloud_handoff.md](cloud_handoff.md) for volume download commands and the `LOB_FORGE_MODAL_VOLUME` override.

Before paying for the run, check the operational state:

```bash
make external-readiness
```

This check is expected to stay red until Modal is authenticated, the full cloud study has completed, the immutable final holdout artifact has been written for the frozen selected candidate, and real paper/live fills have been imported.

If the run is interrupted, rerun the same command. Raw archives remain reusable. Daily features are reused only after content/config/source-hash verification; combined files and manifests are rewritten from verified inputs; result/audit pairs are skipped only when their provenance sidecar verifies against the current plan and artifacts.

The expected-edge study is complete only when `verify_expected_edge_study.sh` reports `complete=1`. The verifier reads `run_plan.json`, expands every expected symbol/horizon/fee cell, verifies each result/audit provenance sidecar and planned split sizes, then checks the candidate registry, `pvalues.csv`, and `pvalue_corrections.csv`. Because threshold choice occurs inside validation, the inference family has one p-value per completed validation-selection procedure, identified by `procedure_sha256`; the threshold grid and selection rule are part of that hash. The stricter remaining-evidence gate also requires at least 20 audit folds per artifact. The full profile therefore needs 80 result CSVs, 80 audits, 80 sidecars, and 80 procedure rows; the laptop quick profile needs 10 of each.

After the full verifier reports `complete=1`, prepare the immutable expected-edge final holdout from the completed run plan and candidate registry. Keep the first pass dry-run so the selected artifact, derived combined feature path, development manifest, candidate JSON, candidate-locked final manifest, and final evaluation command are visible before any immutable artifact is written:

```bash
HOLDOUT_VALUES=<final_source_date_or_dates> \
bash scripts/prepare_final_holdout_from_expected_edge_study.sh
```

The actual write path is deliberately explicit:

```bash
HOLDOUT_VALUES=<final_source_date_or_dates> \
DRY_RUN=0 \
bash scripts/prepare_final_holdout_from_expected_edge_study.sh
```

By default the helper refuses to run unless the study status is complete, an audited selected candidate exists in `candidate_registry.jsonl`, and no candidate/manifest/final-output artifact at the target paths already exists. Override `RESULT_DIR`, `PLAN_PATH`, `CANDIDATE_REGISTRY`, `CANDIDATE_OUTPUT`, `DEVELOPMENT_MANIFEST_OUTPUT`, `FINAL_MANIFEST_OUTPUT`, or `FINAL_OUTPUT` only when preserving a separate final-holdout namespace.

Current local feature status after the latest audit:

- laptop quick: `complete=0`, 0/14 valid daily markers, 14 legacy-invalid markers, 0/2 verified combined feature jobs.
- capped local16: `complete=0`, 0/120 valid daily markers (16 legacy-invalid and 104 missing), 0/2 verified combined feature jobs. Its 10 legacy result/audit pairs have no valid sidecars and all 50 registry candidates are `artifact_error`.
- cloud full: `complete=0`, 0/600 valid daily markers (202 legacy-invalid and 398 missing), 0/10 verified combined feature jobs; three existing combined files are invalid and seven are missing.

If Binance downloads are too slow locally, extract an explicitly partial artifact from already-built daily features instead of pretending the 60-day run completed:

```bash
EXISTING_FEATURES_ONLY=1 \
STUDY_PROFILE=local16_60day \
STUDY_TAG=expected_edge_existing_features_20230516_20230714 \
PROCESSED_ROOT=data/processed/expected_edge_60day_20230516_20230714 \
OUT_DIR=results/expected_edge_existing_features_20230516_20230714 \
bash scripts/run_60day_expected_edge_study.sh
```

That command combines only daily feature files with `.done` markers and reports missing dates. It is useful for pipeline validation, but it does not satisfy the full 60-day study gate.

Current local inventory includes 15 BTCUSDT and 31 ETHUSDT legacy capped daily files in the partial cloud tree, but their markers predate the content/configuration contract. They are inventory only, not verified evidence, until rebuilt with current markers and combined manifests.

Once the local16 feature verifier reports complete, its result matrix can be smoke-tested without rebuilding those already-verified features:

```bash
bash scripts/run_local16_existing_feature_edge_jobs.sh
DRY_RUN=0 MAX_FOLDS=1 bash scripts/run_local16_existing_feature_edge_jobs.sh
```

`MAX_FOLDS=1` should remain labeled as a smoke. It writes the expected file matrix quickly, but `verify_remaining_evidence_gates.sh` keeps the capped gate red until each provenance-valid audit has at least 20 folds. The runner now refuses incomplete or legacy-invalid feature inputs, requires valid sidecars before skipping results, and writes sidecars for new results. The legacy local16 directory contains 20-fold CSVs but fails the current feature/result provenance contract, so it is not completion evidence.

The existing-feature runners use `RERUN_UNDERFOLDED=1` by default, so a stale smoke artifact is rerun when its audit `fold_count` is below `MIN_AUDIT_FOLD_COUNT`. Use `RERUN_UNDERFOLDED=0` only when intentionally preserving partial smoke files.

## Disk Cleanup

To delete partial processed/result artifacts while keeping raw downloads:

```bash
bash scripts/cleanup_partial_expected_edge_study.sh expected_edge_60day_20230516_20230714
```
