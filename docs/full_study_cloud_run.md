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

The plan writes `run_plan.json` inside the result directory and reports days, archive count, feature jobs, edge-evaluation jobs, capped/uncapped row status, depth usage, RAM gate, and whether the current machine can start the profile.

Check feature coverage separately:

```bash
bash scripts/verify_expected_edge_features.sh results/expected_edge_local16_20230516_20230714
bash scripts/verify_expected_edge_features.sh results/expected_edge_60day_20230516_20230714
```

Feature coverage is complete only when this verifier reports `complete=1`. It checks every expected daily `.done` marker plus each combined feature CSV before result evaluation.

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
MODE=sequence make modal-study
```

This uses a Modal Volume for resumable `data/` and `results/` storage. The Modal job defaults to 8 CPU cores, 64 GiB requested RAM, a 128 GiB hard memory limit, 300 GiB ephemeral disk, and a 24-hour timeout. Override with `LOB_FORGE_MODAL_CPU_CORES`, `LOB_FORGE_MODAL_MEMORY_REQUEST_GB`, `LOB_FORGE_MODAL_MEMORY_LIMIT_GB`, or `LOB_FORGE_MODAL_EPHEMERAL_DISK_GB` before invoking the wrapper. The `sequence` mode runs `scripts/run_l2_sequence_experiments.sh` to write Transformer/TCN artifacts after the baseline and L2 gates pass. See [cloud_handoff.md](cloud_handoff.md) for volume download commands and the `LOB_FORGE_MODAL_VOLUME` override.

Before paying for the run, check the operational state:

```bash
make external-readiness
```

This check is expected to stay red until Modal is authenticated, the full cloud study has completed, the immutable final holdout artifact has been written for the frozen selected candidate, and real paper/live fills have been imported.

If the run is interrupted, rerun the same command. The script keeps raw archives, skips existing combined feature files, skips completed result/audit pairs, and daily feature builds use `.done` markers.

The expected-edge study is complete only when `verify_expected_edge_study.sh` reports `complete=1`. The verifier reads `run_plan.json`, expands every expected symbol/horizon/fee result pair, checks the result CSV, audit CSV, candidate registry, `pvalues.csv`, and `pvalue_corrections.csv`, then runs the artifact verifier. The registry must be refreshed after result generation, and `pvalues.csv` must include one candidate/config-linked row per completed threshold-grid attempt using the candidate `config_sha256`; artifact-level p-values alone are not sufficient. The stricter remaining-evidence gate also requires at least 20 audit folds per artifact. For the full profile that means 80 result CSVs and 80 audit CSVs plus the completed candidate/config p-value family. For the laptop quick profile it means 10 result CSVs and 10 audit CSVs.

Current local feature status after the latest audit:

- laptop quick: `complete=1`, 14/14 daily markers, 2/2 feature jobs.
- capped local16 feature rebuild path: `complete=0`, 16/120 daily markers, 0/2 feature jobs under `data/processed/expected_edge_local16_20230516_20230714`.
- capped local16 result gate from existing combined features: `complete=1`, 10/10 result files, 10/10 audit files, 20-fold audit threshold passed in `results/expected_edge_local16_20230516_20230714/`.
- cloud full: `complete=0`, 202/600 daily markers, 1/10 feature jobs. The currently complete job is BNBUSDT at 5000ms.

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

Current local evidence: `results/expected_edge_existing_features_smoke_20230516_20230714/` verifies this mode on the already-built partial feature tree. BTCUSDT has 15 available capped days and ETHUSDT has 31 available capped days. The stricter capped BTC/ETH result gate has now been promoted separately in `results/expected_edge_local16_20230516_20230714/`.

If combined 60-day BTC/ETH feature files already exist under the cloud processed root, the local16 result matrix can be smoke-tested without rebuilding features:

```bash
bash scripts/run_local16_existing_feature_edge_jobs.sh
DRY_RUN=0 MAX_FOLDS=1 bash scripts/run_local16_existing_feature_edge_jobs.sh
```

`MAX_FOLDS=1` should remain labeled as a smoke. It writes the expected file matrix quickly, but `verify_remaining_evidence_gates.sh` keeps the capped 60-day gate red until each audit has at least 20 folds. The current local16 result directory already passes that 20-fold threshold; use `MAX_FOLDS=1` only for future wiring checks.

The existing-feature runners use `RERUN_UNDERFOLDED=1` by default, so a stale smoke artifact is rerun when its audit `fold_count` is below `MIN_AUDIT_FOLD_COUNT`. Use `RERUN_UNDERFOLDED=0` only when intentionally preserving partial smoke files.

## Disk Cleanup

To delete partial processed/result artifacts while keeping raw downloads:

```bash
bash scripts/cleanup_partial_expected_edge_study.sh expected_edge_60day_20230516_20230714
```
