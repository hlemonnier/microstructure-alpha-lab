# Reproduce the performance pilot

Use the repository's pinned `requirements-research.txt` environment. The experiment downloads approximately 1.54 GiB of public daily archives, extracts the registered two-hour windows, and writes raw data and detailed results into ignored local directories. It uses two preparation workers, a 4 GB resident-memory watchdog, and 2 GB per-worker feature-build estimates. No cloud account, exchange credentials or trades are required.

The original protocol is preserved alongside a coverage-only noon-window amendment. The acquisition plan includes only May 16–23, 2023. May 24–26 remain outside this experiment.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_performance_pilot.py \
  --protocol docs/research/performance_pilot_20260907_window_revision.json \
  --plan docs/research/performance_pilot_20260907_acquisition_plan.json \
  --output-root data/research/performance_pilot_20260907_window_revision \
  --workers 2 --prepare

PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 \
  .venv/bin/python scripts/run_performance_pilot.py --run
```

Omitting `--prepare` or `--run` prints the respective scope without executing it. The preparer verifies official SHA-256 checksums; compare the resulting source hashes with [the portable integrity snapshot](performance_pilot_20260907_source_integrity.json). Provider archive revisions may prevent byte-identical reproduction. Local manifest hashes also bind local paths and preparation metadata; portable source hashes identify the actual market observations.

The runner binds protocol, dataset, learner and execution-source hashes before fitting. It rejects incompatible existing runs. Completed jobs can resume only when their recorded artifacts still match. Keep failed/partial runs and use a new output directory for changed code; outcomes already inspected cannot become fresh test evidence by rerunning them.

The default result directory is `results/performance_pilot_20260907_window_revision`. `summary.json` contains all 216 model/session/fee/latency evaluations. Each of the 18 fitted model jobs includes its checkpoint, configuration, predictions, validation-attempt registry, selected-policy signals, orders, fills, equity and artifact hashes. Checkpoints are local trusted artifacts; their predictions are verified after serialization before evaluation.

The pilot is descriptive research over three distinct test dates. Its standalone checkpoint and trial files do not replace the repository's formal immutable holdout or empirical promotion gates. See [the research decision path](performance_research_path.md) for the next experiment rules.
