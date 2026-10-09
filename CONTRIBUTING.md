# Contributing

Microstructure Alpha Lab evaluates short-horizon predictions under explicit timing and execution assumptions. Contributions should make those assumptions easier to inspect and reproduce.

## Setup and verification

Use the [quick start](README.md#quick-start), then install the pinned developer tools:

```bash
.venv/bin/python -m pip install -r requirements-ci.txt
make check-repository PYTHON=.venv/bin/python
.venv/bin/python -m ruff check src tests scripts
git diff --check
```

After committing the proposed source revision, run `make e2e PYTHON=.venv/bin/python`. Include the generated `evidence.json` and the relevant logs in the review. CI retains an equivalent artifact for each supported verification environment.

The E2E run checks source and wheel installation, public-file hygiene, synthetic holdout exclusion, validation-only selection, execution ledgers, archive completeness and Python/C++ replay parity. It does not run a historical study. The existing regression suite remains available through `make test`.

## Changes

- Describe the failure modes and expected behavior before changing a component in isolation.
- Prefer a complete E2E scenario with inspectable output. Do not add unit tests after writing implementation code.
- Use focused commits and explain the problem, resulting behavior and verification in the pull request.
- Keep downloaded data, environments, generated results, credentials and personal working notes out of Git.
- Preserve the contents and paths of frozen protocols and evidence. Register an amendment rather than rewriting a completed or interrupted study.

## Research claims

Report the cohort, source availability, target, split policy, complete control procedure and execution assumptions alongside a result. Keep development evidence, independent confirmation and executable performance distinct. Avoid choosing models from test metrics or interpreting a software check as empirical validation.

The historical research is [paused](docs/research/boundary_research_pause_20260909.md). Documentation and software maintenance do not authorize resuming it or opening its independent confirmation data. Expensive acquisition, cloud execution and observed trading-fill work require a separately agreed study scope.

## Issues

For a reproducibility problem, provide the source commit, Python/platform versions, exact command, expected behavior and a small synthetic example. Remove credentials and private data from logs before sharing them.
