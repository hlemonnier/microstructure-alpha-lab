# Midpoint experiment restart — registered before new fits

The research goal was reactivated on October 11, 2026 (Europe/Paris), with predictive accuracy still the first priority. The September pause is historical. No new result is implied by resumption.

## Decision and failure cases, before execution

Use an explicit fresh attempt of the unchanged 48-fit September protocol. The eight completed fits are inexpensive to repeat relative to adding and validating new resume machinery. Preserve the complete original interrupted attempt. Account for eight previously completed fits, one interrupted ninth fit and 39 previously unstarted fits; the fresh attempt contains 48 fits, not 40.

The following failure cases must be checked before fitting or reporting:

1. The task's configured directory is stale. Use the verified repository under `Pandora Research/Research/Crypto Microstructure Research` and record the actual execution root.
2. Three packaging files on current main differ from the 2,158 frozen input hashes. Do not relax the hashes or revert current main. Export the exact historical Git revision `5b05940c101cf23eb8ec1038edf0896f60248e7b` to an isolated source directory and verify all pins there.
3. A relocated virtual environment could import different code or native libraries. Check the original Python and six dependency versions, resolved runner location and pinned OpenMP binary. Use the original environment variables and two CPU threads. Verify that psutil imports and reads current-process RSS before fitting, and require positive sampled RSS in every completed worker; a failed daemon alone is insufficient to stop the historical worker.
4. A source snapshot could accidentally write through a link to the old attempt. Keep a real, separate results directory for the new output; link historical result directories only as unchanged inputs. Recheck all 173 old artifact hashes and lengths before and after execution.
5. A stale progress file or partial worker could be mistaken for completion. The old attempt has no final summary, completed fold or assessment panels. Its ninth worker is not a completed fit. The new output must not already exist.
6. Repeating stochastic fits could change predictions after relocation even with matching version strings. Before inspecting any new assessment score, compare the eight repeated procedures' saved probabilities and priors with the old arrays, exactly. A discrepancy is an execution-compatibility finding, not permission to pick the better run.
7. Partial results could influence continuation. Keep the model family, partitions, rows, targets, coefficients, seeds, decisions and gates unchanged. All 48 fits must finish before the original runner opens assessment labels. No partial accuracy inspection or additional fit is allowed.
8. A superficially successful process could omit or alter panels. Run the unchanged full assessor: 48 workers, 576 recomputed panels, 96 exact retained controls, all 20 native-source and 20 midpoint attribution comparisons, resource/causality checks and all stronger references.
9. A three-date result could be promoted as independent evidence. The dates are exposed development data. The original five-point/twenty-independent-date and economic gates remain unchanged. The reserved June 13–July 2 data stay unopened for market analysis.

## Execution and accounting

The original protocol remains `docs/research/boundary_midpoint_conversion_screen_20260909.json`, SHA256 `4bd811dbc1a62c445723aab77078bc3963b06a202970934c371ff3994f99446f`. Its runner, assessor and model code are unchanged. The additive JSON registration alongside this document records source/export hashes, environment, exact arguments and the pause inventory; it does not replace or modify that protocol.

Create the source export under `artifacts/midpoint-replay-20261011/source`. Give it links to the already verified local `data`, `.venv` and historical result directories. Use a new real output directory, `results/boundary_midpoint_conversion_restart_20261011`, inside that export; the current checkout may link to this new result for ordinary relative-path references. No fresh market acquisition is required.

Run the original script from the exported source root with `PYTHONPATH=src:scripts:.venv/lib/python3.12/site-packages:data/research/boundary_tabicl_20260908/runtime`, the original sklearn `.dylibs` path, and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`:

```sh
.venv/bin/python scripts/run_boundary_midpoint_conversion_screen.py \
  --protocol docs/research/boundary_midpoint_conversion_screen_20260909.json \
  --output results/boundary_midpoint_conversion_restart_20261011
```

One learner runs at a time with the original 8 GiB sampled RSS and 900-second per-fit limits. No implicit retry, source change or resume is permitted. Any failure is preserved. Successful completion would mean 56 completed fit executions across both attempts, one prior interrupted execution and 48 distinct registered date/model procedures. Eight repeated executions provide a reproducibility check, not eight new independent observations.

The market experiment and unchanged assessor provide end-to-end evidence from pinned prepared sources through model save/reload, forecasts, causality checks and metrics. Keep a verifiable artifact inventory for the exact-parity and old-attempt checks. No new unit tests are introduced. Software checks do not establish a performance gain.
