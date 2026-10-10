# Research paused at the user's request — 2026-10-11

Research is paused. The user requested that everything stop and that the current findings be committed and pushed for continuation later. The goal is marked paused, the active reviewer was interrupted, and the experiment process group was stopped with SIGINT. The driver exited with code 130; a subsequent process inventory found no remaining research driver or worker. **Resume only after a new user instruction.** This pause supersedes proposed next steps in earlier reports.

Predictive accuracy remains the priority. No substantial independently confirmed gain has been established. The [research summary](README.md) retains the completed findings: the twenty-date observation blend reached 54.0516% balanced accuracy, 3.4431 percentage points above the original reference and 0.4821 points above the established blend. Those dates were exposed; the substantial-gain gate remains unmet. Counted-depth and unadjusted quote-currency screens found no advancement candidate. The midpoint representation still has no assessed accuracy result.

## Exact stopping point

The [October restart registration](boundary_midpoint_conversion_restart_20261011.md) was already pushed at `578a31e`. It runs the unchanged 48-fit family from historical revision `5b05940c101cf23eb8ec1038edf0896f60248e7b`, preserving the earlier interrupted attempt. The original protocol SHA256 remains `4bd811dbc1a62c445723aab77078bc3963b06a202970934c371ff3994f99446f`.

| Attempt | Completed fits | Interrupted fits | Unstarted fits | Assessment |
| --- | ---: | ---: | ---: | --- |
| September 9 original | 8 | 1 | 39 | None |
| October 11 restart | 11 | 1 | 36 | None |

The restart completed eleven June 3 workers: all eight observation-only combinations, plus `combined_fx100_raw_side_hgb`, `combined_fx100_raw_side_neural`, and `combined_fx100_midpoint_side_hgb`. `combined_fx100_midpoint_side_neural` was interrupted before a worker completion or successful supervision record was written. The stop was requested by the user, without inspecting predictive scores.

All three date folds were prepared. There are no completed fold markers, final prediction panels, or final summary. The saved `progress.json` still says `training`; this is a historical last update and must not be treated as a live process. Eleven completed workers report assessment-label access as false. Their successful supervision records, log hashes and positive sampled RSS values were checked while recording this pause. These checks do not establish accuracy or continuous peak-memory bounds.

Across the two attempts there were 19 completed fit executions and two interrupted executions, covering eleven distinct successfully completed date/model procedures. Eight procedures were repeated. Their probability/prior arrays have **not** been compared; the registered compatibility gate remains pending. No new assessment scores were computed or inspected, and June 13–July 2 confirmation archives remain unopened for market analysis.

## Preserved material

- [Pause evidence](boundary_research_pause_evidence_20261011.json), SHA256 `3a1c8593431ef96dfff264da098cadffdd43c71baac5765179908f2772413caa`, inventories 193 restart files totaling 2,328,396,098 bytes. It also records successful-worker metadata, interruption, identities and incomplete work.
- All 173 files in the September inventory were rechecked against their original byte counts and SHA256 values; none changed. Both attempts remain local under the existing ignore rules. Market data, matrices and fitted checkpoints are not included in the Git push.
- The exact historical source is `artifacts/midpoint-replay-20261011/source`. Its fresh result directory is `results/boundary_midpoint_conversion_restart_20261011` inside that source export. The current checkout has a same-named result alias pointing there. The original September attempt remains separate.
- The [committed parent log](boundary_midpoint_conversion_restart_pause_20261011.log.txt) is byte-identical to `artifacts/midpoint-replay-20261011/run.log` and preserves the interruption.
- The [verification contract](boundary_midpoint_restart_verification_contract_20261011.md) and [unexecuted E2E source draft](drafts/run_midpoint_restart_verification_e2e.py.txt) preserve the work in progress. The draft was written before the proposed verifier, then moved byte-for-byte out of runnable scripts. The verifier does not exist, and none of the 29 E2E cases has run. Draft SHA256: `103c6aad3901a4db47711698a286b645ed1cb7211a2fc41b34c9c60f84f21e12`.

The pre-fit restart registration verified all 2,158 frozen input contents. This pause checks their saved identity mapping, the source archive and both partial-output inventories; it does not repeat the full input-content scan. No model implementation was changed and no training, unit-test suite or E2E suite was restarted to prepare the handoff.

To repeat the local artifact preservation check from the current repository root, without decoding labels or probabilities:

```sh
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
for date in ("20260909", "20261011"):
    record = json.loads((root / f"docs/research/boundary_research_pause_evidence_{date}.json").read_text())
    expected = record["local_artifacts"]
    actual = {str(p.relative_to(root)) for p in (root / record["attempt_directory"]).rglob("*") if p.is_file()}
    assert actual == set(expected), (date, "file inventory changed")
    for name, saved in expected.items():
        path = root / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert path.stat().st_size == saved["bytes"], name
        assert digest.hexdigest() == saved["sha256"], name
    print(date, len(expected), "artifacts preserved")
PY
```

## Work saved for later

An unfinished research idea is to model observed quote transitions jointly with their durations, then derive the five-second outcome distribution. The review identified requirements around first/last quotes sharing a timestamp, residual waiting time, exact entry/exit clocks, the random entry spread, and reversals. A naive fixed-step transition model could silently change the target. No such model, frozen experiment or measured gain exists; no further design work was completed after the pause request.

After a new instruction to continue, first verify both inventories, all frozen inputs, source and runtime. Preserve both interrupted directories. The current runner rejects an existing output directory, so register an explicit continuation or a new attempt and account for reused and repeated fits before starting. Restore the saved E2E draft to its intended script location only when implementing and verifying the additive compatibility checker. Complete the fixed family and compatibility checks before reading scores, then use the unchanged assessor. Broader development and independent confirmation still require their original gates. Predictive gains require separate economic validation before a trading-performance claim.
