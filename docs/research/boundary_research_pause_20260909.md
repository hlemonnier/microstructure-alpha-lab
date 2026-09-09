# Research paused at the user's request — 2026-09-09

Research is paused. The user asked to stop all work, commit and push the current findings, and continue later. **Do not resume research, downloads, model fitting or assessment until the user asks to continue.** Predictive accuracy remains the first priority. The research objective is unfinished; no independent substantial gain has been confirmed.

The active midpoint-conversion experiment was interrupted with SIGINT and exited with code 130 before 14:17:25 UTC. A subsequent OS process inventory found no remaining parent or worker for the experiment. No matching Pinely automation was found. The interruption was requested by the user, not a model failure or a performance-based stopping decision.

All model code, tests, completed reports and the frozen current experiment were already pushed to `main` at `f88ef67ecebdbd982492c7e29e60f27209eb5742`. This handoff adds the pause record. Historical reports' proposed next steps are superseded by this pause.

## Completed findings

| Study | Verified development result | Interpretation |
| --- | --- | --- |
| Twenty-date observation-context blend | 54.0516% balanced accuracy; +3.4431 percentage points over the original registered reference; +0.4821 points over the established blend under the common decision policy | Both assets improve, but the twenty dates were already exposed. The five-point substantial-gain threshold is unmet. |
| Counted-depth screen, 66 fits | Best new procedure: 57.3511% balanced accuracy on three exposed dates; +0.0103 points over the complete retained control | The best procedure uses quantities without order counts. ETH declines. No advancement candidate. |
| Unadjusted quote-currency screen, 36 fits | Best new procedure: 57.3342% balanced accuracy on three exposed dates; -0.0066 points versus the complete retained control | No native-source advancement candidate. |
| Training-only TUSD/USDT measurement diagnostic | A fixed half-spacing trade-side adjustment reduces observed one-second increment variance by 99.86–99.96% on four training dates | Evidence of bid–ask bounce and a representation hypothesis, not a predictive gain or proof of the latent FX price. |

The three-date percentages are not comparable to the twenty-date percentage as estimates of generalization. The current matched midpoint experiment is the predictive test of the measurement hypothesis; it has no accuracy result yet.

The complete reports and evidence remain in Git:

- [Twenty-date accuracy report](boundary_broad_accuracy_report_20260909.md).
- [Counted-depth report](boundary_okx_counted_report_20260909.md).
- [Unadjusted quote-currency report](boundary_quote_currency_report_20260909.md).
- [Conversion measurement report](boundary_conversion_microstructure_report_20260909.md).
- [Midpoint feature preparation](boundary_midpoint_conversion_features_report_20260909.md).
- [Broader source metadata readiness](boundary_quote_currency_broad_inventory_report_20260909.md).

## Exact stopping point

The frozen [midpoint protocol](boundary_midpoint_conversion_screen_20260909.json) has SHA256 `4bd811dbc1a62c445723aab77078bc3963b06a202970934c371ff3994f99446f`. It registers 48 fits, followed by 576 assessment panels, on June 3, 7 and 11, 2023. All three input folds were prepared. All 2,158 pinned inputs were reverified when recording this pause.

Eight fits completed successfully: all observation-representation combinations of FX-only/BTC-native, raw-side/midpoint-side and tree/neural models for June 3. Their checkpoints, probabilities, training records, causality checks and successful supervision records remain under:

`results/boundary_midpoint_conversion_screen_20260909/dates/2023-06-03/`

The ninth fit, `combined_fx100_raw_side_hgb` for June 3, was interrupted. There are 39 unstarted fits. No fold completion marker, final assessment prediction panel or `summary.json` exists. Assessment labels remain sealed for this experiment; no partial accuracy scores were inspected or computed. The old `progress.json` still says `training`; it is a historical last update, not evidence of a live process.

[Pause evidence](boundary_research_pause_evidence_20260909.json) records the eight successful workers, input verification, SHA256 and byte counts for the preserved local artifacts, reference-report hashes, previous test logs and the stopping/resumption contract. The [parent process log](boundary_midpoint_conversion_pause_20260909.log.txt) preserves the interruption. Large data, fitted checkpoints and partial result matrices remain **local**, under the repository's existing ignore rules; the Git push includes the code, findings, protocols and evidence inventory.

The latest completed validation before this documentation-only pause was **722 direct test cases passed, with no optional skips**. The minimal environment passed 492 cases, with 6 optional-dependency tests and 70 optional modules skipped. Test success establishes implementation checks, not predictive performance. No training or test suite was restarted to create this handoff.

## Resume only after a new user instruction

1. Verify the frozen protocol, all pinned inputs and the pause artifact inventory before reuse. Preserve the original interrupted directory unchanged.
2. Register an explicit continuation or a fresh attempt in a new directory. The current runner rejects an existing output directory and does not support implicit resume. A continuation must verify reused workers and account for the interrupted ninth fit; a fresh attempt must count repeated fits. Do not silently delete, overwrite or skip the interrupted attempt.
3. Retain the fixed model family, targets, date partitions, seeds, cost assumptions and advancement rules. Finish all 48 registered procedures before revealing assessment metrics. Run the frozen assessor only after completion and integrity checks.
4. If a candidate meets the development rules, freeze broader validation on the already exposed twenty-date cohort. The 44 missing source archives have verified metadata/checksum availability, totaling 393,438,613 advertised bytes; their bodies have not been downloaded.
5. Preserve June 13–July 2 as unopened independent confirmation data for market analysis. No confirmation round was consumed by these studies or by the pause. A future accuracy improvement also requires separate economic validation before any trading-performance claim.

This pause must also be honored if an automatic continuation occurs. It is neither completion of the research goal nor a claim that further work is blocked.
