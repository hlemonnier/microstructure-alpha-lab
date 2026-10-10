# Research findings and status

**Status: the resumed 48-fit experiment completed on October 11, 2026.** All 576 panels passed recomputation and all 96 retained panels reproduced exactly. The best new candidate reached 57.313% balanced accuracy versus the complete control’s 57.341%; none passed the frozen expansion gate. Research continues toward predictive accuracy. A substantial independent improvement and executable alpha have not been established.

## Read the findings

| Study | Result and interpretation |
|---|---|
| [Bounded performance pilot](performance_pilot_20260907.md) | Ridge, richer market features and fixed boosting compared under common quote replay on three test dates; descriptive evidence |
| [Independent confirmation](boundary_confirmation_report_20260907.md) | Fell short of the registered substantial-gain requirement |
| [Twenty-date development study](boundary_broad_accuracy_report_20260909.md) | 54.0516% balanced accuracy; +3.4431 percentage points over the original reference, +0.4821 over the established blend under the common policy; the five-point requirement was not met |
| [Counted-depth comparison](boundary_okx_counted_report_20260909.md) | Best new complete procedure improved its retained control by about 0.0103 points on three exposed dates; no advancement candidate |
| [Unadjusted quote-currency comparison](boundary_quote_currency_report_20260909.md) | Best new complete procedure was about 0.0066 points below the retained control; no advancement candidate |
| [Conversion-price diagnostic](boundary_conversion_microstructure_report_20260909.md) | Training-only evidence of bid–ask bounce; the midpoint transform is a representation hypothesis |
| [Midpoint experiment pause](boundary_research_pause_20260909.md) | Eight of 48 fits completed, one interrupted, 39 unstarted; no assessment or predictive result |
| [Explicit midpoint restart](boundary_midpoint_conversion_restart_20261011.md) | Fresh attempt of the unchanged 48-fit family, from an exact historical source snapshot; no new accuracy result yet |
| [October checkpoint](boundary_research_pause_20261011.md) | Restart stopped after 11 completed fits, one interrupted, 36 unstarted; no scores assessed |
| [Quote-renewal prototype](boundary_quote_renewal_synthetic_report_20261011.md) | 44 synthetic E2E checks; numerical stability and market accuracy untested |
| [Completed midpoint continuation](boundary_midpoint_continuation_report_20261011.md) | 48 fits, 576 panels; best new candidate 57.313% versus 57.341% complete control; no advancement |

Results from three-date and twenty-date cohorts estimate different quantities and should not be ranked directly. Improvements over a weaker component do not establish improvement over the strongest complete control. Predictive accuracy also requires separate economic evaluation.

## Reproduce or inspect

- [Completed midpoint evidence](boundary_midpoint_continuation_evidence_20261011.json) and [compatibility verification](boundary_midpoint_continuation_compatibility_20261011.json): original inputs, retained forecasts, full panels and frozen gates.
- [Performance pilot reproduction](performance_pilot_reproduction.md): frozen protocols, hashes and matched economic comparisons.
- [Performance research path](performance_research_path.md): the mechanisms investigated after the pilot.
- [Dated research log](boundary_research_log.md): experiment sequence, controls and advancement decisions.
- [Pause evidence](boundary_research_pause_evidence_20260909.json): stopping point, frozen input inventory and preserved local artifacts.
- [October pause evidence](boundary_research_pause_evidence_20261011.json): restart inventory, successful-worker metadata, interruption and preservation of the earlier attempt.

The historical `boundary_*.json` protocols and evidence records stay at their original paths. Their hashes and internal references are preserved. Reproduction may require acquiring external inputs; this directory contains inventories and reports, not the local market datasets or trained models.

## Evidence still required

The full multi-month confirmatory study, immutable final empirical holdout, sufficient genuine L2 neural coverage and calibration against observed trading fills remain pending. Synthetic fixtures verify software behavior only.

The October 11 restart was paused and subsequently resumed by explicit user instruction. Both interrupted attempts remain preserved, and independent confirmation data remain unopened for market analysis. The eight repeated procedures reproduced all 32 probability/prior arrays exactly. Their repetition adds no independent observations. The completed continuation retained the original model, target, date and advancement contracts. [Execution accounting](boundary_midpoint_completion_accounting_20261011.json) records 48 distinct procedures, 56 completed fit executions across all attempts and two historical interruptions. Both earlier artifact inventories were verified unchanged.
