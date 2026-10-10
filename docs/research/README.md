# Research findings and status

**Status: resumed from the checkpoint at the user's request on October 11, 2026.** The [explicit continuation](boundary_midpoint_continuation_20261011.md) reuses eleven completed workers and executes the remaining 37 fixed procedures. The current repository provides research tooling and historical development evidence. A substantial independent improvement and executable alpha have not been established.

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
| [Explicit continuation](boundary_midpoint_continuation_20261011.md) | User resumed; eleven successful workers copied into a new attempt, 37 remaining fits registered; no new result yet |

Results from three-date and twenty-date cohorts estimate different quantities and should not be ranked directly. Improvements over a weaker component do not establish improvement over the strongest complete control. Predictive accuracy also requires separate economic evaluation.

## Reproduce or inspect

- [Performance pilot reproduction](performance_pilot_reproduction.md): frozen protocols, hashes and matched economic comparisons.
- [Performance research path](performance_research_path.md): the mechanisms investigated after the pilot.
- [Dated research log](boundary_research_log.md): experiment sequence, controls and advancement decisions.
- [Pause evidence](boundary_research_pause_evidence_20260909.json): stopping point, frozen input inventory and preserved local artifacts.
- [Current pause evidence](boundary_research_pause_evidence_20261011.json): restart inventory, successful-worker metadata, interruption and preservation of the earlier attempt.

The historical `boundary_*.json` protocols and evidence records stay at their original paths. Their hashes and internal references are preserved. Reproduction may require acquiring external inputs; this directory contains inventories and reports, not the local market datasets or trained models.

## Evidence still required

The full multi-month confirmatory study, immutable final empirical holdout, sufficient genuine L2 neural coverage and calibration against observed trading fills remain pending. Synthetic fixtures verify software behavior only.

The October 11 restart was paused and subsequently resumed by explicit user instruction. Both interrupted attempts remain preserved, and independent confirmation data remain unopened for market analysis. The eight repeated procedures still require exact array comparison after completion; their repetition adds no independent observations. The continuation uses the unchanged model, target, date and advancement contracts.
