# Research findings and status

**Status: paused since September 2026.** The current repository provides research tooling and historical development evidence. A substantial independent improvement and executable alpha have not been established.

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

Results from three-date and twenty-date cohorts estimate different quantities and should not be ranked directly. Improvements over a weaker component do not establish improvement over the strongest complete control. Predictive accuracy also requires separate economic evaluation.

## Reproduce or inspect

- [Performance pilot reproduction](performance_pilot_reproduction.md): frozen protocols, hashes and matched economic comparisons.
- [Performance research path](performance_research_path.md): the mechanisms investigated after the pilot.
- [Dated research log](boundary_research_log.md): experiment sequence, controls and advancement decisions.
- [Pause evidence](boundary_research_pause_evidence_20260909.json): stopping point, frozen input inventory and preserved local artifacts.

The historical `boundary_*.json` protocols and evidence records stay at their original paths. Their hashes and internal references are preserved. Reproduction may require acquiring external inputs; this directory contains inventories and reports, not the local market datasets or trained models.

## Evidence still required

The full multi-month confirmatory study, immutable final empirical holdout, sufficient genuine L2 neural coverage and calibration against observed trading fills remain pending. Synthetic fixtures verify software behavior only.

Software maintenance does not resume the interrupted experiment. Any future continuation must follow the [explicit resumption contract](boundary_research_pause_20260909.md#resume-only-after-a-new-user-instruction), preserve the old attempt and retain unopened independent confirmation data.
