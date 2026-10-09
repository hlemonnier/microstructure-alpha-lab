# Documentation

The [root README](../README.md) is the project overview. This index separates the methodology, reproduction instructions and dated research record.

## Start here

| Document | What it covers |
|---|---|
| [Research note](research_note.md) | Question, timing, selection, execution and statistical assumptions |
| [Data source reality](data_source_reality.md) | Quote/trade/depth-band data versus replay-grade L2 |
| [Reproducibility](reproducibility.md) | Fresh-install E2E, optional dependencies and evidence artifacts |
| [Research summary](research/README.md) | Completed findings, incomplete validation and the pause record |
| [Model and mathematical remediation](model_math_remediation.md) | September 2026 corrections and their evidence requirements |

## Methodology

- [Pipeline](pipeline.md), [expected edge](expected_edge.md) and [conditional execution](conditional_execution.md).
- [Passive-fill diagnostics](passive_fill_diagnostics.md), [regime analysis](regime_analysis.md) and [cross-asset validation](cross_asset_validation.md).
- [L2 replay](l2_replay.md), [L2 sources](l2_data_sources.md) and [historical L2 acquisition](historical_l2_sources.md).
- [Multiday validation](multiday_validation.md) and [multiple-testing workflow](alpha_factory.md).

## Advanced runbooks

These describe separately scoped empirical workflows. They are not part of the default E2E check.

- [Local source plan](local_free_api_sources.md) and [external market data](../data/README.md).
- [Paper/demo observed fills](paper_demo_fill_sources.md).
- [Cloud handoff](cloud_handoff.md), [full-study runbook](full_study_cloud_run.md) and [Modal execution](modal_full_run.md).

## Historical records

[Implementation traceability](../IMPLEMENTATION_TRACEABILITY.md), [the original implementation backlog](implementation_todo.md) and [the original research plan](research_plan.md) document earlier work. Their dated counts and status statements should be read with the latest [research summary](research/README.md), rather than as current certification.

The files in `research/` preserve registered protocols, dated reports and hash inventories at their original paths. Large local datasets and checkpoints are excluded from Git; an evidence inventory does not imply those inputs are distributed with the repository.
