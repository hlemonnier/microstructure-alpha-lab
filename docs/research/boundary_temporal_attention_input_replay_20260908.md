# Original temporal inputs and forecasts: exact replay

The attention runner's input preparation passes on June 3, 7 and 11. It verifies the original six-field historical grids, actual outcome release times, global class weights, asset priors and saved original neural forecasts. It fits no model or normalizer and computes no new assessment metric. The [evidence record](boundary_temporal_attention_input_evidence_20260908.json) links the frozen protocol, complete local summary and artifact hashes.

| Representation | Normalized width on all three dates | Historical supervised rows | Validation rows | Assessment rows |
| --- | ---: | ---: | ---: | ---: |
| Observations | 215 | 515,922 | 42,420 | 42,420 |
| Combined | 381 | 515,922 | 42,420 | 42,420 |

The two representations reuse the same labeled observations. Adding their row counts would not double the number of independent examples. Constant-column removal reduces the original raw widths of 220 and 399 to 214 and 380 active columns, respectively, before the asset coordinate is added. Both are covered by the prospectively defined runtime upper bounds of 221 and 400.

Every historical supervised logit is recomputed using the exact frozen original network and its original batch convention, and agrees bit for bit with the grid. All validation and assessment logits and original probabilities also agree exactly. Each asset/date keeps all 7,070 original noon queries. Source labels, decision clocks and priors reproduce the saved controls. Training labels are verified as actually released before validation; validation labels are verified as released before the assessment date.

The check took 29.19 seconds, with 47 verified source artifacts per date and seventeen hashed output artifacts. It uses the already completed retrieval inventory solely to reproduce its unchanged original neural controls while the proper-score study is pending. The future attention experiment still retains the complete 256-source proper-score family and adds eight readouts plus sixteen fixed blends: 280 unique sources and 3,360 panels. The source-name and blend-reference checks pass.

The runner reuses the original grids directly and stores only small validation labels, assessment labels and preparation records. Its fitting worker receives historical supervision and previous-noon validation labels; it does not decode assessment labels. Each final model must preserve original parameters exactly, replay its saved forecasts exactly, and pass prefix/partition checks. Full fitting, checkpoint execution and final panel generation remain pending; this input replay does not prove those later market operations or an accuracy gain.

The existing order remains boosting, proper-scoring, then the twenty-date anchor audit. The separate CPU/MPS attention preflight and market protocol still depend on those completed studies. A candidate ready for fresh confirmation retains priority, and the reserved successor-date archives remain opaque.
