# Counted-depth development data ready for predictive testing

All 28 BTC/ETH source days from May 29 through June 11, 2023 are prepared and verified. The completed reconstruction processed **189,484,366 source messages**, retained **2,419,034 original decision rows**, and produced 84 feature groups. Compressed depth/count sidecars occupy 2,695,009,674 bytes. These are data-preparation results; no model accuracy is established by them.

The original 25-source preparation attempt failed on its assumption that every daily archive begins with a snapshot. The preserved amendment waits for a complete snapshot before making any initial book available. Six BTC days in the complete source inventory have leading incremental records. The corrected 25-source run finished successfully, while the three unaffected previously completed sidecars were reused with their exact hashes. No failed attempt was silently resumed or overwritten.

Every source completed its gzip integrity check and exact saved-sidecar replay. Every feature group preserved all original rows and reproduced exactly after Parquet reload. Across the entire source window, usable top-100 coverage is 99.175663% at the 100ms delay and 99.184302% at 500ms; top-25 coverage at 100ms is 99.305136%. Missing observations remain explicitly masked. These percentages cover the full decision grids, not just the assessment windows.

The subsequent predictive screen is fixed at 66 fits and 744 panels, comparing quantity-only and count-inclusive inputs at matched depths, delays, representations and training procedures. Its 1,536 pinned prerequisite hashes include source archives, sidecars, original feature inputs, retained forecasts, checkpoints, model code and the fixed reporting procedure. The quantity controls include absolute displayed-depth scale. No assessment rows, target definitions or independent-confirmation requirements change.

The implementation passed 702 direct test cases with the full local dependency environment. The minimal environment passed 492 cases, with six optional-dependency cases and 64 optional modules skipped. The new freezing procedure verified every prerequisite and completed source; the reporting comparison reproduced the already known retained-versus-expert difference. These checks validate implementation and provenance, not predictive performance.

Source times are publisher timestamps. The historical export lacks arrival clocks and sequence identifiers, so the reconstruction does not establish actual live availability or complete packet delivery. The June 13–July 2 independent reservation remains unopened for assessment.

- Feature manifest: `data/research/boundary_okx_counted_features_20260909/feature_manifest.json`, SHA256 `fc55267769f1e6c6fd84c22cad1ce39c8a110d2db78be2e49a99344cfea531a5`.
- Preparation evidence: `docs/research/boundary_okx_counted_preparation_evidence_20260909.json`, SHA256 `3983a68b6fe8dbf83efc0786e2ce989b3998419859274db018fdecf56b963344`.
- Model protocol: `docs/research/boundary_okx_counted_screen_20260909.json`, SHA256 `ddc43f5c12e401f0941e91c81bb9c0fac52b583859a98b9cedc2a1fca865f143`.
- Model definition: `docs/research/boundary_okx_counted_model_design_20260909.md`.
