# Counted-depth source preflight passed

The frozen BTC June 1, 2023 replay completed in **251.14 seconds**, including preparation supervision. It parsed all **7,202,684** messages, verified the complete gzip trailer, preserved all **86,397** original decision rows and saved/reloaded every observation exactly. It fitted no model and decoded no assessment labels.

| Known complete depth | Available rows, 100ms delay | Available rows, 500ms delay |
| --- | ---: | ---: |
| 1 level per side | 85,142 | 85,145 |
| 25 levels per side | 85,130 | 85,133 |
| 100 levels per side | 84,988 | 84,995 |

All 7,070 original noon decisions have complete top-100 coverage at both delays for this source. Full-day availability is lower: the source contains five publisher gaps above one second and 23 occasions when a known side empties. Recovery requires a new full snapshot. There are 70,202 updates ignored while awaiting such recovery. There are 5,211,583 updates outside the known price range; these do not extend its completeness claim. These diagnostics concern the conservative reconstructed range, not inferred cancellation or execution events.

The arrays occupy 834,940,624 bytes before compression and 107,729,160 bytes in the saved sidecar. The frozen one-GiB array and 1,800-second wall limits are respected. The original exact source audit and this reconstruction have identical message counts, source endpoints and uncompressed source size. Twelve focused parser/reconstructor tests passed, including strict equal-time exclusion, future-prefix and query-partition invariance, gap recovery, known-range shrinkage, atomic message ordering, and corrupted-gzip rejection.

The original 28-file acquisition completed with 26 verified archives and two BTC files failing after their fixed transport timeouts. Both failed prefixes and logs remain intact, and a separately frozen recovery resumes copies of those prefixes. The next preparation partition covers the **25 newly verified sources**; the already prepared BTC June 1 sidecar will be reused in the complete 28-session join. The two recovering sources require their own preparation partition after byte verification. No model fitting starts before the complete original date/asset cohort is available.

The source preflight demonstrates usable coverage and feasible processing cost. It does not establish predictive improvement. The separately reserved June 13–July 2 sources remain unopened for assessment.

- Preflight protocol SHA256: `b0102105e05f869f41008c80277fbf5d763b6c15c22b560ea77b83546111609e`.
- Observation SHA256: `a84e965e18239e4fa33b54208643603f4229a2e6cd61f238c6aa56ba2c77b69d`.
- [Tracked evidence](boundary_okx_depth_preparation_evidence_20260909.json), SHA256 `6dc3d44c90cb672fff6f0991c045d8cfdcf1e90ce16ec4b2a9d56e932e5e0316`.
- [25-source preparation protocol](boundary_okx_depth_development_part_a_20260909.json), SHA256 `985118b078fdc4f773959d742838025240c59aea957bc7b6b304fd95cec9daea`.
