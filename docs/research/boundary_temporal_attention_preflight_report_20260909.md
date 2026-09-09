# Attention runtime preflight: CPU qualifies

All sixteen registered attempts are preserved. Each of the eight CPU architecture/width cases passes the causal-window, finite-gradient, exact checkpoint and exact zero-correction checks. Every MPS attempt stops because this runtime reports the Mac GPU as unavailable. MPS arithmetic was not tested; the zero-valued agreement fields in CPU records are self-comparison placeholders and are not evidence of CPU/GPU equivalence. The unchanged common-backend rule selects CPU.

| Architecture | Input dimensions | Median block time | Median 7,070-query time | Six-epoch projection |
| --- | ---: | ---: | ---: | ---: |
| instant | 221 | 0.00931 s | 0.01501 s | 9.62 s |
| instant | 400 | 0.01020 s | 0.01727 s | 10.55 s |
| uniform | 221 | 0.03470 s | 0.09500 s | 36.32 s |
| uniform | 400 | 0.03384 s | 0.10704 s | 35.59 s |
| age | 221 | 0.03621 s | 0.09069 s | 37.80 s |
| age | 400 | 0.04581 s | 0.11743 s | 47.86 s |
| content | 221 | 0.09972 s | 0.21532 s | 103.70 s |
| content | 400 | 0.10855 s | 0.19400 s | 112.39 s |

Maximum sampled CPU RSS is 571,359,232 bytes. Maximum absolute delta errors are `{'chunking': 4.656612873077393e-10, 'future_prefix': 0.0, 'gap_reset': 0.0, 'shorter_prefix': 0.0}`; the frozen limit is 2e-5. These are synthetic blocks with input-width upper bounds; the verified market representations have 215 and 381 active input dimensions. Full historical-grid allocation, paging, preparation and actual market fitting are excluded from the timing projection and memory measurement.

The market protocol freezes 24 fits: four aggregation rules on two representations across the same three exposed dates. It retains the 256-source proper-score family and adds eight attention sources plus sixteen fixed matched-tree/deep-tree blends, producing 280 sources and 3,360 panels. Instantaneous, uniform, learned-age and content-query attention branches distinguish a useful temporal mechanism from added model capacity. All assessment scores remain sealed until the complete family finishes.

The twenty-date observation audit did not meet the five-point development threshold, so fresh confirmation is not prioritized at this stage. The reserved twenty successor dates remain unopened for market analysis. This preflight proves local numerical/runtime feasibility only, with no new market-accuracy or execution claim.

[Preflight protocol](boundary_temporal_attention_preflight_20260908.json): `6900508da8cdcf3c1a68dbf2938e59db7163a55fa15461667413fb0a9d3cfe17`.

Preflight summary: `55d50858e72d664c08d46cfd426b3e9babee54c05670925b7fb6534b6973a206`.

[Preflight evidence](boundary_temporal_attention_preflight_evidence_20260908.json): `d00dac02f638be111447a25662a4aac648f4ce1c6d9ecdecd3fabc8b34f905b6`.

[Market protocol](boundary_temporal_attention_screen_20260908.json), 1,119 frozen inputs: `e5aea285da9643b20020284106f038ed31ad7c75a3c113246e2dd07f1883c33c`.
