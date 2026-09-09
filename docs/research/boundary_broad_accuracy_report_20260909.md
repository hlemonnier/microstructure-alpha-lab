# Twenty-date accuracy transfer: a modest, persistent development gain

All forty fits, twenty completed folds and 560 reporting panels are verified. The unchanged observation-context blend reaches **54.0516% balanced accuracy** across 282,800 decisions. It gains **3.4431 percentage points** over the original registered reference, **2.4424 points** over the matched-data control using the same forecast-prior policy, and **0.4821 points** over the established blend using that policy. Both assets improve against each of those controls.

All twenty dates were exposed before this audit. The result is development evidence and does not confirm an independent substantial gain. The original five-point requirement remains unmet even as a development point estimate, so the reserved successor dates remain unopened for assessment.

| Procedure | Primary balanced accuracy | Natural accuracy | Natural log loss |
| --- | ---: | ---: | ---: |
| original_reference | 50.8218% | 54.4859% | 0.9601141 |
| matched_data_control | 51.6092% | 55.7910% | 0.9352612 |
| original_blend | 53.5694% | 58.9632% | 0.8749626 |
| pooled_balanced_tree_blend | 53.5350% | 58.9236% | 0.8754744 |
| observations_tree | 54.0092% | 59.4593% | 0.8661211 |
| observations_neural | 53.5999% | 58.9367% | 0.8743114 |
| observations_blend | 54.0516% | 59.4112% | 0.8660100 |

Every row above uses the same causal forecast-prior rule with a 3,600-second half-life. The original reference under its separately frozen registered rule is 50.6085%; under the common primary rule it is 50.8218%. The observation blend gains 3.2298 points against that latter control on all twenty dates, and reduces natural log loss by 9.8013% relative to the original reference. Changing a decision prior does not change natural accuracy or natural log loss.

| Development subset | Observation blend BA | Established blend BA | Paired difference |
| --- | ---: | ---: | ---: |
| three_recent_development_dates | 57.1421% | 56.3124% | +0.8297 pp |
| other_seventeen_already_exposed_dates | 53.5062% | 53.0854% | +0.4208 pp |

The recent three-date score of 57.1421% overstates this procedure’s broader absolute accuracy. Its advantage over the established blend also contracts from 0.8297 points on those three dates to 0.4208 on the other seventeen. Those seventeen dates are also exposed; this is a descriptive transfer diagnostic. The small three-date leaders from other families were not added to this fixed audit, and their twenty-date performance remains unmeasured.

The strongest complete new procedure gains 0.6960 points for BTC and 0.2683 for ETH against the established blend, with positive mean effects on fifteen of twenty dates. Its neural component adds only 0.0423 points to the new tree alone, with BTC +0.0886 and ETH -0.0039. This is evidence that the observation representation contributes more than this particular neural/tree averaging step.

| Date | Blend minus original registered reference | Blend minus established blend, common policy |
| --- | ---: | ---: |
| 2023-05-24 | +2.1216 pp | +0.6694 pp |
| 2023-05-25 | +4.1360 pp | +0.5248 pp |
| 2023-05-26 | +1.8443 pp | +0.5565 pp |
| 2023-05-27 | +5.0409 pp | +0.6908 pp |
| 2023-05-28 | +5.5224 pp | +1.1871 pp |
| 2023-05-29 | +2.6956 pp | -0.1838 pp |
| 2023-05-30 | +4.1632 pp | +0.4609 pp |
| 2023-05-31 | +2.7330 pp | +0.1618 pp |
| 2023-06-01 | +2.4515 pp | +0.2588 pp |
| 2023-06-02 | +3.4685 pp | -0.0580 pp |
| 2023-06-03 | +6.6115 pp | +0.4762 pp |
| 2023-06-04 | +5.5091 pp | -0.2364 pp |
| 2023-06-05 | +0.9981 pp | -0.5638 pp |
| 2023-06-06 | -0.8646 pp | +1.3891 pp |
| 2023-06-07 | +4.3412 pp | +0.7974 pp |
| 2023-06-08 | +1.5312 pp | +0.9215 pp |
| 2023-06-09 | +2.2750 pp | -0.0076 pp |
| 2023-06-10 | +3.8192 pp | +0.6758 pp |
| 2023-06-11 | +8.2326 pp | +1.2154 pp |
| 2023-06-12 | +2.2317 pp | +0.7068 pp |

All forty checkpoints reproduce their saved probabilities exactly. Six fits on the original three-date subset reproduce the old models, transforms, priors and full forecasts exactly. Every historical label release precedes its next partition; workers receive historical/validation labels and do not decode assessment labels. Maximum observed query probability deviation is 1.22e-7, below the fixed 2e-6 tolerance. Future-prefix and full-query-prefix errors are zero.

The supervised worker processes total 1,244.61 seconds; the longest takes 43.42 seconds. Maximum sampled RSS is 2,165,374,976 bytes. This excludes preparation and the two successful initial fits retained with the documented prior-format failure. The completed retry has forty fits; all initial and retry attempts together total 42. The compatibility amendment changes only exact prior comparison after canonical broadcasting.

The original progress file retains its final training-stage update because the runner does not overwrite it at completion. Completion is established by process exit zero, the matching summary identity and all twenty verified completion markers. No status file or study output was silently rewritten.

The next step is the already defined temporal-attention backend preflight, then its fixed matched-aggregation market comparison if feasible. A future promising procedure must also beat this broader anchor and pass separately frozen independent confirmation. The June 13-July 2 reservation remains opaque, and predictive accuracy does not establish an economic or execution gain.

[Execution amendment](boundary_broad_accuracy_prior_shape_revision_20260909.json): `3aae5cd8e3f01c07a1504bd732776b015b2a81b367b92a6c06c0e43c1611b908`.

Summary: `063ae5603bc6af9b7973021371b1f943a6482bbd8d2c70a900ba753161f11a5f`.

[Complete evidence](boundary_broad_accuracy_evidence_20260909.json): `ba0e1957eb951b04f8836032d6d1d8e236edba5f70fbf7f8b5068efd39a501ee`.
