# Matched OKX quantity and order-count predictive screen

The best new procedure is `observations_q100_100_hgb_old_neural`: **57.351124% balanced accuracy**, a **+0.010336 percentage-point** difference from the retained complete procedure. Its natural log-loss ratio is 0.999583796. These are three repeatedly exposed development dates, not independent confirmation.

The frozen study fitted all 66 models, reported all 744 panels, and retained all 42,420 assessment rows per forecast source. Every reported panel was recomputed from its saved forecasts; all 96 retained panels and forecast arrays reproduced exactly. All checkpoint, training-prior and bounded query-independence checks passed.

| Procedure | Balanced accuracy | Log loss | Gain vs complete control (pp) |
| --- | ---: | ---: | ---: |
| `observations_q100_100_hgb_old_neural` | 57.351124% | 0.761888502 | +0.010336 |
| `combined_instant_hgb_old_neural` | 57.340789% | 0.762205735 | retained control |
| `observations_n100_100_hgb_old_neural` | 57.334644% | 0.761624501 | -0.006145 |
| `observations_n100_500_hgb_old_neural` | 57.294585% | 0.762186353 | -0.046204 |
| `q25_100_hgb` | 57.289636% | 0.760428445 | -0.051153 |
| `observations_n100_100_blend` | 57.269339% | 0.760612480 | -0.071450 |
| `q100_500_hgb_old_neural` | 57.263985% | 0.760449006 | -0.076803 |
| `observations_q100_100_neural_deep_hgb` | 57.256996% | 0.757828646 | -0.083793 |
| `q100_100_hgb_old_neural` | 57.250147% | 0.759875631 | -0.090642 |
| `observations_q100_500_hgb_old_neural` | 57.239076% | 0.762342013 | -0.101713 |
| `n100_100_hgb_old_neural` | 57.236326% | 0.759988694 | -0.104463 |
| `q100_100_hgb31` | 57.231600% | 0.757141391 | -0.109188 |
| `observations_n100_100_neural_deep_hgb` | 57.226239% | 0.759643918 | -0.114550 |
| `q100_100_hgb31_old_neural` | 57.217175% | 0.758099232 | -0.123614 |
| `q25_100_hgb_old_neural` | 57.213741% | 0.759718111 | -0.127047 |
| `q100_100_neural_deep_hgb` | 57.202661% | 0.758646087 | -0.138127 |
| `n25_100_hgb_old_neural` | 57.196612% | 0.759692103 | -0.144177 |
| `n25_100_blend` | 57.191443% | 0.758306059 | -0.149346 |
| `q25_100_neural_deep_hgb` | 57.187077% | 0.759496319 | -0.153711 |
| `q100_100_blend` | 57.183048% | 0.758512123 | -0.157741 |
| `q100_100_hgb` | 57.171708% | 0.760685199 | -0.169081 |
| `q25_100_blend` | 57.163020% | 0.759423425 | -0.177768 |
| `n100_500_hgb_old_neural` | 57.161560% | 0.760334598 | -0.179229 |
| `observations_q100_100_blend` | 57.152035% | 0.758900399 | -0.188753 |
| `n100_100_blend` | 57.146567% | 0.760166586 | -0.194222 |
| `observations_blend` | 57.142060% | 0.763113507 | retained control |
| `n100_100_hgb31` | 57.136837% | 0.758177515 | -0.203952 |
| `n25_100_neural_deep_hgb` | 57.133781% | 0.758598746 | -0.207007 |
| `observations_q100_500_hgb` | 57.125050% | 0.762741743 | -0.215738 |
| `n100_500_blend` | 57.124385% | 0.760554011 | -0.216404 |
| `n25_100_hgb` | 57.120922% | 0.760422013 | -0.219867 |
| `observations_n100_100_hgb` | 57.117810% | 0.761743538 | -0.222979 |
| `n100_100_neural_deep_hgb` | 57.117319% | 0.759903331 | -0.223469 |
| `q100_500_blend` | 57.112841% | 0.760242127 | -0.227948 |
| `observations_q100_100_hgb` | 57.097511% | 0.762120619 | -0.243277 |
| `q100_500_hgb` | 57.084317% | 0.761819595 | -0.256472 |
| `n100_500_neural_deep_hgb` | 57.079748% | 0.760282626 | -0.261041 |
| `observations_q100_500_neural_deep_hgb` | 57.078883% | 0.758853431 | -0.261905 |
| `n100_100_hgb31_old_neural` | 57.049290% | 0.758512471 | -0.291499 |
| `combined_hgb` | 57.045512% | 0.762086580 | retained control |
| `n100_500_hgb` | 57.034559% | 0.761683774 | -0.306229 |
| `deep500_hgb` | 57.032660% | 0.761705480 | retained control |
| `observations_q100_500_blend` | 57.029692% | 0.759983063 | -0.311096 |
| `observations_n100_500_neural_deep_hgb` | 57.025893% | 0.761277871 | -0.314896 |
| `q100_500_neural_deep_hgb` | 56.986741% | 0.759742908 | -0.354048 |
| `observations_tree` | 56.981383% | 0.763488256 | retained control |
| `n100_100_hgb` | 56.942203% | 0.761066371 | -0.398585 |
| `observations_n100_500_blend` | 56.938070% | 0.762485437 | -0.402718 |
| `observations_n100_500_hgb` | 56.923313% | 0.762671527 | -0.417476 |
| `observations_n100_100_neural` | 56.815371% | 0.768884078 | -0.525417 |
| `observations_q100_100_neural` | 56.738699% | 0.763909056 | -0.602090 |
| `observations_q100_500_neural` | 56.548792% | 0.765312679 | -0.791996 |
| `observations_n100_500_neural` | 56.541393% | 0.771997098 | -0.799395 |
| `q100_100_neural` | 56.502837% | 0.766709421 | -0.837952 |
| `q25_100_neural` | 56.497664% | 0.766556390 | -0.843125 |
| `n100_500_neural` | 56.454133% | 0.768082071 | -0.886656 |
| `n25_100_neural` | 56.407713% | 0.768143629 | -0.933076 |
| `observations_neural` | 56.310827% | 0.773227265 | retained control |
| `n100_100_neural` | 56.242927% | 0.773100359 | -1.097862 |
| `combined_neural` | 56.228665% | 0.770318578 | retained control |
| `q100_500_neural` | 56.189622% | 0.768678828 | -1.151166 |
| `original_reference` | 53.137500% | 0.914588427 | retained control |

Matched count-minus-quantity comparisons:

| Count procedure | Mean gain (pp) | BTC gain (pp) | ETH gain (pp) | Log-loss ratio |
| --- | ---: | ---: | ---: | ---: |
| `n25_100_hgb` | -0.168714 | -0.183702 | -0.153726 | 0.999991542 |
| `n25_100_neural` | -0.089951 | -0.037295 | -0.142607 | 1.002070610 |
| `n100_100_hgb` | -0.229505 | -0.191157 | -0.267852 | 1.000501091 |
| `n100_100_neural` | -0.259910 | +0.023145 | -0.542965 | 1.008335542 |
| `n100_500_hgb` | -0.049758 | +0.042597 | -0.142112 | 0.999821716 |
| `n100_500_neural` | +0.264511 | +0.306056 | +0.222966 | 0.999223658 |
| `n100_100_hgb31` | -0.094763 | -0.043837 | -0.145690 | 1.001368468 |
| `observations_n100_100_hgb` | +0.020299 | +0.025079 | +0.015518 | 0.999505221 |
| `observations_n100_100_neural` | +0.076673 | +0.434902 | -0.281556 | 1.006512584 |
| `observations_n100_500_hgb` | -0.201737 | -0.150882 | -0.252593 | 0.999907943 |
| `observations_n100_500_neural` | -0.007399 | +0.079138 | -0.093936 | 1.008734233 |
| `n25_100_blend` | +0.028423 | +0.095453 | -0.038607 | 0.998528665 |
| `n100_100_blend` | -0.036481 | -0.028619 | -0.044343 | 1.002181194 |
| `n100_500_blend` | +0.011544 | -0.014954 | +0.038041 | 1.000410243 |
| `n25_100_hgb_old_neural` | -0.017130 | +0.056065 | -0.090324 | 0.999965767 |
| `n100_100_hgb_old_neural` | -0.013821 | +0.059792 | -0.087433 | 1.000148792 |
| `n100_500_hgb_old_neural` | -0.102425 | -0.058728 | -0.146122 | 0.999849551 |
| `n25_100_neural_deep_hgb` | -0.053296 | +0.219134 | -0.325727 | 0.998818200 |
| `n100_100_neural_deep_hgb` | -0.085342 | -0.162426 | -0.008258 | 1.001657221 |
| `n100_500_neural_deep_hgb` | +0.093007 | +0.037211 | +0.148803 | 1.000710395 |
| `n100_100_hgb31_old_neural` | -0.167885 | -0.286820 | -0.048951 | 1.000545099 |
| `observations_n100_100_blend` | +0.117303 | +0.284246 | -0.049639 | 1.002256002 |
| `observations_n100_500_blend` | -0.091622 | +0.074068 | -0.257312 | 1.003292671 |
| `observations_n100_100_hgb_old_neural` | -0.016480 | +0.002051 | -0.035011 | 0.999653491 |
| `observations_n100_500_hgb_old_neural` | +0.055509 | +0.061282 | +0.049736 | 0.999795813 |
| `observations_n100_100_neural_deep_hgb` | -0.030757 | +0.020392 | -0.081905 | 1.002395359 |
| `observations_n100_500_neural_deep_hgb` | -0.052990 | +0.165106 | -0.271087 | 1.003194872 |

Candidates passing the prospectively fixed development expansion rule: none.

The rule requires at least 0.5 points above the retained complete procedure, improvement on both assets, and a log-loss ratio at most 1.01. Passing this rule only justifies broader evaluation. The unchanged substantial-gain requirement still needs at least five points above the original registered reference and the frozen uncertainty, per-asset and twenty-independent-date requirements. This experiment consumes no independent confirmation round.

Historical publisher-time sampling does not certify live source arrival or execution latency. Published quantities are not converted across venues. Count-specific conclusions require the matched quantity-control comparison; all candidates were selected from exposed development data.

Protocol SHA256: `ddc43f5c12e401f0941e91c81bb9c0fac52b583859a98b9cedc2a1fca865f143`. Full result SHA256: `fb7d757ffb7560ed08da7f1b157e92b964d39081b36714100b4b94fb69c3ce8d`. Tracked evidence: `docs/research/boundary_okx_counted_evidence_20260909.json`, SHA256 `7871e376018f0d16a415ca0c69727f93be3a1f2a8bb874a240ba322c28beb4c6`.
