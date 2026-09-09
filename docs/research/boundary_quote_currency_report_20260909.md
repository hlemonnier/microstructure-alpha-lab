# Alternative quote-currency predictive screen

The best new procedure is `observations_fx100_hgb_old_neural`: **57.334174% balanced accuracy**, a **-0.006614 percentage-point** difference from the retained complete procedure. Its natural log-loss ratio is 1.000774282. These are three repeatedly exposed development dates, not independent confirmation.

All 36 fits and 456 reporting panels completed. Every source retained all 42,420 assessment rows. All panels were recomputed exactly from saved forecasts; all 96 retained panels and forecast arrays reproduced exactly. All checkpoint, training-prior and query-independence checks passed. These verification results do not themselves establish a predictive gain.

| Procedure | Balanced accuracy | Log loss | Gain vs complete control (pp) |
| --- | ---: | ---: | ---: |
| `combined_instant_hgb_old_neural` | 57.340789% | 0.762205735 | retained control |
| `observations_fx100_hgb_old_neural` | 57.334174% | 0.762795897 | -0.006614 |
| `combined_fx100_hgb` | 57.250942% | 0.762087485 | -0.089846 |
| `combined_both100_hgb_old_neural` | 57.227645% | 0.761704983 | -0.113143 |
| `combined_fx100_blend` | 57.207199% | 0.759421914 | -0.133590 |
| `observations_both100_hgb_old_neural` | 57.165146% | 0.763336728 | -0.175643 |
| `observations_fx100_neural_deep_hgb` | 57.162294% | 0.759372366 | -0.178495 |
| `combined_btc100_neural_deep_hgb` | 57.157633% | 0.758782106 | -0.183156 |
| `combined_both100_neural_deep_hgb` | 57.142819% | 0.760597278 | -0.197970 |
| `observations_blend` | 57.142060% | 0.763113507 | retained control |
| `combined_btc100_blend` | 57.133937% | 0.761237932 | -0.206852 |
| `combined_btc100_hgb_old_neural` | 57.124417% | 0.761678948 | -0.216372 |
| `combined_fx100_hgb_old_neural` | 57.121147% | 0.760834072 | -0.219642 |
| `combined_fx100_neural_deep_hgb` | 57.110091% | 0.759191776 | -0.230698 |
| `observations_btc100_neural_deep_hgb` | 57.109983% | 0.759778238 | -0.230806 |
| `observations_btc100_hgb_old_neural` | 57.109049% | 0.763557850 | -0.231739 |
| `combined_both100_hgb` | 57.070544% | 0.765443608 | -0.270244 |
| `combined_btc100_hgb` | 57.061807% | 0.765178301 | -0.278981 |
| `combined_hgb` | 57.045512% | 0.762086580 | retained control |
| `deep500_hgb` | 57.032660% | 0.761705480 | retained control |
| `observations_fx100_blend` | 57.025269% | 0.760431555 | -0.315519 |
| `observations_btc100_hgb` | 56.983454% | 0.766198740 | -0.357335 |
| `observations_tree` | 56.981383% | 0.763488256 | retained control |
| `combined_both100_blend` | 56.955755% | 0.762970500 | -0.385033 |
| `observations_both100_neural_deep_hgb` | 56.935735% | 0.762165926 | -0.405053 |
| `observations_btc100_blend` | 56.925736% | 0.763197612 | -0.415052 |
| `observations_fx100_hgb` | 56.904865% | 0.763162229 | -0.435924 |
| `observations_both100_hgb` | 56.903495% | 0.766220432 | -0.437294 |
| `observations_both100_blend` | 56.829527% | 0.765859687 | -0.511262 |
| `observations_fx100_neural` | 56.459632% | 0.766324895 | -0.881156 |
| `observations_btc100_neural` | 56.403527% | 0.768017128 | -0.937261 |
| `combined_both100_neural` | 56.394556% | 0.769624411 | -0.946232 |
| `combined_fx100_neural` | 56.392016% | 0.765609168 | -0.948773 |
| `combined_btc100_neural` | 56.364418% | 0.764882027 | -0.976371 |
| `observations_neural` | 56.310827% | 0.773227265 | retained control |
| `combined_neural` | 56.228665% | 0.770318578 | retained control |
| `observations_both100_neural` | 55.950094% | 0.777502472 | -1.390694 |
| `original_reference` | 53.137500% | 0.914588427 | retained control |

Native-source minus matched FX-only procedure:

| Procedure | Mean gain (pp) | BTC gain (pp) | ETH gain (pp) | Log-loss ratio |
| --- | ---: | ---: | ---: | ---: |
| `observations_btc100_hgb` | +0.078589 | +0.071745 | +0.085433 | 1.003978854 |
| `observations_btc100_neural` | -0.056105 | -0.144259 | +0.032049 | 1.002208245 |
| `observations_both100_hgb` | -0.001370 | -0.019067 | +0.016327 | 1.004007278 |
| `observations_both100_neural` | -0.509538 | -0.508354 | -0.510722 | 1.014585950 |
| `combined_btc100_hgb` | -0.189135 | -0.073003 | -0.305267 | 1.004055723 |
| `combined_btc100_neural` | -0.027598 | -0.349830 | +0.294633 | 0.999050245 |
| `combined_both100_hgb` | -0.180398 | -0.095656 | -0.265140 | 1.004403855 |
| `combined_both100_neural` | +0.002540 | -0.331623 | +0.336703 | 1.005244507 |
| `observations_btc100_blend` | -0.099533 | +0.102916 | -0.301983 | 1.003637484 |
| `observations_btc100_hgb_old_neural` | -0.225125 | -0.101167 | -0.349082 | 1.000998894 |
| `observations_btc100_neural_deep_hgb` | -0.052311 | -0.048350 | -0.056273 | 1.000534483 |
| `observations_both100_blend` | -0.195742 | -0.170910 | -0.220574 | 1.007138225 |
| `observations_both100_hgb_old_neural` | -0.169028 | -0.108980 | -0.229077 | 1.000709012 |
| `observations_both100_neural_deep_hgb` | -0.226559 | -0.275307 | -0.177810 | 1.003678775 |
| `combined_btc100_blend` | -0.073262 | -0.096861 | -0.049663 | 1.002391316 |
| `combined_btc100_hgb_old_neural` | +0.003270 | -0.115958 | +0.122498 | 1.001110461 |
| `combined_btc100_neural_deep_hgb` | +0.047542 | -0.012551 | +0.107635 | 0.999460387 |
| `combined_both100_blend` | -0.251444 | -0.072121 | -0.430766 | 1.004672747 |
| `combined_both100_hgb_old_neural` | +0.106498 | -0.004092 | +0.217089 | 1.001144680 |
| `combined_both100_neural_deep_hgb` | +0.032728 | +0.000581 | +0.064875 | 1.001851314 |

Joint-source minus matched BTC-only procedure:

| Procedure | Mean gain (pp) | BTC gain (pp) | ETH gain (pp) | Log-loss ratio |
| --- | ---: | ---: | ---: | ---: |
| `observations_both100_hgb` | -0.079959 | -0.090813 | -0.069105 | 1.000028311 |
| `observations_both100_neural` | -0.453433 | -0.364096 | -0.542771 | 1.012350432 |
| `combined_both100_hgb` | +0.008737 | -0.022653 | +0.040127 | 1.000346725 |
| `combined_both100_neural` | +0.030139 | +0.018207 | +0.042070 | 1.006200150 |
| `observations_both100_blend` | -0.096209 | -0.273827 | +0.081408 | 1.003488054 |
| `observations_both100_hgb_old_neural` | +0.056096 | -0.007813 | +0.120005 | 0.999710407 |
| `observations_both100_neural_deep_hgb` | -0.174247 | -0.226957 | -0.121537 | 1.003142613 |
| `combined_both100_blend` | -0.178182 | +0.024740 | -0.381103 | 1.002275988 |
| `combined_both100_hgb_old_neural` | +0.103229 | +0.111866 | +0.094591 | 1.000034181 |
| `combined_both100_neural_deep_hgb` | -0.014814 | +0.013132 | -0.042760 | 1.002392218 |

Native procedures passing all frozen development expansion gates: none.

A native procedure must gain at least 0.5 points over the complete control and any stronger measured counted-depth or expert reference, and at least 0.25 points over its exact FX-only counterpart. Both assets must improve and each corresponding log-loss ratio must be at most 1.01. Passing justifies broader development evaluation. The unchanged substantial-gain requirement still needs five points above the original frozen reference and its uncertainty, per-asset and twenty-independent-date gates. No independent confirmation round was consumed.

Source activity, publisher-time causality and exact software checks do not establish live source arrival, executable fills or economic returns. All candidate selection here uses exposed development data.

Protocol SHA256: `44a4ecfb495af4123cf56ab81d578aa6bc4c57585cd313e05c12ad256a3363ed`. Result SHA256: `f7a4252b894232e58b42ba2cee0fb602da5878e422372e3ecd7a9c143cf1a0e3`. Evidence SHA256: `6a0c647f6053d6824c100204d3c0c22af30cdfbd66eb0cc1e768a4b61249c708`.
