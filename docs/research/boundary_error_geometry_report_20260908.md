# Completed-model error overlap

The fixed library shares much of its predictive error. Its eight individual models and strongest retained blend achieve a hindsight any-correct balanced accuracy of **70.2112%**, versus **57.3408%** for the reference. This leaves 12.8704 percentage points of errors that another saved decision could repair if the realized label were known. **It does not establish a learnable selector or a predictive gain.**

The diagnostic uses all 42,420 original rows across BTC/ETH on June 3, 7 and 11. These dates and the models' marginal scores were already exposed. The current ordered/Langevin family is excluded and its scores remain sealed. No model or weight was fitted, and no forecast source was promoted.

## What can and cannot be corrected by choosing an expert

All nine saved decisions are wrong on 29.7888% of class-balanced row mass. That accounts for **69.8296% of the reference's balanced error**. They unanimously choose the same wrong class on 25.5303% of balanced mass; disagreement does not flag those errors. The full library is unanimous on 67.5711% of balanced mass, including the 42.0408% on which every model is correct.

| Saved procedure | Balanced accuracy | Reference errors repaired | Reference successes lost |
|---|---:|---:|---:|
| Observation HGB | 56.9814% | 3.4146 pp | 3.7740 pp |
| Observation MLP | 56.3108% | 2.7216 pp | 3.7516 pp |
| Deep-book HGB | 57.0327% | 3.6350 pp | 3.9431 pp |
| Instantaneous-feature HGB | 57.1610% | 3.2920 pp | 3.4718 pp |
| Piecewise-embedding neural model | 57.1224% | 4.5198 pp | 4.7382 pp |
| GRU residual | 56.1839% | 5.0137 pp | 6.1706 pp |
| Selected-feature, 12,288-context TabICL | 56.3236% | 4.7731 pp | 5.7903 pp |
| Nonlinear historical retriever | 56.6198% | 4.2283 pp | 4.9493 pp |
| Retained instantaneous-HGB/MLP blend | 57.3408% | 0 | 0 |

Repairs and losses are measured against the reference on the same rows. Their difference exactly reproduces each procedure's balanced-accuracy difference. The GRU and transformer supply comparatively different decisions, but replacing the reference with either also discards more correct decisions than it repairs.

| Panel group | Reference balanced accuracy | Hindsight any-correct ceiling |
|---|---:|---:|
| BTC | 58.4637% | 71.7005% |
| ETH | 56.2178% | 68.7219% |
| June 3 | 62.3238% | 74.2857% |
| June 7 | 52.3233% | 66.9621% |
| June 11 | 57.3753% | 69.3859% |

The ceiling applies only to selecting among these nine fixed hard decisions on these rows. A new probability transformation can produce a different decision, and new observations or a new fitted model lie outside this finite library. The ceiling is neither a Bayes bound nor evidence that all remaining error is irreducible noise.

## Averaging removes little additional squared error

For natural posterior vectors and an equal mean of the eight individual experts, the exact row-wise squared-error identity gives:

    mean individual Brier = mean-probability Brier + mean squared disagreement
               0.451184 = 0.446667 + 0.004517

This is the vector form of the ambiguity decomposition in [Krogh and Vedelsby](https://proceedings.neurips.cc/paper_files/paper/1994/file/b8c37e33defde51cf91e1e03e51657da-Paper.pdf). It guarantees that the mean beats average individual squared error. It does not guarantee improvement over the strongest individual or improvement in balanced accuracy.

The equal mean lowers Brier by only **0.1458% relative to the best individual** (the piecewise-embedding model, 0.447319), or 0.3174% relative to the retained blend (0.448089). The off-diagonal cosines of natural vector-residuals range from 0.9778 to 0.9989. These are uncentered residual cosines, not Pearson correlations. Both observations support substantial common error among this selected library.

Brier uses uniform row weights within each panel. Decision error overlap uses equal class weights within each panel. Both then average asset/date panels equally. The two differently weighted quantities are not subtracted from one another.

## Research implication and evidence

The diagnostic supports testing whether past data can identify when a different expert is useful; it does not demonstrate that this selection is predictable. A causal selector would need historical forecasts generated without their own target outcomes, delayed-label discipline, fixed learning and selection rules, and a chronological transfer test. [Polley and van der Laan](https://biostats.bepress.com/ucbbiostat/paper266/) motivate fitting combinations from cross-validated predictions. Their asymptotic claims are not assumed to hold for these dependent market rows. No selector is implemented or registered by this diagnostic.

The prepared twenty-date fixed-anchor audit remains the broader transfer check. More data or an observation mechanism that changes the shared errors remains a separate research route; error agreement alone cannot identify which route will work.

The [frozen protocol](boundary_error_geometry_20260908.json) and [complete evidence](boundary_error_geometry_evidence_20260908.json) preserve the exact expert names, source hashes, every panel, class breakdown, disagreement matrix and residual matrix. All completed source artifacts, row labels, clocks and original balanced accuracies were verified. Three targeted tests cover independent repair/loss counts, the pair-distance Brier identity, duplicate and perfect experts, and invalid inputs. Ruff passes. No substantial independent predictive gain is confirmed.
