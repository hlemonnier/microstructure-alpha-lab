# Proper neural scoring losses: completed development comparison

All 24 registered fits and 3,072 panels are complete. **The alternative losses do not improve the strongest complete forecast procedure.** The best source in this family is the reproduced log-loss observation model blended with deep500 HGB, at 57.3074% balanced accuracy; it recreates an existing procedure. The retained leader remains 57.3408%, with the separately evaluated expert rule at 57.3647%.

The [frozen protocol](boundary_proper_score_screen_20260908.json), [mathematical derivation](boundary_proper_score_math_20260908.md) and [complete evidence](boundary_proper_score_evidence_20260908.json) preserve the intended comparison. Every result below uses the shared primary forecast-prior rule with a 3,600-second half-life. All three dates are exposed development data; they do not establish independent generalization.

| Neural training loss | Observations: balanced accuracy | Combined: balanced accuracy |
| --- | ---: | ---: |
| Original log loss | 56.3108% | 56.2287% |
| Scaled Brier | 56.2101% | 56.2482% |
| Spherical, power two | 56.4254% | 55.3549% |
| Pseudospherical, power four | 55.7647% | 55.9144% |

The observation spherical model gains 0.1146 percentage points over its own log-loss model. Its deep500 blend instead loses 0.2249 points relative to the matching log-loss blend. Combined Brier gains 0.0195 points as an individual and 0.0590 points in its deep500 blend. These small component effects do not become a new accuracy lead.

The best source using an altered loss is combined Brier plus deep500 HGB: 57.1127% balanced accuracy and 0.7630159 natural log loss. It remains below both complete leaders. Power four does not help either representation in this fixed setup.

| Complete procedure | Balanced accuracy | Natural log loss |
| --- | ---: | ---: |
| Reproduced observation log model + deep500 HGB | 57.3074% | 0.7617557 |
| Best altered-loss blend: combined Brier + deep500 HGB | 57.1127% | 0.7630159 |
| Retained instantaneous-feature HGB/neural blend | 57.3408% | 0.7622057 |
| Separate delayed expert-combination rule | 57.3647% | 0.7605550 |

The reproduced log blend is 0.0333 points below the retained leader, with BTC +0.1393 and ETH −0.2060. Its date effects are −0.0797, +0.2224 and −0.2427 points. It is also 0.0572 points below the expert rule. This family supplies no substantial new predictive or economic gain.

## What was controlled and verified

Six fresh log-loss fits reproduce every original neural parameter and original probability exactly. The alternative losses keep the same historical examples, training-only quantile transformation, architecture, initialization seed, optimizer, twelve-epoch budget and previous-noon checkpoint rule. They use the same global historical asset/class weights and natural-posterior conversion. Every model checkpoint reproduces its saved forecasts exactly; all original rows, labels, clocks and priors remain preserved.

The losses elicit the same ideal weighted probability distribution, as derived using [proper-scoring theory](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). Their gradients match log loss at uniform predictions only. Neither strict propriety nor that local scaling proves finite-network calibration, robustness or optimizer convergence. The completed results demonstrate why the empirical comparison is necessary.

The final reporting check distinguishes probability reproduction from secondary decision rules. Legacy neural sources use half historical plus half previous-noon priors under their `registered` tag; the new proper-score sources use historical priors. The initial helper wrongly required equal decisions under those different denominators. The [policy-scope clarification](boundary_decision_policy_scope_20260909.md) records the verified formulas and preserved failed check. Primary forecast-prior decisions and all metrics reproduce exactly. No market artifact, score or protocol was changed to resolve that reporting error.

The fits perform 48,384 optimizer updates. The timed fit/save/replay portions, after input-hash verification, total 117.97 seconds; the largest takes 5.53 seconds. Full supervised process times also include startup and provenance checks. Maximum sampled worker RSS is 765,673,472 bytes. Maximum observed single-query probability deviation is 2.73e-7, below the fixed 2e-6 tolerance; full-query-prefix and future-prefix errors are zero.

The separately registered twenty-date observation-anchor audit follows. It tests broader development transfer without adding a selected loss implicitly. The fresh successor dates remain reserved and require their own frozen confirmation protocol. All substantive accuracy, per-asset, log-loss, nominal uncertainty and later execution requirements remain in force.

Protocol SHA256: `bee030901f5eba05651895fc3976848098657b376e67ca9c0bbb2070f7e32447`.

Summary SHA256: `1bf6695cf72a839318998afb1ff34d1cc6a35977f2df7501bc64e706af7ba459`.

Evidence SHA256: `8b0587a3e235ac9602027e548cc6635f412477470d81f5c5494309b13ba2b2ac`.
