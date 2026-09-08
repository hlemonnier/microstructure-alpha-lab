# Historical learned retrieval

The completed family does **not** beat the strongest retained development procedure. Its best new blend, `observations_nonlinear_nca_deep500_hgb`, reaches **57.2976%** balanced accuracy versus **57.3408%** for `combined_instant_hgb_old_neural`: **−0.0432 percentage points**. BTC loses 0.2340 points and ETH gains 0.1476. Date-average effects are +0.3127, +0.2673 and −0.7097 points. No independent confirmation is triggered.

Learning the distance function substantially improves fixed-distance retrieval. That is a component result against a weak comparator. The observation-only nonlinear retriever gains only **0.0300 points** over its matched parametric network, with ETH degradation. Its tree blend gains 0.1778 points over the matched parametric blend, positive on both assets and all three dates, but remains below the strongest complete procedure.

## Method and controls

The [ModernNCA paper](https://arxiv.org/abs/2407.03257), presented at ICLR 2025 as *Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later*, motivates differentiable neighborhood classification. The [official implementation](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py) uses stochastic training neighborhoods and a full inference bank. These sources motivate the experiment; they do not establish performance on this dependent market dataset. This implementation adds complete query-date exclusion and class weights, uses a different encoder, and omits PLR embeddings. It is not a full paper replication.

The [prospective definition](boundary_retrieval_definition_20260908.json) fixes four procedures for each original representation: normalized Euclidean distance, a learned linear distance, a nonlinear distance, and a matched parametric classifier. Each uses the original historical rows from d−5 through d−2, at calendar stride four seconds. Original frozen quantile transformations, columns and asset priors are preserved. The neural encoder has six epochs; previous-day noon selects its checkpoint. No assessment outcome enters fitting or selection.

During training, a query can retrieve only the same asset on a different historical UTC date. The balanced 3,072-key stochastic bank has 128 observations per asset/date/class cell; inverse sampling weights preserve the historical date mixture within each eligible class. The full historical bank is used at inference with inverse asset/class counts. Kernels use `exp(-Euclidean distance)` and temperature one. Class likelihoods recover natural probabilities with the original historical prior. This ideal prior convention does not prove calibration.

The parametric control has the same nonlinear encoder, optimizer, query batches and stochastic-bank feature exposure, with a three-class output head. Its loss uses query labels; the bank labels influence stratified sampling but are not passed as prediction values. Thus it controls much of the representation-learning machinery, without claiming every operation or compute cost is identical.

All three exposed dates—June 3, 7 and 11, 2023—completed before assessment scores were opened. There are 18 trainable fits, six fixed-metric procedures, 18,144 optimizer steps, 208 retained/new sources, 2,496 source/policy/asset/date panels, and 42,420 unchanged observations per source. Both decision policies are retained; `forecast_3600` remains primary.

## Primary results

| Representation | Fixed distance | Linear retrieval | Nonlinear retrieval | Matched parametric |
|---|---:|---:|---:|---:|
| Original 220 observations | 52.2848% | 56.4233% | 56.6198% | 56.5898% |
| Combined 399 observations | 51.2882% | 56.5643% | 56.4838% | 56.6342% |

Nonlinear retrieval gains 4.3350 and 5.1956 points against the corresponding fixed-distance classifiers. Against the corresponding parametric controls, the effects are +0.0300 and −0.1505 points. The large fixed-distance improvement therefore does not establish a broad advantage for retrieval.

The best new complete blend has natural log loss 0.765782 versus 0.762206 for the retained strongest procedure. Its difference from the original `deep500_hgb_old_neural` blend is −0.0099 balanced-accuracy points. These are descriptive comparisons on repeatedly exposed dates; no confidence interval or independent generalization claim is made.

## Verification and artifacts

Every original historical normalized matrix, row, label, timestamp, prior and retained forecast agrees exactly. Every restored checkpoint reproduces its complete assessment forecasts exactly. Maximum future-prefix error is zero; the largest alternate query/key partition error is 2.7482e−6. The longest fit/save/reload is 136.16 seconds, the longest complete worker is 153.87 seconds, and maximum sampled worker RSS is 1,340,866,560 bytes. These timings include research verification and are not live prediction-latency measurements.

The [preflight evidence](boundary_retrieval_preflight_evidence_20260908.json) preserves all eight passing CPU cases and eight rejected MPS cases. Six retrieval MPS cases failed the registered extreme normalized-input stress test, beyond the production quantile-transform range. The two parametric gradient comparisons did not synchronize CPU/MPS dropout masks; their discrepancy cannot isolate arithmetic error. Neither limitation changes the prospectively selected CPU execution.

The [final protocol](boundary_retrieval_screen_20260908.json) pins 930 inputs, SHA256 `3c17a934eea7147707815509c88f03005880935ec5ab34fbb02b5b62277cf67c`. The local summary is `results/boundary_retrieval_screen_20260908/summary.json`, SHA256 `9b4d9494940ca333579c4bbb7ba5c2b21635bfd8ecc1cdc2c46273fa4fe0b58b`. The [tracked evidence](boundary_retrieval_evidence_20260908.json) contains exact comparisons, worker records and completed artifact inventories. Large matrices and checkpoints remain local and ignored by Git.

The next ordered/Langevin boosting family was defined before these retrieval scores were inspected. Its fixed final tree counts avoid assuming that a prefix of a shrinking final ensemble equals an earlier training checkpoint.
