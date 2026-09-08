# Continuous fixed memory: no substantial accuracy gain

The complete comparison's best new blend reaches **57.3408% balanced accuracy**, only **0.0333 percentage points** above the previous strongest control. It uses instantaneous random features, not either temporal mechanism. Its BTC average deteriorates by 0.1393 points, its ETH average improves by 0.2060 points, and June 7 deteriorates by 0.2224 points. Natural log loss also rises slightly, from 0.761756 to 0.762206. This is not a substantial discovery or a candidate for independent promotion.

All 36 fits completed on June 3, 7 and 11 before any new assessment metrics were opened. The family preserves 42,420 original observations per source, with 145 sources and two decision policies producing 1,740 asset/date panels. Every inherited control, label, observation time and prior is checked against its saved original. The primary comparison below uses the unchanged prediction-only one-hour prior adaptation.

| Representation and readout | Instantaneous BA | Exponential-memory BA | Contractive-reservoir BA |
|---|---:|---:|---:|
| Original observations, HGB | 57.0094% | 56.9350% | 56.8337% |
| Original observations, neural | 56.5673% | 56.1467% | 56.4853% |
| Combined sources, HGB | 57.1610% | 57.1100% | 57.1100% |
| Combined sources, neural | 56.6203% | 56.3713% | 56.3590% |

Every temporal individual loses to its own matched instantaneous readout on mean balanced accuracy. Some log losses improve; that alone does not establish calibration improvement. The most successful temporal blend, the combined reservoir neural model averaged with the deeper queue-flow tree, reaches 57.2787%, also below the existing 57.3074% control. These results reject a useful accuracy gain for this specific fixed-memory construction; they do not rule out trained gates, other states or other markets.

The three mechanisms share the same past-fitted normalization, Gaussian input projection, bias and 256 output coordinates. The exponential and reservoir branches contain two 128-coordinate states with leaks 0.1 and 0.01. The reservoir uses an orthogonal recurrent matrix scaled by 0.9. State advances through every eligible one-second observation before the original stride-four supervision is selected, resets at every coverage gap, and never observes outcomes at inference. The contraction bounds describe stability of these states; they do not guarantee forecasting accuracy. Both HGB and neural readouts retain their original fixed fitting procedures.

The [prospective definition](boundary_fading_memory_definition_20260908.json) predates the transformer and delayed-adaptation scores. The [final protocol](boundary_fading_memory_screen_20260908.json) pins 716 inputs and records the later information boundary. Synthetic chunking, gap, future-prefix and state-bound checks passed before market fitting. All original neural checkpoint forecasts and every fitted readout's save/reload forecasts matched exactly. The maximum recorded fit/checkpoint/prediction procedure took 109.001 seconds; this is not an end-to-end trading latency measurement.

The [machine-readable evidence](boundary_fading_memory_evidence_20260908.json) records every matched mechanism comparison and completion hash. The complete local summary is `results/boundary_fading_memory_screen_20260908/summary.json`, SHA256 `ad2a9b61f69ceb7b1861f2285de80b884371addf6feb777548fd8a138b978b05`.

Reproduce at the commit matching the protocol's pinned source files:

```sh
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/run_boundary_fading_memory_screen.py \
  --protocol docs/research/boundary_fading_memory_screen_20260908.json \
  --output results/boundary_fading_memory_screen_20260908 --run
```

These are already exposed development dates. The continuing independent-date error budget and substantial-gain gate remain unchanged. The separately frozen [learned-memory definition](boundary_learned_memory_definition_20260908.json) was written before these fixed-memory scores were revealed.
