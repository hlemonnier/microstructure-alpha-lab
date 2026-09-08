# Learned temporal corrections: no substantial accuracy gain

The best new GRU blend reaches **57.3346% balanced accuracy**, slightly below the strongest retained development control at **57.3408%**. It improves its own original frozen neural/tree blend by only **0.0272 percentage points**. Although that small paired gain is positive on all three date averages, its BTC average is slightly negative. The new procedure does not meet the substantial-gain criterion.

All eighteen fits and 18,252 optimizer steps completed before scores were opened. The family preserves 42,420 original query observations per source, with 163 sources and two policies producing 1,956 asset/date panels. The primary policy remains prediction-only prior adaptation with a one-hour half-life.

| Representation | Frozen original neural | Memoryless correction | GRU correction | LSTM correction |
|---|---:|---:|---:|---:|
| Original observations | 56.3108% | 56.2712% | 56.2944% | 56.2455% |
| Combined sources | 56.2287% | 56.4012% | 56.1839% | 56.3015% |

The observed-feature GRU gains only 0.0231 points over its memoryless residual control. Both combined-feature recurrent branches lose to their memoryless residual control. The latter improves the original combined neural component by 0.1726 points, positive on both asset and all date averages, but none of these component changes establishes a substantial whole-procedure gain. The best GRU blend's natural log loss is 0.760640, compared with 0.762206 for the strongest retained accuracy control and 0.761756 for its own original blend. Log loss alone does not identify a calibration improvement.

Every branch adds a learned logit correction to the exact original neural checkpoint. Original weights, normalization, active columns and class priors stay frozen. A zero correction reproduces original forecasts exactly. The memoryless branch provides a similar-capacity control for the GRU; parameter counts are recorded, not claimed identical across all branches. The one-layer GRU and LSTM follow PyTorch 2.8 gate conventions.

The eight asset/day training streams advance through every eligible one-second observation, use the original stride-four supervised rows, and reset on gaps. Six epochs use 512-second optimization blocks with numerical state carry and gradient detachment at 128-second or eligibility-pattern boundaries. Epoch selection uses only the previous complete noon. Assessment labels are absent from the fitting interface. The frozen base's training logits are in-sample outputs, not out-of-fold teacher predictions; all three branches share that limitation.

The [prospective definition](boundary_learned_memory_definition_20260908.json) predates the preceding fixed-memory score reveal. The [final protocol](boundary_learned_memory_screen_20260908.json) pins 730 inputs and records the later information boundary. All twelve synthetic CPU/MPS cases pass numerical, gradient, gap and causality checks. The registered aggregate timing rule selects two-thread CPU, which is 10.33 times faster for these small blocks. That local synthetic runtime ratio is not an accuracy gain or a general hardware result.

Six mathematical and learning tests cover prior correction, exact zero residuals, frozen base parameters, independent-stream gradient equivalence, truncation, gaps, prefix and chunk causality, persistence, learning a synthetic delayed signal, and full fitter row accounting. Before market fitting, 639 dependency-complete direct tests pass, along with Ruff and all five CI mypy targets. Minimal CI passes 488 direct cases, with six optional tests and 51 optional modules skipped. Every market checkpoint forecast reloads exactly; the maximum measured future-prefix correction difference is 2.2352e-8, below the frozen 2e-5 tolerance. The longest complete fit/checkpoint/prediction procedure is 61.406 seconds.

The [machine-readable evidence](boundary_learned_memory_evidence_20260908.json) contains asset/date comparisons, fitted parameter counts, chosen epochs and completion hashes. The complete local summary is `results/boundary_learned_memory_screen_20260908/summary.json`, SHA256 `22c8591ba96f5c11610a052687a52d81f2eaf8662124cbc76ecba9f19845ae79`.

Reproduce at the commit matching the pinned source files:

```sh
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/run_boundary_learned_memory_screen.py \
  --protocol docs/research/boundary_learned_memory_screen_20260908.json \
  --output results/boundary_learned_memory_screen_20260908 --run
```

These remain exposed development dates. No second independent confirmation or model promotion follows this result. The next separately registered [feature-budget/context experiment](boundary_tabicl_feature_budget_definition_20260908.json) addresses a different, documented transformer limitation.
