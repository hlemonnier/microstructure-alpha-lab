# Temporal-attention backend checks: prepared, not executed

The [operational definition](boundary_temporal_attention_preflight_definition_20260908.json) fixes sixteen cases: four aggregation branches, two input-width upper bounds, and CPU/MPS. It was recorded while CatBoost assessment scores remained sealed. No production-size learner or MPS case has run. This preparation adds no market forecast or accuracy result.

Each worker starts from the registered zero-correction model, performs three warmup updates, and measures eight subsequent updates on an eight-stream 512-second block with a 127-row prefix. Missing inputs include the extreme finite value that previously exposed a gradient-masking defect. Synthetic class weights are computed once and kept fixed. Three query repetitions each produce 7,070 synthetic forecasts.

Two backend comparisons have distinct interpretations. One copies identical CPU post-warmup weights and compares probabilities and every parameter gradient. The other compares probabilities after three independently executed CPU/MPS updates. Fixed tolerances, gap/prefix/chunk checks, exact checkpoint replay, zero-correction reproduction and sampled memory limits apply before a backend can qualify. Timing extrapolation excludes full historical-grid allocation, paging, preparation and actual market fitting.

The backend selector requires all sixteen results, retains failed cases, rejects missing or duplicate cases and nonfinite passing timings, and qualifies only a backend passing every architecture/width case. It minimizes summed median block times, with CPU winning an exact tie. Small orchestration checks pass without fitting a learner. A separate check verifies that protocol freezing refuses the still-incomplete CatBoost prerequisite and writes no final protocol.

The initial manual smoke check summed float32 weights in float32 and missed its own 1e-7 relative assertion by rounding: 1017.00012207 versus 1017. Accumulating the same stored weights in float64 passes. No production code or registered numerical tolerance was changed for this probe; the definition records the discrepancy and the successful check log.

The execution protocol remains pending. After the required boosting, proper-score and twenty-date studies finish, the freezer verifies their complete artifacts and pins their summaries plus the current source closure:

    PYTHONPATH=src:scripts .venv/bin/python scripts/freeze_boundary_temporal_attention_preflight.py --freeze

The previously registered priority still applies: send a candidate supported by the broader audit to fresh confirmation before adding this architecture family. If the attention preflight is then selected for execution, its command is:

    PYTHONPATH=src:data/research/boundary_tabicl_20260908/runtime OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/benchmark_boundary_temporal_attention.py --protocol docs/research/boundary_temporal_attention_preflight_20260908.json --output data/research/boundary_temporal_attention_preflight_20260908

Existing attempts are preserved. The separate attention market protocol and the fresh-date confirmation protocol have not been frozen.
