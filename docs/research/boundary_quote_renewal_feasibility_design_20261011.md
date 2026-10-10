# Training-only feasibility plan for quote renewal

This is a preparation note, not an execution registration or a predictive result. Finish the fixed midpoint family and the synthetic quote-renewal E2E first. The final source hashes, exact input windows, commands, runtime and numerical/resource ceilings must be frozen in a separate machine-readable registration before market execution.

## Failure cases before a market adapter exists

Reject an unregistered source, checksum mismatch, stale path resolving to different bytes, any reserved confirmation date, an assessment-label input, overlapping output directories, changed source/runtime after registration, missing or extra source windows, an invented transition across a window boundary, split timestamp blocks at a boundary, feature access at or after its decision, a missing query, changed query order, invalid physical support, arithmetic overflow, an incomplete simulation cache, query-dependent randomness, changed save/reload forecasts, unconverged numerical estimates, or exceeded wall/RSS limits. Preserve every failed attempt; never silently retry with different parameters or omit difficult rows.

## Information and comparison boundary

Use only already exposed historical training data. A possible first feasibility window is the BTCUSDT and ETHUSDT bookTicker source for May 29, 2023, restricted to a precisely registered UTC interval. Existing source manifests contain official SHA256 and byte counts. Resolve their historical absolute paths to the current checkout only after exact identity checks. Reading a training window is permitted; the June 13–July 2 reservation remains opaque.

Use explicit half-open source intervals. The first retained publisher-time block initializes the state. The last retained block contributes the interval's right-censor term. Never concatenate two intervals into an observed transition or use a quote at the exclusive endpoint as an in-window event. A source-shaped derived ZIP, if needed for bounded processing, must preserve exact rows, update order and the original-to-derived provenance/hash record.

This feasibility stage measures parsing, exact arithmetic, state/mark cardinality, regularized-likelihood fitting, simulation cost, numerical convergence and deterministic forecasting. It must not rank prediction accuracy, choose a blend, inspect assessment labels or select hyperparameters from a predictive score. Resource-driven implementation repairs remain visible in separate attempts and are followed by the prewritten synthetic E2E.

## Numerical resolution before accuracy

The initial 256-path synthetic fixture establishes wiring only. Freeze a finite sequence of path budgets and independent deterministic seeds before the market preflight. Compare component/state probability estimates and complete causal query forecasts across seeds and doubled budgets. Assess both probability discrepancies and decision stability under a fixed historical-prior rule, with priors sourced only from training data. A small probability error can change a decision near its boundary; report both quantities.

Choose the smallest registered budget meeting all predefined tolerances, using no assessment outcomes. Account for every feasibility fit/cache construction and stop with a computational-limitation result if the largest budget or resource ceiling fails. Do not relax tolerance after seeing accuracy or present Monte Carlo variability as model improvement. A subsequent market screen must use one frozen numerical procedure, common original query rows, the exact five-second target, and the strongest completed supervised controls. Any broader or independent confirmation remains a separate gate.
