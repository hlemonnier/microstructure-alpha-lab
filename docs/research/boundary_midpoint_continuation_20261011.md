# Explicit continuation of the fixed midpoint experiment

The user resumed research on October 11, 2026 and explicitly requested completion of the 48-fit experiment, followed by continued research toward substantial predictive accuracy. This supersedes the pause instruction; the previous pause records remain immutable historical evidence.

## Failure cases, recorded before continuation implementation

Reject changed protocol, source, runtime or frozen inputs; changed file sets, bytes or hashes in either interrupted attempt; old/new directories aliasing or nesting; an existing continuation output; reuse of an incomplete worker; missing or extra reused workers; changed copied preparation, checkpoint, probability, metadata or supervision bytes; wrong model specification; unsuccessful supervision or changed logs; zero/excessive sampled RSS or nonzero GPU driver memory; failed checkpoint/prior or query-causality checks; changed driver after registration; missing/extra completed procedures; and finalization before every registered procedure completes. Preserve failed attempts without overwrite. Do not inspect partial accuracy or confirmation dates.

## Execution contract

Create a new result directory inside the unchanged historical source export. Copy the three prepared input folds and eleven successful October workers byte-for-byte, including their successful supervision/log records. Do not copy the incomplete worker or stale progress. Keep both earlier attempts intact. Use the original model iteration order and original worker CLI to run the remaining 37 procedures once, including the interrupted twelfth procedure. Keep two CPU threads, the 900-second per-fit bound and original sampled-memory limits. The learner, labels, dates, seeds, models, controls and advancement rules remain unchanged.

The additive continuation driver has separate training and finalization phases. Training produces metadata only and cannot call the original finalization functions. After all 48 procedures are present, reverify both interrupted inventories and all frozen inputs, validate every worker's execution metadata, and compare the original eight workers' probability/prior arrays exactly with their repeated versions. Only then call the historical `finish_fold` and `reveal` functions. The independent additive verifier and unchanged 576-panel assessor must pass before interpreting results. Neither copied workers nor repeated runs count as independent observations.

The entire frozen experiment plus the artifact verifiers is the E2E validation of continuation; preserve commands, registration, progress, per-worker supervision, checksums, completion evidence and assessor output. No unit tests are added. Any failure is retained and must be resolved explicitly, without silently rerunning or replacing results.

This is 48 registered date/model procedures: eleven reused and 37 newly executed. Across the September attempt, October restart and this continuation, successful completion would mean 56 completed fit executions, two historical interruptions and 48 distinct completed procedures. The compatibility comparison covers 16 asset pairs and 32 arrays. Research remains incomplete until the original independent accuracy and uncertainty gates pass; predictive improvements still need separate economic evaluation before any trading-performance claim.
