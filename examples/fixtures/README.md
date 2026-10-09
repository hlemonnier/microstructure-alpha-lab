# Synthetic fixtures

These small inputs are deterministic software fixtures. They contain no downloaded market data and support no trading-performance claim.

| File | Purpose |
|---|---|
| `feature_fixture.csv` | Twelve quote/feature rows over three synthetic dates; the third date is held out before threshold selection |
| `l2_replay_fixture.csv` | Snapshot/delta replay, including a deliberately missing sequence for gap detection and Python/C++ parity |
| `l2_sequence_fixture.csv` | Original short L2 fixture, retained unchanged for existing checks |
| `l2_sequence_e2e_fixture.csv` | Sixty-four quote times for the optional TCN/Transformer E2E, with enough rows for holdout exclusion and purged train/validation/test partitions |

The sequence E2E fixture starts with a bid/ask snapshot at timestamp 1000, then alternates one bid or ask delta every 10 time units. Its bid reference is `100 + 0.02 * triangular_phase`, where a 16-step cycle rises from 0 to 8 and falls back to 1; the ask reference adds 0.5. Quantity cycles through 1.0, 1.1 and 1.2. Sequence/update IDs increase by one per quote time, and local timestamps follow exchange timestamps by one unit. This creates rising and falling movements without crossed books or missing updates.

The terminal times 1620 and 1630 are predeclared as the E2E holdout. The 62 development quote times contain enough sequence samples to retain the existing window-plus-horizon purge on both partition boundaries. The smoke uses one epoch on CPU. Increasing fixture coverage does not reduce causal separation or open an empirical holdout.
