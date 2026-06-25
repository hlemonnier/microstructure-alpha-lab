# L2 Replay

Date checked: 2026-06-03.

`src/lob_forge/l2_replay.py` is the dependency-free deterministic replay layer for normalized market-by-price rows.

## Input Contract

Replay accepts `NormalizedL2Row` rows with:

- `event_type`: `snapshot` or `delta`
- `exchange_timestamp`
- `side`: `bid` or `ask`
- `price`
- `size`
- optional `sequence`
- optional `update_id`

Snapshot rows with the same `(exchange_timestamp, sequence, update_id)` are treated as one snapshot group. A new snapshot group clears book state before applying its first price level. Delta rows update or delete one price level; `size == 0` deletes the level.

## Validation

The replayer reports, per applied row:

- sequence/update gaps,
- book resets,
- crossed-book state,
- current best bid and best ask.

`validate_monotonic_snapshot` checks that bids are descending, asks are ascending, and the book is not crossed.

## Tensor Extraction

`OrderBookReplayer.top_n_tensor(depth=N)` returns rows in a stable LOB tensor order:

```text
ask_price, ask_size, bid_price, bid_size
```

Missing levels are zero-padded by default.

## Current Limits

This is a market-by-price L2 replay layer, not an L3 queue simulator. It does not infer individual order priority, cancellations ahead of our order, or hidden liquidity. Those need live/paper fill calibration before any production passive-execution claim.
