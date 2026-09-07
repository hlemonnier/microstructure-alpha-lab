"""Strict reader for Binance's eight-column 2023 spot aggregate-trade archives."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

SPOT_COLUMNS = ("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match")


def read_spot_trades(path, *, day_start_ms):
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Exactly one spot CSV member is required")
        with archive.open(members[0]) as source:
            frame = pd.read_csv(source, header=None, float_precision="round_trip")
    if frame.shape[1] != 8 or not len(frame):
        raise ValueError("The registered spot archive has eight columns and no header")
    frame.columns = SPOT_COLUMNS
    for name in ("agg_trade_id", "first_trade_id", "last_trade_id", "transact_time"):
        values = pd.to_numeric(frame[name], errors="raise").to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Spot IDs and milliseconds must be finite integers")
        frame[name] = values.astype(np.int64)
    if (frame[["agg_trade_id", "first_trade_id", "last_trade_id"]] < 0).any().any() or (frame.first_trade_id > frame.last_trade_id).any():
        raise ValueError("Spot trade IDs must be nonnegative and ordered")
    times = frame.transact_time.to_numpy()
    if (np.diff(times) < 0).any() or (np.diff(frame.agg_trade_id.to_numpy()) <= 0).any():
        raise ValueError("Spot timestamps must not regress and aggregate IDs must increase")
    if (times < day_start_ms).any() or (times >= day_start_ms + 86400000).any():
        raise ValueError("The registered 2023 archive must contain UTC-day millisecond timestamps")
    for name in ("price", "quantity"):
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Spot aggregate prices and quantities must be positive and finite")
        frame[name] = values
    if not frame.is_buyer_maker.isin([True, False]).all() or not frame.is_best_match.isin([True, False]).all():
        raise ValueError("Spot maker and best-match flags must be boolean")
    return frame
