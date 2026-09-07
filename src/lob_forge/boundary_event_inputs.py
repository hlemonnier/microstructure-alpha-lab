"""Join forecast-only event data within contiguous, observable asset sessions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_events import EVENT_FEATURES
from lob_forge.boundary_forecasts import BASE_FEATURES, feature_frame


def utc_ms(day, clock="00:00:00"):
    return int(pd.Timestamp(f"{day}T{clock}", tz="UTC").timestamp() * 1000)


def representation_columns(frame, representation):
    if representation == "base":
        return list(BASE_FEATURES)
    if representation == "temporal_cross":
        return [name for name in frame.columns if name not in EVENT_FEATURES]
    if representation == "event_cross":
        return list(frame.columns)
    raise ValueError("Unknown event-screen representation")


def load_event_inputs(root: Path, manifest_path: Path, *, verify_hashes=True):
    manifest = json.loads(manifest_path.read_text())
    sessions = {(s["symbol"], s["session_date"]): s for s in manifest["sessions"]}
    features, labels, times = {}, {}, {}
    for day in sorted({day for symbol, day in sessions}):
        raw = {}
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            record = sessions[symbol, day]
            path = root / record["features_path"]
            if verify_hashes and sha256_file(path) != record["sha256"]:
                raise ValueError("Forecast dataset hash mismatch")
            raw[symbol] = pd.read_parquet(path).set_index("decision_time", drop=False)
        common = np.intersect1d(raw["BTCUSDT"].index.to_numpy(), raw["ETHUSDT"].index.to_numpy())
        if not len(common):
            raise ValueError("No simultaneous observable session")
        coverage_end = min(utc_ms(day) + 86400000, int(common[-1])) - 60000
        breaks = np.r_[0, np.flatnonzero(np.diff(common) != 1000) + 1, len(common)]
        collected = {symbol: [] for symbol in raw}
        collected_y = {symbol: [] for symbol in raw}
        collected_t = {symbol: [] for symbol in raw}
        for left, right in zip(breaks[:-1], breaks[1:]):
            if right - left <= 120:
                continue
            group = common[left:right]
            paired = {symbol: frame.loc[group].reset_index(drop=True) for symbol, frame in raw.items()}
            for symbol in raw:
                peer = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
                x = feature_frame(paired[symbol], paired[peer], "temporal_cross")
                x = pd.concat([x, paired[symbol][list(EVENT_FEATURES)]], axis=1)
                eligible = (np.arange(len(group)) >= 120) & (group < coverage_end)
                eligible &= paired[symbol]["quote_age_ms"].to_numpy() <= 1000
                y = paired[symbol]["label"].to_numpy()[eligible]
                if not np.isfinite(y).all():
                    raise ValueError("An eligible decision lacks a future label: preserve and repair source coverage")
                collected[symbol].append(x.loc[eligible])
                collected_y[symbol].append(y.astype(int))
                collected_t[symbol].append(group[eligible])
        for symbol in raw:
            features[symbol, day] = pd.concat(collected[symbol], ignore_index=True)
            labels[symbol, day] = np.concatenate(collected_y[symbol])
            times[symbol, day] = np.concatenate(collected_t[symbol])
    return features, labels, times
