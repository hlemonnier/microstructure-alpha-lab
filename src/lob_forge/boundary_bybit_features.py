"""Causal cross-venue quote and depth-curve observations on the Binance clock."""

from __future__ import annotations

import numpy as np
import pandas as pd

BYBIT_FEATURE_SEMANTICS = "cross_venue_top1_vs_top25_observations_v1"


def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def _ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=b > 0)


def _transition(bid, ask, bq, aq, lag):
    result = np.zeros_like(bid)
    result[lag:] = ((bid[lag:] >= bid[:-lag]) * bq[lag:] - (bid[lag:] <= bid[:-lag]) * bq[:-lag]
                   - (ask[lag:] <= ask[:-lag]) * aq[lag:] + (ask[lag:] >= ask[:-lag]) * aq[:-lag])
    return result


def bybit_depth_features(canonical, observations, *, delay_ms, levels):
    """Read only current Binance quotes and already published foreign book states.

    Transition pressure compares sampled endpoint books. It is not the integral
    of every intervening exchange flow, nor identified cancellations or fills.
    The top-one control uses the same depth stream and delay as top-25 features.
    """
    if levels not in (1, 25):
        raise ValueError("Use the registered top-one or top-25 representation")
    delays = np.asarray(observations["delays_ms"])
    locations = np.flatnonzero(delays == delay_ms)
    if len(locations) != 1:
        raise ValueError("Exactly one stored registered information delay required")
    slot = int(locations[0])
    clock = canonical.decision_time.to_numpy()
    if not np.array_equal(clock, observations["decision_times"]) or not len(clock) or (np.diff(clock) <= 0).any():
        raise ValueError("Exact increasing canonical and depth decision clocks required")
    raw = np.asarray(observations["depth"][:, slot], dtype=float)
    available = np.asarray(observations["available"][:, slot], dtype=bool)
    publisher = np.asarray(observations["publisher_times"][:, slot])
    if raw.shape != (len(clock), 25, 4) or available.shape != clock.shape or publisher.shape != clock.shape or not np.isfinite(raw).all():
        raise ValueError("Finite aligned top-25 depth sidecars required")
    present = publisher >= 0
    if not np.all((publisher + delay_ms < clock)[present]) or (available & ~present).any():
        raise ValueError("Foreign depth crossed its strict information boundary")
    raw = np.where(available[:, None, None], raw, 0)
    b, a, bq, aq = (canonical[k].to_numpy(dtype=float) for k in ("bid", "ask", "bid_qty", "ask_qty"))
    if any(not np.isfinite(v).all() for v in (b, a, bq, aq)) or (b <= 0).any() or (a <= b).any() or (bq < 0).any() or (aq < 0).any() or (bq + aq <= 0).any():
        raise ValueError("Valid known Binance quotes and positive displayed depth required")
    ap, av, bp, bv = (raw[:, :, j] for j in range(4))
    if ((ap[available] <= 0).any() or (bp[available] <= 0).any() or (av[available] <= 0).any() or (bv[available] <= 0).any()
        or (np.diff(ap[available], axis=1) <= 0).any() or (np.diff(bp[available], axis=1) >= 0).any()
        or (bp[available, 0] >= ap[available, 0]).any()):
        raise ValueError("Available foreign books require 25 ordered positive uncrossed levels")
    local_mid, local_spread, local_depth = (a + b) / 2, a - b, aq + bq
    mid, spread, top_depth = (ap[:, 0] + bp[:, 0]) / 2, ap[:, 0] - bp[:, 0], av[:, 0] + bv[:, 0]
    basis = _ratio(mid - local_mid, local_mid) * 10000
    imbalance = _ratio(bv[:, 0] - av[:, 0], top_depth)
    microprice = _ratio(ap[:, 0] * bv[:, 0] + bp[:, 0] * av[:, 0], top_depth)
    values = {
        "venue_available": available.astype(float),
        "venue_log_publisher_age_ms": np.log1p(np.maximum(0, clock - publisher)),
        "venue_mid_basis_bps": basis,
        "venue_log_spread_ratio": np.log1p(_ratio(spread, local_spread)),
        "venue_log_depth_ratio": np.log1p(_ratio(top_depth, local_depth)),
        "venue_top_imbalance": imbalance,
        "venue_microprice_basis_bps": _ratio(microprice - local_mid, local_mid) * 10000,
        "venue_bid_basis_spreads": _signed_log(_ratio(bp[:, 0] - local_mid, local_spread)),
        "venue_ask_basis_spreads": _signed_log(_ratio(ap[:, 0] - local_mid, local_spread)),
    }
    for lag in (1, 5, 15):
        valid = np.zeros(len(clock), dtype=bool)
        valid[lag:] = available[lag:] & available[:-lag] & (clock[lag:] - clock[:-lag] == lag * 1000)
        ret, basis_change, imbalance_change, spread_change = (np.zeros(len(clock)) for _ in range(4))
        ret[lag:] = _ratio(mid[lag:] - mid[:-lag], mid[:-lag]) * 10000
        basis_change[lag:] = basis[lag:] - basis[:-lag]
        imbalance_change[lag:] = imbalance[lag:] - imbalance[:-lag]
        spread_change[lag:] = _signed_log(_ratio(spread[lag:] - spread[:-lag], spread[:-lag]))
        pressure = _signed_log(_ratio(_transition(bp[:, 0], ap[:, 0], bv[:, 0], av[:, 0], lag), top_depth))
        for name, value in (("return_bps", ret), ("basis_change_bps", basis_change), ("imbalance_change", imbalance_change),
                            ("spread_change", spread_change), ("transition_pressure", pressure)):
            values[f"venue_{name}_{lag}s"] = np.where(valid, value, 0)
        values[f"venue_history_available_{lag}s"] = valid.astype(float)
    if levels == 25:
        bid_distance = _ratio(bp[:, :1] - bp, spread[:, None])
        ask_distance = _ratio(ap - ap[:, :1], spread[:, None])
        for rank in (1, 2, 4, 9, 24):
            for name, value in (("bid_quantity", np.log1p(_ratio(bv[:, rank], top_depth))),
                ("ask_quantity", np.log1p(_ratio(av[:, rank], top_depth))),
                ("bid_distance", np.log1p(bid_distance[:, rank])), ("ask_distance", np.log1p(ask_distance[:, rank])),
                ("imbalance", _ratio(bv[:, rank] - av[:, rank], bv[:, rank] + av[:, rank]))):
                values[f"depth_rank_{rank + 1}_{name}"] = value
        for count in (5, 10, 25):
            bid_mass, ask_mass = bv[:, :count].sum(axis=1), av[:, :count].sum(axis=1)
            mass = bid_mass + ask_mass
            bid_vwap = _ratio((bp[:, :count] * bv[:, :count]).sum(axis=1), bid_mass)
            ask_vwap = _ratio((ap[:, :count] * av[:, :count]).sum(axis=1), ask_mass)
            fair = _ratio(ask_vwap * bid_mass + bid_vwap * ask_mass, mass)
            for name, value in (("imbalance", _ratio(bid_mass - ask_mass, mass)),
                ("log_depth_ratio", np.log1p(_ratio(mass, top_depth))),
                ("weighted_price_spreads", _signed_log(_ratio(fair - mid, spread))),
                ("bid_span", np.log1p(bid_distance[:, count - 1])), ("ask_span", np.log1p(ask_distance[:, count - 1])),
                ("bid_concentration", _ratio((bv[:, :count]**2).sum(axis=1), bid_mass**2)),
                ("ask_concentration", _ratio((av[:, :count]**2).sum(axis=1), ask_mass**2))):
                values[f"depth_{count}_{name}"] = value
            for lag in (1, 5, 15):
                pressure = _transition(bp[:, :count], ap[:, :count], bv[:, :count], av[:, :count], lag).sum(axis=1)
                values[f"depth_{count}_transition_pressure_{lag}s"] = np.where(values[f"venue_history_available_{lag}s"] > 0,
                    _signed_log(_ratio(pressure, mass)), 0)
        for width in (1, 5, 20):
            bid_weight = (bv * np.exp(-bid_distance / width)).sum(axis=1)
            ask_weight = (av * np.exp(-ask_distance / width)).sum(axis=1)
            values[f"depth_kernel_{width}_imbalance"] = _ratio(bid_weight - ask_weight, bid_weight + ask_weight)
            values[f"depth_kernel_{width}_log_depth_ratio"] = np.log1p(_ratio(bid_weight + ask_weight, top_depth))
    result = pd.DataFrame({name: np.where(available, value, 0) for name, value in values.items()})
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Foreign depth observations must remain finite")
    return result
