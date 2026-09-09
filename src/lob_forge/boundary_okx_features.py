"""Matched quantity and order-count observations from one delayed OKX book."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.boundary_bybit_features import _ratio, _signed_log, _transition

OKX_FEATURE_SEMANTICS = "okx_within_book_units_count_ablation_contiguous_history_v1"


def counted_depth_features(canonical, observations, *, delay_ms, levels, include_counts):
    """Keep every canonical row and use counts only in explicitly named columns.

    All quantity normalizations are within the OKX book. No unconverted contract
    quantity is divided by Binance base-currency depth. Q/N is mean order size,
    not an observation of individual order sizes or order flow event identities.
    """
    if levels not in (25, 100) or type(include_counts) is not bool:
        raise ValueError("Use registered top-25/top-100 and explicit count inclusion")
    delays = np.asarray(observations["delays_ms"])
    locations = np.flatnonzero(delays == delay_ms)
    if len(locations) != 1:
        raise ValueError("Exactly one registered assumed publisher delay required")
    slot = int(locations[0])
    clock = canonical.decision_time.to_numpy()
    if (not np.issubdtype(clock.dtype, np.integer) or not len(clock) or (np.diff(clock) <= 0).any()
        or not np.array_equal(clock, observations["decision_times"])):
        raise ValueError("Exact increasing canonical and counted-depth clocks required")
    n = len(clock)
    publisher = np.asarray(observations["publisher_times"][:, slot])
    cutoff = np.asarray(observations["query_cutoffs"][:, slot])
    segment = np.asarray(observations["segment_ids"][:, slot])
    known = np.asarray(observations["known_depths"][:, slot])
    raw = np.asarray(observations["depth"][:, slot, :levels], dtype=float)
    counts = np.asarray(observations["order_counts"][:, slot, :levels])
    if (raw.shape != (n, levels, 4) or counts.shape != (n, levels, 2) or known.shape != (n, 2)
        or publisher.shape != (n,) or cutoff.shape != (n,) or segment.shape != (n,)
        or not np.issubdtype(counts.dtype, np.integer) or not np.isfinite(raw).all()):
        raise ValueError("Aligned finite depth and integer count sidecars required")
    present = publisher >= 0
    available = present & (known.min(axis=1) >= levels)
    if (not np.array_equal(cutoff, ((clock - delay_ms) // 100) * 100)
        or not np.all(publisher[present] < cutoff[present])
        or ((cutoff[present] - publisher[present]) > 1000).any()
        or (segment[present] <= 0).any() or ((known > 0).any(axis=1) & ~present).any()):
        raise ValueError("Counted depth violates the registered causal or coverage contract")
    raw = np.where(available[:, None, None], raw, 0)
    counts = np.where(available[:, None, None], counts, 0).astype(float)
    ap, aq, bp, bq = (raw[:, :, i] for i in range(4))
    an, bn = counts[:, :, 0], counts[:, :, 1]
    if ((raw[available] <= 0).any() or (counts[available] <= 0).any()
        or (np.diff(ap[available], axis=1) <= 0).any() or (np.diff(bp[available], axis=1) >= 0).any()
        or (bp[available, 0] >= ap[available, 0]).any()):
        raise ValueError("Available counted depth must be positive, ordered and uncrossed")
    local_bid, local_ask = (canonical[k].to_numpy(dtype=float) for k in ("bid", "ask"))
    if (not np.isfinite(local_bid).all() or not np.isfinite(local_ask).all()
        or (local_bid <= 0).any() or (local_ask <= local_bid).any()):
        raise ValueError("Current canonical quotes must be finite, positive and uncrossed")
    local_mid, local_spread = (local_bid + local_ask) / 2, local_ask - local_bid
    mid, spread = (ap[:, 0] + bp[:, 0]) / 2, ap[:, 0] - bp[:, 0]
    top_q, top_n = aq[:, 0] + bq[:, 0], an[:, 0] + bn[:, 0]
    basis = _ratio(mid - local_mid, local_mid) * 10000
    imbalance = _ratio(bq[:, 0] - aq[:, 0], top_q)
    values = {"venue_available": available.astype(float),
        "venue_log_age_ms": np.log1p(np.maximum(0, clock - publisher)),
        "venue_mid_basis_bps": basis,
        "venue_log_spread_ratio": np.log1p(_ratio(spread, local_spread)),
        "venue_top_quantity_imbalance": imbalance,
        "venue_microprice_basis_bps": _ratio(_ratio(ap[:, 0] * bq[:, 0] + bp[:, 0] * aq[:, 0], top_q) - local_mid, local_mid) * 10000,
        "venue_bid_basis_spreads": _signed_log(_ratio(bp[:, 0] - local_mid, local_spread)),
        "venue_ask_basis_spreads": _signed_log(_ratio(ap[:, 0] - local_mid, local_spread))}
    history = {}
    for lag in (1, 5, 15):
        valid = np.zeros(n, dtype=bool)
        if n > lag:
            valid[lag:] = (clock[lag:] - clock[:-lag] == lag * 1000) & (segment[lag:] == segment[:-lag])
            for offset in range(lag + 1):
                valid[lag:] &= available[lag - offset:n - offset]
        history[lag] = valid
        values[f"venue_history_available_{lag}s"] = valid.astype(float)
        ret, basis_change, imbalance_change = (np.zeros(n) for _ in range(3))
        ret[lag:] = _ratio(mid[lag:] - mid[:-lag], mid[:-lag]) * 10000
        basis_change[lag:] = basis[lag:] - basis[:-lag]
        imbalance_change[lag:] = imbalance[lag:] - imbalance[:-lag]
        for name, value in (("return_bps", ret), ("basis_change_bps", basis_change), ("quantity_imbalance_change", imbalance_change)):
            values[f"venue_{name}_{lag}s"] = np.where(valid, value, 0)
    bid_distance, ask_distance = _ratio(bp[:, :1] - bp, spread[:, None]), _ratio(ap - ap[:, :1], spread[:, None])
    for rank in (1, 2, 5, 10, 25, 50, 100):
        if rank > levels:
            continue
        i = rank - 1
        for name, value in (("bid_relative_quantity", np.log1p(_ratio(bq[:, i], top_q))),
            ("ask_relative_quantity", np.log1p(_ratio(aq[:, i], top_q))),
            ("bid_distance", np.log1p(bid_distance[:, i])), ("ask_distance", np.log1p(ask_distance[:, i])),
            ("quantity_imbalance", _ratio(bq[:, i] - aq[:, i], bq[:, i] + aq[:, i]))):
            values[f"depth_rank_{rank}_{name}"] = value
        if include_counts:
            for name, value in (("bid_relative_count", np.log1p(_ratio(bn[:, i], top_n))),
                ("ask_relative_count", np.log1p(_ratio(an[:, i], top_n))),
                ("imbalance", _ratio(bn[:, i] - an[:, i], bn[:, i] + an[:, i])),
                ("bid_mean_order_ratio", np.log1p(_ratio(_ratio(bq[:, i], bn[:, i]), _ratio(top_q, top_n)))),
                ("ask_mean_order_ratio", np.log1p(_ratio(_ratio(aq[:, i], an[:, i]), _ratio(top_q, top_n))))):
                values[f"count_rank_{rank}_{name}"] = value
    for k in (1, 5, 10, 25, 50, 100):
        if k > levels:
            continue
        bmass, amass = bq[:, :k].sum(axis=1), aq[:, :k].sum(axis=1)
        mass = bmass + amass
        q_imb = _ratio(bmass - amass, mass)
        b_vwap, a_vwap = _ratio((bp[:, :k] * bq[:, :k]).sum(axis=1), bmass), _ratio((ap[:, :k] * aq[:, :k]).sum(axis=1), amass)
        fair = _ratio(a_vwap * bmass + b_vwap * amass, mass)
        for name, value in (("quantity_imbalance", q_imb), ("relative_quantity", np.log1p(_ratio(mass, top_q))),
            ("weighted_price_spreads", _signed_log(_ratio(fair - mid, spread))),
            ("bid_quantity_concentration", _ratio((bq[:, :k]**2).sum(axis=1), bmass**2)),
            ("ask_quantity_concentration", _ratio((aq[:, :k]**2).sum(axis=1), amass**2)),
            ("bid_span", np.log1p(bid_distance[:, k-1])), ("ask_span", np.log1p(ask_distance[:, k-1]))):
            values[f"depth_{k}_{name}"] = value
        for lag, valid in history.items():
            pressure = _transition(bp[:, :k], ap[:, :k], bq[:, :k], aq[:, :k], lag).sum(axis=1)
            values[f"depth_{k}_sampled_quantity_pressure_{lag}s"] = np.where(valid, _signed_log(_ratio(pressure, mass)), 0)
        if include_counts:
            bnum, anum = bn[:, :k].sum(axis=1), an[:, :k].sum(axis=1)
            num = bnum + anum
            bmean, amean, mean = _ratio(bmass, bnum), _ratio(amass, anum), _ratio(mass, num)
            n_imb = _ratio(bnum - anum, num)
            for name, value in (("log_orders", np.log1p(num)), ("imbalance", n_imb),
                ("bid_mean_order_ratio", np.log1p(_ratio(bmean, mean))),
                ("ask_mean_order_ratio", np.log1p(_ratio(amean, mean))),
                ("mean_order_imbalance", _ratio(bmean - amean, bmean + amean)),
                ("quantity_minus_count_imbalance", q_imb - n_imb),
                ("bid_singleton_level_fraction", (bn[:, :k] == 1).mean(axis=1)),
                ("ask_singleton_level_fraction", (an[:, :k] == 1).mean(axis=1)),
                ("bid_level_count_concentration", _ratio((bn[:, :k]**2).sum(axis=1), bnum**2)),
                ("ask_level_count_concentration", _ratio((an[:, :k]**2).sum(axis=1), anum**2))):
                values[f"count_{k}_{name}"] = value
            for lag, valid in history.items():
                pressure = _transition(bp[:, :k], ap[:, :k], bn[:, :k], an[:, :k], lag).sum(axis=1)
                values[f"count_{k}_sampled_count_pressure_{lag}s"] = np.where(valid, _signed_log(_ratio(pressure, num)), 0)
                change = np.zeros(n)
                change[lag:] = n_imb[lag:] - n_imb[:-lag]
                values[f"count_{k}_imbalance_change_{lag}s"] = np.where(valid, change, 0)
    result = pd.DataFrame({name: np.where(available, value, 0) for name, value in values.items()})
    if not np.isfinite(result.to_numpy()).all() or result.columns.duplicated().any():
        raise ValueError("Finite unambiguous counted-depth features required")
    return result
