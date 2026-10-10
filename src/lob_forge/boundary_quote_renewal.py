"""Synthetic prototype of a timestamp-block, reflected additive renewal law.

This is a conditional-law hypothesis, not a market-trained model or an execution
model. Its pre-code contract is docs/research/boundary_quote_renewal_design_20261011.md.
Only selected rows are decoded into exact shortest-decimal primitive fractions.
No simulated path, duration tail, first/last substep, or query row is discarded.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import resource
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# Set before the optional numerical import. The CLI never starts worker pools.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _variable in THREAD_VARS:
    os.environ[_variable] = "2"

FORMAT = "quote_block_reflected_renewal_v1_synthetic"
CACHE_ALGORITHM = "ordered_first_last_c_r_full_paths_linear_cdf_v2"
COLUMNS = ("update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
           "best_ask_qty", "transaction_time", "event_time")
MARK_FIELDS = ("first_delta", "first_spread", "last_delta", "last_spread", "next_state")
SUMMARY_FIELDS = ("entry_c", "exit_c", "entry_r", "exit_r", "entry_spread", "exit_spread")
ENTRY_MS, EXIT_MS, MAX_BLOCKS = 100, 5100, 5101
INT64_MAX = (1 << 63) - 1
MAX_EXACT_FLOAT_INT = 1 << 53
OBJECTIVE_TOLERANCE = 1e-7
LIMITS = {"selected_rows_per_source": 200000, "selected_blocks_per_source": 100000,
          "unique_marks_per_kernel": 100000, "paths_per_component": 65536,
          "total_cached_paths": 250000, "cache_wall_seconds": 90,
          "source_uncompressed_bytes": 1 << 30, "maximum_integer_bits": 4096,
          "blocks_through_first_exit": MAX_BLOCKS, "substeps_through_first_exit": 10201}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    require(abs(value) <= INT64_MAX, f"{name} exceeds the signed integer clock/resource bound")
    return value


def primitive(value, name):
    number = float(value)
    require(math.isfinite(number), f"Nonfinite {name}")
    result = Fraction(str(number))
    require(max(result.numerator.bit_length(), result.denominator.bit_length())
            <= LIMITS["maximum_integer_bits"], f"Exact {name} exceeds rational resource bound")
    return result


@dataclass(frozen=True)
class Quote:
    update_id: int
    time: int
    bid: Fraction
    ask: Fraction
    bid_quantity: Fraction
    ask_quantity: Fraction

    @property
    def spread(self):
        return self.ask - self.bid

    @property
    def price_sum(self):
        return self.ask + self.bid


@dataclass
class Block:
    time: int
    first: Quote
    last: Quote
    rows: int = 1


def csv_rows(path):
    """Stream a single member, without extracting it or decoding outside prices."""
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if item.filename.lower().endswith(".csv")
                   and not item.is_dir()]
        require(len(members) == 1, "A source ZIP must contain exactly one CSV member")
        require(members[0].file_size <= LIMITS["source_uncompressed_bytes"], "Source byte ceiling exceeded")
        with archive.open(members[0]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                rows = csv.reader(text)
                first = next(rows, None)
                require(first is not None, "Empty source CSV")
                header = tuple(first)
                if set(COLUMNS).issubset(header):
                    columns = header
                else:
                    columns = COLUMNS
                    require(len(first) == len(columns), "Malformed headerless source row")
                    yield dict(zip(columns, first))
                for row in rows:
                    require(len(row) == len(columns), "Malformed source CSV row length")
                    yield dict(zip(columns, row))


def read_blocks(path, start=None, end=None):
    """Inclusive start, exclusive end; chronology metadata precedes price parsing.

    The first row at end is never price-decoded. A query calls this with end=d,
    so a quote at d cannot enter its features. Each selected window initializes
    at its own first block; no prior or following transition is manufactured.
    """
    path = Path(path).resolve()
    if start is not None:
        integer(start, "window_start_inclusive")
    if end is not None:
        integer(end, "coverage_end_exclusive")
    require(start is None or end is None or start < end, "Source window start must precede exclusive end")
    blocks, previous_id, previous_time, denominator, selected_rows = [], None, None, 1, 0
    for row in csv_rows(path):
        timestamp = int(row["event_time"])
        integer(timestamp, "event_time")
        require(previous_time is None or timestamp >= previous_time, "Regressing publisher timestamp")
        if end is not None and timestamp >= end:
            break
        uid = int(row["update_id"])
        integer(uid, "update_id")
        require(previous_id is None or uid > previous_id, "Duplicate or regressing update ID")
        previous_id, previous_time = uid, timestamp
        if start is not None and timestamp < start:
            continue
        bid, ask = primitive(row["best_bid_price"], "bid price"), primitive(row["best_ask_price"], "ask price")
        bq, aq = primitive(row["best_bid_qty"], "bid quantity"), primitive(row["best_ask_qty"], "ask quantity")
        require(bid > 0 and ask > bid, "Quote must have positive bid and strict spread; locked/crossed quote")
        require(bq >= 0 and aq >= 0, "Negative displayed quantity")
        quote = Quote(uid, timestamp, bid, ask, bq, aq)
        denominator = math.lcm(denominator, bid.denominator, ask.denominator)
        selected_rows += 1
        require(selected_rows <= LIMITS["selected_rows_per_source"], "Selected source row resource ceiling exceeded")
        if blocks and blocks[-1].time == timestamp:
            blocks[-1].last = quote
            blocks[-1].rows += 1
        else:
            blocks.append(Block(timestamp, quote, quote))
            require(len(blocks) <= LIMITS["selected_blocks_per_source"], "Selected block resource ceiling exceeded")
    require(blocks, "No quotes in selected source window")
    return blocks, denominator, selected_rows


def read_manifest(path):
    manifest = load_json(path)
    require(manifest.get("data_kind") == "synthetic", "This bounded CLI is authorized for synthetic manifests only")
    require(isinstance(manifest.get("asset"), str) and manifest["asset"], "Missing asset")
    tick = primitive(manifest["minimum_tick"], "minimum tick")
    require(tick > 0, "Minimum tick must be positive")
    require(manifest.get("sources"), "Manifest has no source windows")
    sessions, denominator, seen = [], tick.denominator, set()
    for source in manifest["sources"]:
        archive = Path(source["path"]).resolve()
        require(sha256(archive) == source["sha256"], "Source SHA256 mismatch")
        end = integer(source["coverage_end_exclusive"], "coverage_end_exclusive")
        start = source.get("window_start_inclusive")
        key = (str(archive), start, end)
        require(key not in seen, "Duplicate selected source window")
        seen.add(key)
        blocks, common, rows = read_blocks(archive, start, end)
        if "expected_blocks" in source:
            require(len(blocks) == source["expected_blocks"], "Coverage boundary/expected block count mismatch")
        censor = end - blocks[-1].time
        require(censor >= 1, "Exclusive source boundary must follow last retained timestamp")
        require(censor <= MAX_EXACT_FLOAT_INT, "Censor duration exceeds numerical exposure bound")
        denominator = math.lcm(denominator, common)
        sessions.append({"path": str(archive), "sha256": source["sha256"],
                         "window_start_inclusive": start, "coverage_end_exclusive": end,
                         "censor_duration_at_least": censor, "blocks": blocks, "selected_rows": rows})
    return manifest, sessions, denominator, tick


def endpoint_ledger(blocks, decision, tick, archive):
    times = [block.time for block in blocks]
    known = bisect.bisect_left(times, decision) - 1
    entry = bisect.bisect_left(times, decision + ENTRY_MS)
    future = bisect.bisect_left(times, decision + EXIT_MS)
    require(known >= 0 and future < len(blocks), "Inspection query lacks feature/entry/exit coverage")
    q, a, b = blocks[known].last, blocks[entry].first, blocks[future].first
    delta, threshold = b.price_sum - a.price_sum, max(a.spread, 2 * tick)
    return {"archive": archive, "decision_time": decision, "feature_update_id": q.update_id,
            "feature_event_time": q.time, "entry_update_id": a.update_id, "entry_event_time": a.time,
            "exit_update_id": b.update_id, "exit_event_time": b.time,
            "delta_exact": str(delta), "threshold_exact": str(threshold),
            "label": 1 if delta > threshold else -1 if delta < -threshold else 0}


def inspect(manifest_path, queries_path=None):
    manifest, sessions, denominator, tick = read_manifest(manifest_path)
    report = {"data_kind": "synthetic", "asset": manifest["asset"], "common_denominator": denominator,
              "sessions": [], "endpoint_ledger": []}
    source_map = {}
    for session in sessions:
        blocks = session["blocks"]
        report["sessions"].append({key: value for key, value in session.items() if key != "blocks"})
        report["sessions"][-1].update(complete_transitions=len(blocks) - 1,
            blocks=[{"time": block.time, "first_id": block.first.update_id, "last_id": block.last.update_id,
                     "rows": block.rows, "first_sum_exact": str(block.first.price_sum),
                     "last_sum_exact": str(block.last.price_sum)} for block in blocks])
        source_map.setdefault(session["path"], []).append(blocks)
    if queries_path is not None:
        queries = load_json(queries_path)
        require(queries["asset"] == manifest["asset"], "Inspection query asset mismatch")
        for query in queries["queries"]:
            path = str(Path(query["archive"]).resolve())
            require(path in source_map and len(source_map[path]) == 1, "Inspection query needs one selected source window")
            report["endpoint_ledger"].append(endpoint_ledger(source_map[path][0],
                integer(query["decision_time"], "decision_time"), tick, path))
    return report


def exact_units(value, scale):
    scaled = value * scale
    require(scaled.denominator == 1, "Training primitive cannot be represented at common exact scale")
    return scaled.numerator


def spread_edges(spreads):
    ordered = sorted(spreads)
    # Nearest-rank quantiles; a tie at an edge belongs to the upper bin.
    return sorted(set(ordered[math.ceil(len(ordered) * q) - 1] for q in (0.25, 0.50, 0.75)))


def state_of(quote, edges, scale):
    spread_bin = bisect.bisect_right(edges, quote.spread * scale)
    total = quote.bid_quantity + quote.ask_quantity
    if total == 0:
        imbalance_bin = 7
    else:
        imbalance = (quote.bid_quantity - quote.ask_quantity) / total
        cuts = [Fraction(-1) + Fraction(2 * i, 7) for i in range(1, 7)]
        imbalance_bin = bisect.bisect_right(cuts, imbalance)
    state = spread_bin * 8 + imbalance_bin
    require(0 <= state < 32, "State ID outside declared spread/imbalance domain")
    return state


def transition_counts(sessions, edges, scale):
    completed, censored = {state: Counter() for state in range(32)}, {state: Counter() for state in range(32)}
    for session in sessions:
        blocks = session["blocks"]
        for previous, future in zip(blocks, blocks[1:]):
            gap = future.time - previous.time
            require(1 <= gap <= MAX_EXACT_FLOAT_INT, "Complete duration outside positive numerical exposure domain")
            state, next_state = state_of(previous.last, edges, scale), state_of(future.last, edges, scale)
            mark = (exact_units(future.first.price_sum - previous.last.price_sum, scale),
                    exact_units(future.first.spread, scale),
                    exact_units(future.last.price_sum - previous.last.price_sum, scale),
                    exact_units(future.last.spread, scale), next_state)
            completed[state][(gap, mark)] += 1
        state = state_of(blocks[-1].last, edges, scale)
        censored[state][session["censor_duration_at_least"]] += 1
    return completed, censored


def fit_kernel(complete_counts, censor_counts, iterations):
    import numpy as np

    require(complete_counts, "A kernel needs observed complete joint transitions")
    marks = sorted({mark for _, mark in complete_counts})
    require(len(marks) <= LIMITS["unique_marks_per_kernel"], "Joint mark support resource ceiling exceeded")
    index = {mark: i for i, mark in enumerate(marks)}
    records = sorted(complete_counts.items())
    gap_minus_one = np.asarray([gap - 1 for (gap, _), _count in records], dtype=float)
    mark_index = np.asarray([index[mark] for (_gap, mark), _count in records], dtype=int)
    counts = np.asarray([count for _key, count in records], dtype=float)
    censor_records = sorted(censor_counts.items())
    censor_minus_one = np.asarray([c - 1 for c, _count in censor_records], dtype=float)
    censor_n = np.asarray([count for _c, count in censor_records], dtype=float)
    base = np.zeros(len(marks), dtype=float)
    np.add.at(base, mark_index, counts)
    base /= counts.sum()
    mean_gap = float(np.dot(counts, gap_minus_one + 1) / counts.sum())
    p = np.asarray([1 / (1 + 0.35 * mean_gap), 1 / (1 + 2 * mean_gap)])
    weights = np.asarray([0.5, 0.5])
    emissions = np.tile(base, (2, 1))

    def likelihood_and_responsibilities():
        logs = (np.log(weights)[None, :] + np.log(p)[None, :]
                + gap_minus_one[:, None] * np.log1p(-p)[None, :]
                + np.log(emissions[:, mark_index].T))
        complete_max = logs.max(axis=1)
        complete_exp = np.exp(logs - complete_max[:, None])
        complete_norm = complete_exp.sum(axis=1)
        responsibility = complete_exp / complete_norm[:, None]
        objective = float(np.dot(counts, complete_max + np.log(complete_norm)))
        if len(censor_records):
            censor_logs = np.log(weights)[None, :] + censor_minus_one[:, None] * np.log1p(-p)[None, :]
            censor_max = censor_logs.max(axis=1)
            censor_exp = np.exp(censor_logs - censor_max[:, None])
            censor_norm = censor_exp.sum(axis=1)
            censor_responsibility = censor_exp / censor_norm[:, None]
            objective += float(np.dot(censor_n, censor_max + np.log(censor_norm)))
        else:
            censor_responsibility = np.zeros((0, 2))
        objective += float(np.log(p).sum() + np.log1p(-p).sum() + np.log(weights).sum()
                           + (np.log(emissions) * base[None, :]).sum())
        require(math.isfinite(objective), "Nonfinite regularized joint EM objective")
        return objective, responsibility, censor_responsibility

    objective, responsibility, censor_responsibility = likelihood_and_responsibilities()
    trace = [objective]
    for _iteration in range(iterations):
        weighted = counts[:, None] * responsibility
        censor_weighted = censor_n[:, None] * censor_responsibility
        completed_mass = weighted.sum(axis=0)
        failures = (weighted * gap_minus_one[:, None]).sum(axis=0)
        if len(censor_records):
            failures += (censor_weighted * censor_minus_one[:, None]).sum(axis=0)
        p = (completed_mass + 1) / (completed_mass + failures + 2)
        total_mass = completed_mass + censor_weighted.sum(axis=0)
        weights = (total_mass + 1) / (total_mass.sum() + 2)
        for component in range(2):
            tally = np.zeros(len(marks))
            np.add.at(tally, mark_index, weighted[:, component])
            emissions[component] = (tally + base) / (completed_mass[component] + 1)
        objective, responsibility, censor_responsibility = likelihood_and_responsibilities()
        require(objective >= trace[-1] - OBJECTIVE_TOLERANCE, "Regularized joint EM objective decreased")
        trace.append(objective)
    return {"weights": weights.tolist(), "p": p.tolist(), "emissions": emissions.tolist(),
            "marks": [dict(zip(MARK_FIELDS, mark)) for mark in marks], "objective_trace": trace,
            "objective_tolerance": OBJECTIVE_TOLERANCE, "complete_transitions": int(counts.sum()),
            "censored_intervals": int(censor_n.sum()), "aggregated_complete_records": len(records),
            "aggregated_censor_records": len(censor_records), "empirical_emission_prior": base.tolist(),
            "duration_component_ratio": float(max(p) / min(p)),
            "near_coincident_duration_components": bool(max(p) / min(p) < 1.05)}


def probability_vector(values, name, strictly_positive=True):
    require(values and all(math.isfinite(x) and (x > 0 if strictly_positive else x >= 0) for x in values),
            f"Illegal {name} probabilities")
    require(abs(math.fsum(values) - 1) < 1e-12, f"{name} probability mass is not one")


def validate_model(model):
    require(model.get("format_version") == FORMAT and model.get("data_kind") == "synthetic", "Wrong synthetic model format")
    scale = integer(model["scale"], "exact scale")
    require(scale >= 2 and model["floor"] == {"numerator": 1, "denominator": scale}, "Wrong exact half-quantum floor")
    require(model["entry_delay_ms"] == ENTRY_MS and model["exit_delay_ms"] == EXIT_MS, "Target stopping clocks changed")
    tick = integer(model["minimum_tick_units"], "minimum tick units")
    require(tick > 0, "Nonpositive tick")
    edges = model["spread_edges_units"]
    require(len(edges) <= 3 and edges == sorted(set(edges)) and all(isinstance(x, int) and x > 0 for x in edges),
            "Invalid spread-bin edges")
    require(set(model["state_kernel_map"]) == {str(i) for i in range(32)}, "Missing state backoff mapping")
    require(model["kernels"] and all(key in model["kernels"] for key in model["state_kernel_map"].values()),
            "Unknown state kernel")
    maximum_jump, maximum_spread = 0, 0
    for kernel in model["kernels"].values():
        require(len(kernel["weights"]) == len(kernel["p"]) == len(kernel["emissions"]) == 2,
                "Exactly two joint components are required")
        probability_vector(kernel["weights"], "component weights")
        require(all(math.isfinite(p) and 0 < p < 1 for p in kernel["p"]), "Illegal geometric duration probability")
        require(kernel["marks"] and len(kernel["marks"]) <= LIMITS["unique_marks_per_kernel"], "Invalid mark support")
        for row in kernel["emissions"]:
            require(len(row) == len(kernel["marks"]), "Emission/mark support mismatch")
            probability_vector(row, "joint emissions")
        for mark in kernel["marks"]:
            require(set(mark) == set(MARK_FIELDS) and all(isinstance(mark[key], int) for key in MARK_FIELDS), "Malformed joint mark")
            require(mark["first_spread"] > 0 and mark["last_spread"] > 0 and 0 <= mark["next_state"] < 32,
                    "Invalid spread or next-state joint mark")
            require(mark["next_state"] // 8 == bisect.bisect_right(edges, mark["last_spread"]),
                    "Joint mark next state disagrees with observed last-spread bin")
            maximum_jump = max(maximum_jump, abs(mark["first_delta"]), abs(mark["last_delta"]))
            maximum_spread = max(maximum_spread, mark["first_spread"], mark["last_spread"])
    cumulative_bound = MAX_BLOCKS * maximum_jump
    frontier_bound = cumulative_bound + maximum_spread + 2
    require(max(scale, 2 * tick, 2 * cumulative_bound + frontier_bound) <= INT64_MAX,
            "Unsafe integer path bound/overflow; no simulation or arithmetic clipping is allowed")
    return {"maximum_absolute_mark_displacement_units": maximum_jump,
            "maximum_sampled_spread_units": maximum_spread, "cumulative_displacement_bound_units": cumulative_bound,
            "regulator_frontier_bound_units": frontier_bound, "signed_integer_limit": INT64_MAX}


def core_identity(model):
    return canonical_hash({"algorithm": CACHE_ALGORITHM, "format_version": FORMAT,
        "scale": model["scale"], "minimum_tick_units": model["minimum_tick_units"],
        "spread_edges_units": model["spread_edges_units"],
        "entry_delay_ms": ENTRY_MS, "exit_delay_ms": EXIT_MS, "state_kernel_map": model["state_kernel_map"],
        "kernels": {key: {field: kernel[field] for field in ("weights", "p", "emissions", "marks")}
                    for key, kernel in model["kernels"].items()}})


def categorical_cdf(values):
    probability_vector(values, "simulation categorical")
    cumulative, total, compensation = [], 0.0, 0.0
    for value in values:
        adjusted = value - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        total = updated
        cumulative.append(total)
    # Fix the endpoint representation before testing positive support, so even
    # the final category cannot silently lose its representable interval.
    cumulative[-1] = 1.0
    require(all(b > a for a, b in zip([0.0] + cumulative[:-1], cumulative)),
            "Positive categorical mass is unrepresentable by floating cumulative probabilities")
    return cumulative


def geometric(rng, p):
    # Inverse CDF retains the full geometric tail; integers do not wrap.
    value = math.log1p(-rng.random()) / math.log1p(-p)
    require(math.isfinite(value), "Geometric numerical resource failure; tail was not clipped")
    return math.floor(value) + 1


def classify(delta, threshold):
    return 2 if delta > threshold else 0 if delta < -threshold else 1


def simulate_caches(model, paths, seed):
    validate_model(model)
    integer(paths, "simulation paths")
    integer(seed, "simulation seed")
    require(1 <= paths <= LIMITS["paths_per_component"], "Path budget exceeds the bounded synthetic ceiling")
    keys = sorted(set(model["state_kernel_map"].values()))
    require(len(keys) * 2 * paths <= LIMITS["total_cached_paths"], "Total cache resource ceiling exceeded")
    identity = core_identity(model)
    prepared = {key: {"weights_cdf": categorical_cdf(kernel["weights"]),
                     "emission_cdf": [categorical_cdf(row) for row in kernel["emissions"]],
                     "kernel": kernel} for key, kernel in model["kernels"].items()}
    cache, started = {}, time.monotonic()
    for key in keys:
        components, probabilities, block_totals, state_draws, component_draws = [], [], [], [], []
        for initial_component in range(2):
            rng_seed = int(canonical_hash({"core": identity, "initial_kernel": key,
                                          "component": initial_component, "seed": seed}), 16)
            rng = random.Random(rng_seed)
            summary = {field: [] for field in SUMMARY_FIELDS}
            counts, blocks_sampled, later_state_draws, later_component_draws = [0, 0, 0], 0, 0, 0
            for path_index in range(paths):
                if path_index % 32 == 0:
                    require(time.monotonic() - started <= LIMITS["cache_wall_seconds"],
                            "Cache wall-clock resource ceiling exceeded; partial cache abandoned")
                current_key, component, cumulative, frontier, timestamp = key, initial_component, 0, 0, 0
                entry, future = None, None
                for block_number in range(1, MAX_BLOCKS + 1):
                    item = prepared[current_key]
                    kernel = item["kernel"]
                    gap = geometric(rng, kernel["p"][component])
                    timestamp += gap - 1 if block_number == 1 else gap
                    mark_index = bisect.bisect_right(item["emission_cdf"][component], rng.random())
                    mark = kernel["marks"][mark_index]
                    prior_cumulative = cumulative
                    cumulative = prior_cumulative + mark["first_delta"]
                    frontier = max(frontier, mark["first_spread"] + 2 - cumulative)
                    first = (cumulative, frontier, mark["first_spread"])
                    if entry is None and timestamp >= ENTRY_MS:
                        entry = first
                    if timestamp >= EXIT_MS:
                        future = first
                        blocks_sampled += block_number
                        break
                    # The endpoint above uses first, before the same-time last.
                    cumulative = prior_cumulative + mark["last_delta"]
                    frontier = max(frontier, mark["last_spread"] + 2 - cumulative)
                    current_key = model["state_kernel_map"][str(mark["next_state"])]
                    later_state_draws += 1
                    component = bisect.bisect_right(prepared[current_key]["weights_cdf"], rng.random())
                    later_component_draws += 1
                require(entry is not None and future is not None, "Mathematical stopping bound violated; no path truncation")
                values = (entry[0], future[0], entry[1], future[1], entry[2], future[2])
                for field, value in zip(SUMMARY_FIELDS, values):
                    summary[field].append(value)
                counts[classify(future[0] - entry[0], max(entry[2], 2 * model["minimum_tick_units"]))] += 1
            require(sum(counts) == paths, "Simulated component lost probability mass")
            components.append(summary)
            probabilities.append([count / paths for count in counts])
            block_totals.append(blocks_sampled)
            state_draws.append(later_state_draws)
            component_draws.append(later_component_draws)
        cache[key] = {"components": components, "component_probabilities": probabilities,
                      "blocks_sampled": block_totals, "later_state_draws": state_draws,
                      "later_component_draws": component_draws, "discarded_paths": 0, "paths_per_component": paths}
    metadata = {"algorithm": CACHE_ALGORITHM, "core_sha256": identity, "seed": seed, "paths_per_component": paths,
                "full_cached_path_count": len(keys) * 2 * paths, "wall_seconds": time.monotonic() - started,
                "random_generator": "Python Random MT19937; binary64 categorical/inverse geometric draws",
                "discarded_paths": 0, "duration_tail_clipping": False, "limits": LIMITS}
    return cache, metadata


def validate_cache(model, cache, metadata, paths, seed):
    require(metadata["algorithm"] == CACHE_ALGORITHM and metadata["core_sha256"] == core_identity(model)
            and metadata["seed"] == seed and metadata["paths_per_component"] == paths, "Cached law/seed/path identity mismatch")
    require(set(cache) == set(model["state_kernel_map"].values()), "Missing initial transition kernel cache")
    for item in cache.values():
        require(item["discarded_paths"] == 0 and item["paths_per_component"] == paths
                and len(item["components"]) == len(item["component_probabilities"]) == 2, "Invalid complete cache count")
        for component, probabilities in zip(item["components"], item["component_probabilities"]):
            require(set(component) == set(SUMMARY_FIELDS), "Wrong exact path summary fields")
            require(all(len(component[field]) == paths and all(isinstance(value, int) for value in component[field])
                        for field in SUMMARY_FIELDS), "Incomplete or noninteger path summary")
            counts = [0, 0, 0]
            for ce, cx, re, rx, se, sx in zip(*(component[field] for field in SUMMARY_FIELDS)):
                require(se > 0 and sx > 0 and 0 <= re <= rx and re >= se + 2 - ce and rx >= sx + 2 - cx,
                        "Cache reflection support/frontier invariant violated")
                counts[classify(cx - ce, max(se, 2 * model["minimum_tick_units"]))] += 1
            require(probabilities == [count / paths for count in counts], "Cached unreflected probabilities mismatch full mass")
            probability_vector(probabilities, "cached natural", strictly_positive=False)


def fit(manifest_path, iterations=30, minimum_state_transitions=12, paths=256, seed=914):
    integer(iterations, "EM iterations")
    integer(minimum_state_transitions, "minimum state transitions")
    require(1 <= iterations <= 100 and minimum_state_transitions >= 1, "Invalid frozen EM/sparse-state settings")
    manifest, sessions, denominator, tick = read_manifest(manifest_path)
    scale = 2 * denominator
    require(scale <= INT64_MAX, "Common exact denominator exceeds integer simulation bound")
    spreads = []
    for session in sessions:
        for block in session["blocks"]:
            for quote in (block.first, block.last):
                require(exact_units(quote.bid, scale) > 1, "Training quote is not strictly interior to its half-quantum floor")
                require(max(exact_units(quote.bid, scale), exact_units(quote.ask, scale)) <= INT64_MAX,
                        "Primitive training integer overflow; no arithmetic clipping")
            spreads.append(exact_units(block.last.spread, scale))
    edges = spread_edges(spreads)
    completed, censored = transition_counts(sessions, edges, scale)
    pooled_complete, pooled_censored = Counter(), Counter()
    for state in range(32):
        pooled_complete.update(completed[state])
        pooled_censored.update(censored[state])
    require(pooled_complete, "No complete transition in the selected source windows")
    kernels = {"pooled": fit_kernel(pooled_complete, pooled_censored, iterations)}
    mapping, backoff = {}, []
    for state in range(32):
        if sum(completed[state].values()) >= minimum_state_transitions:
            key = f"state_{state}"
            kernels[key] = fit_kernel(completed[state], censored[state], iterations)
            mapping[str(state)] = key
        else:
            mapping[str(state)] = "pooled"
            backoff.append(state)
    model = {"format_version": FORMAT, "data_kind": "synthetic", "asset": manifest["asset"], "scale": scale,
             "common_primitive_denominator": denominator, "minimum_tick_units": exact_units(tick, scale),
             "floor": {"numerator": 1, "denominator": scale}, "spread_edges_units": edges,
             "state_definition": {"spread": "Training block-last nearest-rank quartiles ceil(q*N)-1; duplicate cuts removed; edge ties upper via bisect_right",
                 "imbalance": "(bid_qty-ask_qty)/(bid_qty+ask_qty); seven equal-width bins on [-1,1]; internal cut ties upper; zero total depth category 7",
                 "state_id": "8*spread_bin+imbalance_category; at most32 states"},
             "entry_delay_ms": ENTRY_MS, "exit_delay_ms": EXIT_MS, "seed": seed, "simulation_paths": paths,
             "state_kernel_map": mapping, "kernels": kernels, "cache": {},
             "fit_summary": {"complete_transitions": sum(pooled_complete.values()),
                 "censored_intervals": len(sessions),
                 "censor_duration_at_least": [s["censor_duration_at_least"] for s in sessions],
                 "backoff_states": backoff, "minimum_state_transitions": minimum_state_transitions,
                 "iterations": iterations, "regularization": "Beta(2,2) duration; Dirichlet(2,2) weights; one empirical-emission base mass",
                 "source_windows": [{key: value for key, value in s.items() if key != "blocks"} for s in sessions]},
             "assumptions": ["Unimplemented market hypothesis; only synthetic fitting is authorized by this CLI.",
                 "Coarse-state, stationary, two-component geometric Markov renewal conditional law.",
                 "Joint observed first/last additive innovations; no individual order-event interpretation.",
                 "Explicit persistent one-sided reflected additive price law with bid floor 1/(2Q); no exchange-grid claim.",
                 "Primitive prices are Fraction(str(float(csv_value))); query fractions are not rounded to training quantum.",
                 "Exact .1/5.1-second first-at/after clocks, random entry spread, integer/rational target comparisons.",
                 "Finite deterministic Monte Carlo with binary64 random draws; empirical cached probabilities are approximate.",
                 "No empirical predictive or trading improvement follows from a synthetic E2E pass."],
             "provenance": {"module_sha256": sha256(__file__), "manifest_sha256": sha256(manifest_path),
                            "python": sys.version, "platform": platform.platform(), "thread_environment": {k: os.environ[k] for k in THREAD_VARS}}}
    model["integer_bounds"] = validate_model(model)
    model["cache"], model["cache_metadata"] = simulate_caches(model, paths, seed)
    validate_cache(model, model["cache"], model["cache_metadata"], paths, seed)
    return model


def initial_weights(kernel, age):
    require(1 <= age <= MAX_EXACT_FLOAT_INT, "Initial age outside positive numerical exposure domain")
    logs = [math.log(weight) + (age - 1) * math.log1p(-p) for weight, p in zip(kernel["weights"], kernel["p"])]
    maximum = max(logs)
    weights = [math.exp(value - maximum) for value in logs]
    total = math.fsum(weights)
    normalized = [weight / total for weight in weights]
    require(all(value > 0 for value in normalized),
            f"Positive initial-component survival mass underflow; age_ms={age}, log_weights={logs}, "
            f"normalized_weights={normalized}; whole attempt failed without discarding its mass")
    return normalized


def component_forecast(component, unreflected_probabilities, origin, tick, paths):
    # Use rational numerators so off-grid queries keep their exact boundary ties.
    numerator, denominator = origin.numerator, origin.denominator
    if all(frontier * denominator <= numerator for frontier in component["exit_r"]):
        return unreflected_probabilities, 0.0
    counts, reflected = [0, 0, 0], 0
    for ce, cx, re, rx, se, sx in zip(*(component[field] for field in SUMMARY_FIELDS)):
        push_entry = max(0, re * denominator - numerator)
        push_exit = max(0, rx * denominator - numerator)
        delta_numerator = (cx - ce) * denominator + push_exit - push_entry
        threshold_numerator = max(se, 2 * tick) * denominator
        require(numerator + ce * denominator + push_entry >= (se + 2) * denominator
                and numerator + cx * denominator + push_exit >= (sx + 2) * denominator,
                "Generated endpoint violated physical support; whole attempt failed")
        counts[classify(delta_numerator, threshold_numerator)] += 1
        reflected += push_exit > 0
    require(sum(counts) == paths, "Reflected evaluation lost path mass")
    return [count / paths for count in counts], reflected / paths


def predict(model_path, queries_path, paths=None, seed=None):
    model, queries = load_json(model_path), load_json(queries_path)
    validate_model(model)
    require(queries["asset"] == model["asset"] and queries.get("queries"), "Prediction asset mismatch or empty query universe")
    paths = model["simulation_paths"] if paths is None else paths
    seed = model["seed"] if seed is None else seed
    if (model.get("cache") and model.get("cache_metadata", {}).get("paths_per_component") == paths
            and model.get("cache_metadata", {}).get("seed") == seed):
        cache, metadata = model["cache"], model["cache_metadata"]
    else:
        cache, metadata = simulate_caches(model, paths, seed)
    validate_cache(model, cache, metadata, paths, seed)
    # Read each archive once to its latest required prefix; no future prices are
    # decoded and each decision subsequently selects its own strict prefix.
    maximum_decisions = {}
    for query in queries["queries"]:
        archive = str(Path(query["archive"]).resolve())
        decision = integer(query["decision_time"], "decision_time")
        maximum_decisions[archive] = max(maximum_decisions.get(archive, decision), decision)
    prefixes = {}
    for path, decision in maximum_decisions.items():
        blocks = read_blocks(path, end=decision)[0]
        prefixes[path] = (blocks, [block.time for block in blocks])
    predictions = []
    for query in queries["queries"]:
        archive, decision = str(Path(query["archive"]).resolve()), query["decision_time"]
        blocks, times = prefixes[archive]
        known_index = bisect.bisect_left(times, decision) - 1
        require(known_index >= 0, "Query lacks any strictly predecision quote")
        quote = blocks[known_index].last
        require(quote.bid * model["scale"] >= 1, "Initial query bid is below the exact reflected-law floor")
        state = state_of(quote, model["spread_edges_units"], model["scale"])
        key = model["state_kernel_map"][str(state)]
        weights = initial_weights(model["kernels"][key], decision - quote.time)
        origin = quote.price_sum * model["scale"]
        component_probabilities, reflected_fractions = [], []
        for component, probabilities in zip(cache[key]["components"], cache[key]["component_probabilities"]):
            forecast, fraction = component_forecast(component, probabilities, origin, model["minimum_tick_units"], paths)
            component_probabilities.append(forecast)
            reflected_fractions.append(fraction)
        natural = [sum(weights[r] * component_probabilities[r][label] for r in range(2)) for label in range(3)]
        probability_vector(natural, "query natural", strictly_positive=False)
        predictions.append({"archive": archive, "decision_time": decision, "feature_update_id": quote.update_id,
            "feature_event_time": quote.time, "initial_age_ms": decision - quote.time, "state": state, "kernel": key,
            "origin_sum_exact": str(quote.price_sum), "origin_sum_units_exact": str(origin),
            "initial_component_weights": weights, "natural_probabilities": natural,
            "reflected_fraction": sum(w * f for w, f in zip(weights, reflected_fractions)),
            "positive_support_certified": True, "physical_violation_mass": 0, "discarded_paths": 0,
            "simulation_paths_per_component": paths})
    return {"status": "passed", "data_kind": "synthetic", "asset": model["asset"], "predictions": predictions,
            "probability_order": ["down", "neutral", "up"], "cache_metadata": metadata,
            "model_file_sha256": sha256(model_path), "query_file_sha256": sha256(queries_path),
            "assumptions": model["assumptions"], "market_preflight_passed": False}


def replay_paths(path):
    source = load_json(path)
    require(source["data_kind"] == "synthetic" and source["entry_delay_ms"] == ENTRY_MS
            and source["exit_delay_ms"] == EXIT_MS, "Invalid synthetic replay target")
    result = []
    for item in source["paths"]:
        origin, cumulative, frontier = Fraction(item["origin_sum_units"]), 0, 0
        require((origin - item["initial_spread_units"]) / 2 >= 1, "Replay initial bid below floor")
        entry, future, ledger, previous_time = None, None, [], None
        for block in item["blocks"]:
            require(previous_time is None or block["time"] > previous_time, "Replay block times must increase")
            previous_time, prior_cumulative = block["time"], cumulative
            for kind, delta_field, spread_field in (("first", "first_delta", "first_spread"),
                                                    ("last", "last_delta", "last_spread")):
                cumulative = prior_cumulative + block[delta_field]
                spread = block[spread_field]
                require(spread > 0, "Replay spread must be strictly positive")
                frontier = max(frontier, spread + 2 - cumulative)
                current = origin + cumulative + max(Fraction(0), frontier - origin)
                require((current - spread) / 2 >= 1, "Replay reflection support violated")
                ledger.append({"time": block["time"], "kind": kind, "sum_units": str(current),
                               "spread_units": spread, "c": cumulative, "r": frontier})
                endpoint = (cumulative, frontier, spread, current)
                if kind == "first" and entry is None and block["time"] >= ENTRY_MS:
                    entry = endpoint
                if kind == "first" and block["time"] >= EXIT_MS:
                    future = endpoint
                    break
            if future is not None:
                break
        require(entry is not None and future is not None, "Replay lacks stopping endpoints")
        delta, threshold = future[3] - entry[3], max(entry[2], 2 * source["minimum_tick_units"])
        result.append({"case": item["case"], "origin_sum_units": str(origin), "entry_c": entry[0], "exit_c": future[0],
            "entry_r": entry[1], "exit_r": future[1], "entry_spread": entry[2], "exit_spread": future[2],
            "delta_exact": str(delta), "threshold_units": threshold, "label": classify(delta, threshold) - 1,
            "ledger": ledger})
    return {"paths": result, "data_kind": "synthetic"}


def resource_report(started):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"wall_seconds": time.monotonic() - started, "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "thread_environment": {key: os.environ[key] for key in THREAD_VARS}, "limits": LIMITS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspection = sub.add_parser("inspect")
    inspection.add_argument("--manifest", type=Path, required=True)
    inspection.add_argument("--queries", type=Path)
    inspection.add_argument("--output", type=Path, required=True)
    training = sub.add_parser("fit")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--iterations", type=int, default=30)
    training.add_argument("--min-state-transitions", type=int, default=12)
    training.add_argument("--paths", type=int, default=256)
    training.add_argument("--seed", type=int, default=914)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--queries", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--paths", type=int)
    inference.add_argument("--seed", type=int)
    replay = sub.add_parser("replay-paths")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.model if args.command == "fit" else args.output
    started = time.monotonic()
    try:
        require(not destination.exists(), "Output already exists; preserve the earlier attempt")
        if args.command == "inspect":
            result = inspect(args.manifest, args.queries)
        elif args.command == "fit":
            result = fit(args.manifest, args.iterations, args.min_state_transitions, args.paths, args.seed)
        elif args.command == "predict":
            result = predict(args.model, args.queries, args.paths, args.seed)
        else:
            result = replay_paths(args.input)
        result["resource_report"] = resource_report(started)
        dump_new(destination, result)
        print(json.dumps({"status": "passed", "command": args.command, "artifact": str(destination.resolve()),
                          "wall_seconds": result["resource_report"]["wall_seconds"]}))
        return 0
    except Exception as error:
        failure = {"status": "failed", "command": args.command, "error": f"{type(error).__name__}: {error}",
                   "data_kind": "synthetic", "module_sha256": sha256(__file__), "resource_report": resource_report(started)}
        if not destination.exists():
            dump_new(destination, failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
