from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FI2010Sample:
    ask_prices: list[float]
    ask_sizes: list[float]
    bid_prices: list[float]
    bid_sizes: list[float]
    label: int | None = None


@dataclass(frozen=True)
class FI2010Dataset:
    levels: int
    samples: list[FI2010Sample]

    @property
    def rows(self) -> int:
        return len(self.samples)


def load_fi2010_matrix(
    path: Path | str,
    *,
    levels: int = 10,
    label_index: int | None = -1,
    has_header: bool = False,
    max_rows: int | None = None,
) -> FI2010Dataset:
    """Load FI-2010-style 10-level tensors from a CSV matrix.

    The common FI-2010 representation stores each level as ask price, ask size,
    bid price, bid size. This loader accepts that matrix shape and keeps labels
    optional so it can be used with raw feature files or benchmark splits.
    """
    if levels <= 0:
        raise ValueError("levels must be positive")
    feature_count = levels * 4
    samples: list[FI2010Sample] = []

    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        if has_header:
            next(reader, None)
        for row in reader:
            if max_rows is not None and len(samples) >= max_rows:
                break
            if not row:
                continue
            values = [float(value) for value in row]
            if len(values) < feature_count:
                raise ValueError(f"expected at least {feature_count} FI-2010 feature columns")
            label = None
            if label_index is not None:
                normalized_index = label_index if label_index >= 0 else len(values) + label_index
                if normalized_index >= feature_count and normalized_index < len(values):
                    label = int(values[normalized_index])
            samples.append(_sample_from_values(values[:feature_count], levels=levels, label=label))

    return FI2010Dataset(levels=levels, samples=samples)


def load_fi2010_named_columns(
    path: Path | str,
    *,
    levels: int = 10,
    label_column: str | None = None,
    max_rows: int | None = None,
) -> FI2010Dataset:
    if levels <= 0:
        raise ValueError("levels must be positive")
    samples: list[FI2010Sample] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_rows is not None and len(samples) >= max_rows:
                break
            ask_prices = [_float(row, f"ask_price_{level}") for level in range(1, levels + 1)]
            ask_sizes = [_float(row, f"ask_size_{level}") for level in range(1, levels + 1)]
            bid_prices = [_float(row, f"bid_price_{level}") for level in range(1, levels + 1)]
            bid_sizes = [_float(row, f"bid_size_{level}") for level in range(1, levels + 1)]
            label = int(float(row[label_column])) if label_column else None
            samples.append(FI2010Sample(ask_prices, ask_sizes, bid_prices, bid_sizes, label))
    return FI2010Dataset(levels=levels, samples=samples)


def fi2010_to_topn_tensor(dataset: FI2010Dataset) -> list[list[list[float]]]:
    tensor: list[list[list[float]]] = []
    for sample in dataset.samples:
        levels: list[list[float]] = []
        for index in range(dataset.levels):
            levels.append(
                [
                    sample.ask_prices[index],
                    sample.ask_sizes[index],
                    sample.bid_prices[index],
                    sample.bid_sizes[index],
                ]
            )
        tensor.append(levels)
    return tensor


def _sample_from_values(values: list[float], *, levels: int, label: int | None) -> FI2010Sample:
    ask_prices: list[float] = []
    ask_sizes: list[float] = []
    bid_prices: list[float] = []
    bid_sizes: list[float] = []
    for level in range(levels):
        offset = level * 4
        ask_prices.append(values[offset])
        ask_sizes.append(values[offset + 1])
        bid_prices.append(values[offset + 2])
        bid_sizes.append(values[offset + 3])
    return FI2010Sample(
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        label=label,
    )


def _float(row: dict[str, str], column: str) -> float:
    if column not in row:
        raise ValueError(f"missing FI-2010 column {column!r}")
    return float(row[column])
