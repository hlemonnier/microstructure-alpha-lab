"""Exact parsing of OKX's published historical L2 price/quantity/order counts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True, slots=True)
class CountedLevel:
    price: Decimal
    quantity: Decimal
    order_count: int


@dataclass(frozen=True, slots=True)
class CountedDepthMessage:
    instrument: str
    action: str
    timestamp_ms: int
    asks: tuple[CountedLevel, ...]
    bids: tuple[CountedLevel, ...]


def _integer(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Source integer fields cannot be floating-point or boolean")
    if isinstance(value, str) and (not value.isascii() or not value.isdecimal()):
        raise ValueError("Source integer strings must contain only decimal digits")
    result = int(value)
    if result < 0 or result >= 2**53:
        raise ValueError("Source integer is outside the exact nonnegative clock/count range")
    return result


def _decimal(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Prices and quantities require original decimal strings or integers")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal source value") from exc
    if not result.is_finite():
        raise ValueError("Nonfinite decimal source value")
    return result


def parse_counted_depth(line: str | bytes, *, instrument: str) -> CountedDepthMessage:
    """Parse a published message, without inventing arrival clocks or event types.

    Quantities and counts are absolute revisions. A deletion is zero/zero; an
    individual cancellation or execution cannot be identified from it alone.
    """
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid source JSON") from exc
    if not isinstance(record, dict) or not {"instId", "action", "ts", "asks", "bids"}.issubset(record):
        raise ValueError("Missing historical depth fields")
    if record["instId"] != instrument or record["action"] not in ("snapshot", "update"):
        raise ValueError("Unexpected instrument or depth action")
    sides = {}
    for side in ("asks", "bids"):
        if not isinstance(record[side], list):
            raise ValueError("Each depth side must be an array")
        levels = []
        prices = set()
        for row in record[side]:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError("Each level needs price, quantity and order count")
            price, quantity, count = _decimal(row[0]), _decimal(row[1]), _integer(row[2])
            if price <= 0 or quantity < 0 or (quantity == 0) != (count == 0):
                raise ValueError("Price, absolute quantity and order count disagree")
            if price in prices:
                raise ValueError("Duplicate price within one published side")
            prices.add(price)
            levels.append(CountedLevel(price, quantity, count))
        if record["action"] == "snapshot":
            if not levels or any(level.quantity == 0 for level in levels):
                raise ValueError("A snapshot must contain positive resting levels on both sides")
            if [level.price for level in levels] != sorted(prices, reverse=side == "bids"):
                raise ValueError("Snapshot price ordering is invalid")
        sides[side] = tuple(levels)
    if record["action"] == "snapshot" and sides["bids"][0].price >= sides["asks"][0].price:
        raise ValueError("Crossed or locked snapshot")
    return CountedDepthMessage(instrument, record["action"], _integer(record["ts"]), sides["asks"], sides["bids"])
