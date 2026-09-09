import copy
import json
from decimal import Decimal

import pytest

from lob_forge.boundary_okx_source import parse_counted_depth


def _record():
    return {"instId": "BTC-USDT-SWAP", "action": "snapshot", "ts": "1685577600002",
        "asks": [["27206.600000000000000001", "411.0", "16"], ["27207.7", "29", "2"]],
        "bids": [["27206.5", "189.0", "11"], ["27205.9", "40", "3"]]}


def _parse(record):
    return parse_counted_depth(json.dumps(record), instrument="BTC-USDT-SWAP")


def test_original_decimals_and_order_counts_are_preserved_exactly():
    result = _parse(_record())
    assert result.asks[0].price == Decimal("27206.600000000000000001")
    assert result.asks[0].quantity == Decimal("411.0")
    assert result.asks[0].order_count == 16
    assert result.timestamp_ms == 1685577600002
    assert not hasattr(result, "arrival_timestamp")


def test_absolute_updates_allow_deletions_and_empty_sides_without_event_inference():
    record = _record()
    record.update(action="update", asks=[["27206.6", "0", "0"]], bids=[])
    result = _parse(record)
    assert result.asks[0].quantity == 0 and result.asks[0].order_count == 0
    assert result.bids == () and result.action == "update"


def test_invalid_levels_and_inexact_numeric_fields_are_rejected():
    for row in (["10", "1", "0"], ["10", "0", "1"], ["10", "-1", "1"],
                ["NaN", "1", "1"], ["Infinity", "1", "1"], ["-1", "1", "1"],
                [10.0, "1", "1"], ["10", "1", 1.0], ["10", "1", True],
                ["10", "1", "1.0"], ["10", "1"], ["10", "1", "1", "2"]):
        record = _record()
        record.update(action="update", asks=[row])
        with pytest.raises(ValueError):
            _parse(record)
    for clock in (1685577600002.0, True, "1.5", "-1", "١٢", 2**53):
        record = _record()
        record["ts"] = clock
        with pytest.raises(ValueError):
            _parse(record)


def test_malformed_instrument_snapshot_ordering_and_duplicate_prices_fail():
    variants = []
    for field, value in (("instId", "ETH-USDT-SWAP"), ("action", "trade"),
                         ("asks", []), ("bids", [["30000", "1", "1"]]),
                         ("asks", list(reversed(_record()["asks"]))),
                         ("asks", [["27207", "1", "1"], ["27207.0", "2", "2"]])):
        record = _record()
        record[field] = value
        variants.append(record)
    record = copy.deepcopy(_record())
    del record["ts"]
    variants.append(record)
    for record in variants:
        with pytest.raises(ValueError):
            _parse(record)
    for malformed in ("{", "null", "[]"):
        with pytest.raises(ValueError):
            parse_counted_depth(malformed, instrument="BTC-USDT-SWAP")
