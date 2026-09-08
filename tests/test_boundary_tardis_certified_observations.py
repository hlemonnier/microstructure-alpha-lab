import json

import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_tardis_certified import CertifiedBinanceFuturesDepthState  # noqa: E402
from lob_forge.boundary_tardis_observations import sample_tardis_depth  # noqa: E402


def test_certificate_capture_is_strict_and_does_not_refresh_stale_depth(tmp_path):
    snap = {"stream": "btcusdt@depthSnapshot", "generated": True, "data": {"lastUpdateId": 10, "E": 1000,
        "bids": [[str(100 - i), "10"] for i in range(25)], "asks": [[str(102 + i), "10"] for i in range(25)]}}

    def depth(first, final, previous, event, bids):
        return {"stream": "btcusdt@depth@0ms", "data": {"e": "depthUpdate", "s": "BTCUSDT", "U": first,
            "u": final, "pu": previous, "E": event, "T": event - 1, "b": bids, "a": []}}

    def quote(uid, bid, event):
        return {"stream": "btcusdt@bookTicker", "data": {"s": "BTCUSDT", "u": uid, "E": event, "T": event - 1,
            "b": str(bid), "B": "10", "a": "102", "A": "10"}}

    messages = [(1400, quote(10, 100, 1100)), (1500, snap), (1600, depth(9, 10, 8, 1500, [])),
        (1800, depth(11, 20, 10, 1750, [[str(100 - i), "0"] for i in range(25)] + [[str(40 - i), "10"] for i in range(25)])),
        (2000, quote(20, 40, 1990))]
    path = tmp_path / "capture.ndjson"

    def write(items):
        path.write_text("".join(f"1970-01-01T00:00:{ms // 1000:02d}.{ms % 1000:03d}000000Z {json.dumps(m)}\n" for ms, m in items))

    def factory(symbol):
        return CertifiedBinanceFuturesDepthState(symbol, min_tick=1)

    clock = {"BTCUSDT": np.array([2100, 2101, 3100])}
    write(messages)
    first, _, _ = sample_tardis_depth([path], clock, delays_ms=(100,), state_factory=factory)
    result = first["BTCUSDT"]
    assert result["available"].ravel().tolist() == [False, True, False]
    assert result["capture_times_ns"].ravel().tolist() == [1800_000_000, 2000_000_000, 2000_000_000]
    assert result["publisher_times_ms"].ravel().tolist() == [1750, 1990, 1990]
    assert result["depth"][1, 0, 0].tolist() == [102, 10, 40, 10]
    # A future source alteration cannot affect any earlier observation or mask.
    changed = messages + [(4000, depth(21, 30, 20, 3990, [["40", "999"]])), (4001, quote(30, 40, 3990))]
    changed[-1][1]["data"]["B"] = "999"
    write(changed)
    later, _, _ = sample_tardis_depth([path], clock, delays_ms=(100,), state_factory=factory)
    for key in result:
        np.testing.assert_array_equal(result[key], later["BTCUSDT"][key])
