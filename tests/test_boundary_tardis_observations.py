import json

import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_tardis_observations import sample_tardis_depth  # noqa: E402


def test_capture_sampling_excludes_equal_time_events_and_late_snapshots(tmp_path):
    def depth(first, final, previous, amount, event):
        return {"stream": "btcusdt@depth@0ms", "data": {"e": "depthUpdate", "s": "BTCUSDT", "U": first, "u": final,
                "pu": previous, "E": event, "T": event - 1, "b": [["100", str(amount)]], "a": []}}

    def quote(uid, amount, event):
        return {"stream": "btcusdt@bookTicker", "data": {"s": "BTCUSDT", "u": uid, "E": event, "T": event - 1,
                "a": "102", "A": "10", "b": "100", "B": str(amount)}}

    snap = {"stream": "btcusdt@depthSnapshot", "generated": True, "data": {"lastUpdateId": 10, "E": 1000, "T": 999,
        "bids": [[str(100 - i), "10"] for i in range(25)], "asks": [[str(102 + i), "10"] for i in range(25)]}}
    messages = [(1200, depth(9, 12, 8, 12, 1100)), (1400, quote(12, 12, 1100)), (1500, snap),
                (1900, depth(13, 14, 12, 20, 1850)), (1901, quote(14, 20, 1850))]
    path = tmp_path / "native.ndjson"
    path.write_text("".join(f"1970-01-01T00:00:{ms // 1000:02d}.{ms % 1000:03d}000000Z {json.dumps(m)}\n" for ms, m in messages))
    samples, quotes, checks = sample_tardis_depth([path], {"BTCUSDT": np.array([2000, 2100], dtype=np.int64)})
    result = samples["BTCUSDT"]
    assert result["available"].tolist() == [[True, False], [True, True]]
    assert result["update_ids"].tolist() == [[12, -1], [14, 12]]
    assert result["depth"][0, 0, 0, 3] == 12  # Exact 1900ms update is excluded from 2000ms minus 100ms.
    assert result["depth"][1, 0, 0, 3] == 20
    assert result["capture_times_ns"][0, 0] == 1500_000_000
    assert checks["assets"]["BTCUSDT"]["sampled_depth_quote_comparisons"] == 3
    np.testing.assert_array_equal(quotes["BTCUSDT"]["ids"], [12, 14])
    path.write_text(path.read_text().replace("1970-01-01T00:00:01.900000000Z", "\n1970-01-01T00:00:01.900000000Z"))
    gap, _, gap_checks = sample_tardis_depth([path], {"BTCUSDT": np.array([1950], dtype=np.int64)})
    # The untimestamped disconnect cannot invalidate the 1850ms decision
    # cutoff using information first observed at the next 1900ms capture.
    assert gap["BTCUSDT"]["available"][0, 0]
    assert gap_checks["disconnects"] == 1
