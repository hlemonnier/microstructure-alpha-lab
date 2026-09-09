import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_quote_currency_inputs import (  # noqa: E402
    CONVERSION_COLUMNS, NATIVE_COLUMNS, QUOTE_VARIANTS, append_quote_sources, quote_variant_columns,
)


def _frames():
    clocks = np.array([1000, 2000, 3000], dtype=np.int64)
    original = pd.DataFrame({"original_signal": [1., 2., 3.]})
    sources = {}
    for i, (symbol, delay) in enumerate((s, d) for s in ("BTCUSDT", "ETHUSDT") for d in (100, 500)):
        sources[symbol, delay] = pd.DataFrame({c: np.arange(3, dtype=float) + 100*i
            for c in (*CONVERSION_COLUMNS, *NATIVE_COLUMNS)}, index=clocks)
    return original, clocks, sources


def test_conversion_controls_retain_both_rates_without_native_market_information():
    original, clocks, sources = _frames()
    wide = append_quote_sources(original, clocks, sources)
    pd.testing.assert_frame_equal(wide[list(original.columns)], original, check_exact=True)
    for variant in ("fx100", "fx500"):
        columns = quote_variant_columns(original.columns, variant)
        assert len(columns) == 7 and not any("__spot_" in c or "__converted_basis_" in c for c in columns)
        assert any("BTCUSDT" in c for c in columns) and any("ETHUSDT" in c for c in columns)
        assert len(wide[columns]) == 3


def test_native_source_and_delay_ablations_keep_exact_shared_values():
    original, clocks, sources = _frames()
    wide = append_quote_sources(original, clocks, sources)
    for variant in QUOTE_VARIANTS:
        chosen = wide[quote_variant_columns(original.columns, variant)]
        assert len(chosen) == 3
        if variant.startswith("btc"):
            assert not any(c.startswith("quote_ETHUSDT") and "__spot_" in c for c in chosen)
    np.testing.assert_array_equal(wide.quote_BTCUSDT_100__spot_last_side, [0., 1., 2.])
    np.testing.assert_array_equal(wide.quote_BTCUSDT_500__spot_last_side, [100., 101., 102.])
    np.testing.assert_array_equal(wide.quote_ETHUSDT_100__spot_last_side, [200., 201., 202.])
    for delay in (100, 500):
        fx = quote_variant_columns(original.columns, f"fx{delay}")
        btc = quote_variant_columns(original.columns, f"btc{delay}")
        both = quote_variant_columns(original.columns, f"both{delay}")
        assert set(fx) < set(btc) < set(both)
        pd.testing.assert_frame_equal(wide[both][fx], wide[fx], check_exact=True)


def test_missing_peer_clock_and_unexpected_source_fields_do_not_silently_change_rows():
    original, clocks, sources = _frames()
    sources["ETHUSDT", 500] = sources["ETHUSDT", 500].drop(index=2000)
    with pytest.raises(ValueError, match="cover every original row"):
        append_quote_sources(original, clocks, sources)
    original, clocks, sources = _frames()
    sources["BTCUSDT", 100]["label"] = 1
    with pytest.raises(ValueError, match="unique schema"):
        append_quote_sources(original, clocks, sources)
