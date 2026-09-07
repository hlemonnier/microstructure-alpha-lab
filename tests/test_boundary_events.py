import csv
import io
import zipfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from lob_forge.boundary_events import event_frame  # noqa: E402
from lob_forge.boundary_forecasts import OBSERVED_FIELDS  # noqa: E402
from lob_forge.features import AGG_TRADE_COLUMNS, BOOK_TICKER_COLUMNS, build_quote_trade_dataset  # noqa: E402


def example_events():
    rows = []
    for second in range(40):
        for offset in [10, 500, 900, 950, 980]:
            timestamp = 100000 + second * 1000 + offset
            bid = 100 + ((second * 5 + offset // 10) % 7) * 0.1
            rows.append([len(rows) + 1, bid, 1 + second % 3, bid + 0.1, 2 + second % 4, timestamp, timestamp])
    quotes = pd.DataFrame(rows, columns=BOOK_TICKER_COLUMNS)
    trades = pd.DataFrame(
        [[i + 1, 100.0, 0.3, i + 1, i + 1, 100000 + i * 1000 + 960, i % 2 == 0] for i in range(40)],
        columns=AGG_TRADE_COLUMNS,
    )
    return quotes, trades


def test_event_builder_matches_canonical_quote_features_and_labels(tmp_path):
    quotes, trades = example_events()
    for name, frame in [("quotes", quotes), ("trades", trades)]:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(frame.columns)
        for row in frame.itertuples(index=False, name=None):
            writer.writerow([str(value).lower() if isinstance(value, bool) else value for value in row])
        with zipfile.ZipFile(tmp_path / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}.csv", buffer.getvalue())
    build_quote_trade_dataset(
        book_ticker_zip=tmp_path / "quotes.zip",
        agg_trades_zip=tmp_path / "trades.zip",
        output_csv=tmp_path / "canonical.csv",
        horizon_ms=5000,
        execution_latency_ms=100,
        min_tick=0.1,
    )
    canonical = pd.read_csv(tmp_path / "canonical.csv").set_index("decision_time")
    actual = event_frame(quotes, trades, min_tick=0.1).set_index("decision_time").loc[canonical.index]
    observed = [column for column in OBSERVED_FIELDS if column != "decision_time"]
    np.testing.assert_allclose(actual[observed], canonical[observed], rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(actual["label"], canonical["label"])
    np.testing.assert_array_equal(actual["entry_event_time"], canonical["entry_event_time"])
    np.testing.assert_array_equal(actual["future_event_time"], canonical["future_event_time"])


def test_event_features_use_only_closed_bucket_and_known_history():
    quotes, trades = example_events()
    full = event_frame(quotes, trades, min_tick=0.1)
    prefix = event_frame(quotes.iloc[:100], trades.iloc[:20], min_tick=0.1)
    future_columns = {"label", "entry_event_time", "future_event_time", "entry_mid", "future_mid"}
    columns = [column for column in full if column not in future_columns]
    pd.testing.assert_frame_equal(full.iloc[: len(prefix)][columns], prefix[columns])
    assert (full["event_count_50ms"] == 2).all()
    assert (full["event_count_200ms"] == 3).all()
    broken = quotes.copy()
    broken.loc[10, "update_id"] = 1
    with pytest.raises(ValueError, match="increasing IDs"):
        event_frame(broken, trades, min_tick=0.1)
