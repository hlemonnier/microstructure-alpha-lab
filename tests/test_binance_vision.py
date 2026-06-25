from lob_forge.binance_vision import archive_key, dataset_prefix, infer_date_from_key, iter_dates


def test_builds_futures_book_depth_key() -> None:
    assert (
        archive_key(
            market="futures/um",
            frequency="daily",
            dataset="bookDepth",
            symbol="btcusdt",
            date_value="2023-01-01",
        )
        == "data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2023-01-01.zip"
    )


def test_builds_kline_key_with_interval() -> None:
    assert (
        archive_key(
            market="spot",
            frequency="daily",
            dataset="klines",
            symbol="ethusdt",
            interval="1m",
            date_value="2024-01-02",
        )
        == "data/spot/daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-01-02.zip"
    )


def test_dataset_prefix() -> None:
    assert (
        dataset_prefix(
            market="futures/um",
            frequency="daily",
            dataset="trades",
            symbol="BTCUSDT",
        )
        == "data/futures/um/daily/trades/BTCUSDT/"
    )


def test_date_helpers() -> None:
    assert list(iter_dates("2024-01-01", "2024-01-03")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert infer_date_from_key("data/x/BTCUSDT-trades-2024-01-03.zip") == "2024-01-03"
