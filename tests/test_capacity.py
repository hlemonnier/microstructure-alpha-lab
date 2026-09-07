from pathlib import Path

from lob_forge.capacity import format_capacity_diagnostics, run_capacity_diagnostics


def test_capacity_diagnostics_uses_volume_and_top_book_caps(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        "\n".join(
            [
                "source_date,label,microprice_deviation,bid,ask,entry_bid,entry_ask,entry_bid_qty,entry_ask_qty,trade_notional",
                "2023-05-16,1,0.5,99,100,99,100,10,20,10000",
                "2023-05-16,-1,-0.5,99,100,99,100,10,20,10000",
                "2023-05-17,0,0.0,99,100,99,100,10,20,10000",
            ]
        )
    )

    diagnostics = run_capacity_diagnostics(
        path,
        feature="microprice_deviation",
        threshold=0.1,
        participation_rate=0.01,
        top_book_fraction=0.05,
    )
    output = format_capacity_diagnostics(diagnostics)
    by_side = {item.side: item for item in diagnostics}

    assert by_side["all"].signals == 2
    assert by_side["all"].positive_capacity_signals == 2
    assert by_side["long"].median_capacity_notional == 100.0
    assert by_side["short"].median_capacity_notional == 49.5
    assert by_side["all"].total_capacity_notional == 149.5
    assert output.splitlines()[0].startswith("group,side,signals")


def test_capacity_diagnostics_can_group_by_source_date(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        "\n".join(
            [
                "source_date,label,microprice_deviation,bid,ask,bid_qty,ask_qty,trade_notional",
                "2023-05-16,1,0.5,99,100,10,20,10000",
                "2023-05-17,1,0.5,99,100,10,20,0",
            ]
        )
    )

    diagnostics = run_capacity_diagnostics(
        path,
        feature="microprice_deviation",
        threshold=0.1,
        by_source_date=True,
    )

    groups = {item.group for item in diagnostics}
    assert groups == {"2023-05-16", "2023-05-17", "all"}
    assert len(diagnostics) == 9


def test_capacity_shares_repeated_observation_liquidity(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        "event_time,entry_update_id,microprice_deviation,bid,ask,bid_qty,ask_qty,trade_notional\n"
        "100,5,1,99,100,100,100,10000\n100,5,1,99,100,100,100,10000\n"
    )
    result = run_capacity_diagnostics(
        path, feature="microprice_deviation", threshold=0.1, participation_rate=0.01, top_book_fraction=0.5
    )
    assert result[0].total_capacity_notional == 100
    assert result[0].positive_capacity_signals == 1
    assert result[0].identified_liquidity_windows == 1
