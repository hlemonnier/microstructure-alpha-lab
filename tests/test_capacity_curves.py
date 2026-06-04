from lob_forge.capacity_curves import decayed_edge_bps, format_capacity_curve, pnl_by_notional_curve


def test_capacity_curve_decays_edge_by_notional() -> None:
    points = pnl_by_notional_curve(
        base_edge_bps=2.0,
        fee_bps=0.5,
        notionals=[100.0, 1000.0, 2000.0],
        decay_start=1000.0,
        zero_edge_notional=2000.0,
    )

    assert points[0].net_edge_bps == 1.5
    assert points[-1].gross_edge_bps == 0.0
    assert not points[-1].capacity_ok
    assert format_capacity_curve(points).startswith("notional,gross_edge")


def test_decayed_edge_before_start_is_unchanged() -> None:
    assert decayed_edge_bps(3.0, notional=500.0, decay_start=1000.0, zero_edge_notional=2000.0) == 3.0
