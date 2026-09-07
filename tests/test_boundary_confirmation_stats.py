from datetime import date, timedelta

import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_confirmation_stats import cluster_interval, confirmation_summary  # noqa: E402


def fake_records():
    dates = [(date(2023, 5, 24) + timedelta(days=i)).isoformat() for i in range(20)]
    records = []
    for day in dates:
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            records.append(
                {
                    "symbol": symbol,
                    "assessment_date": day,
                    **{
                        model: {
                            "balanced_accuracy": score,
                            "natural_accuracy": score,
                            "log_loss": 1 - score,
                            "macro_f1": score,
                        }
                        for model, score in [
                            ("original_reference", 0.5),
                            ("matched_data_control", 0.53),
                            ("candidate", 0.6),
                        ]
                    },
                }
            )
    return dates, records


def test_date_cluster_preserves_constant_paired_effect_and_confirmation_gates():
    interval = cluster_interval(np.full(20, 0.1), block_length=3)
    assert np.isclose(interval["lower_95"], 0.1)
    assert np.isclose(interval["upper_95"], 0.1)
    days, records = fake_records()
    result = confirmation_summary(records, days)
    assert result["substantial_predictive_gain_confirmed"]
    assert not result["economic_performance_confirmed"]
    assert np.isclose(result["uncertainty"]["original_reference"]["date_cluster"]["mean"], 0.1)


def test_confirmation_rejects_incomplete_pairs_and_one_asset_deterioration():
    days, records = fake_records()
    with pytest.raises(ValueError, match="Complete paired"):
        confirmation_summary(records[:-1], days)
    for record in records:
        record["candidate"]["balanced_accuracy"] = 0.49 if record["symbol"] == "BTCUSDT" else 0.75
    result = confirmation_summary(records, days)
    assert result["gates"]["mean_balanced_accuracy_gain_at_least_five_points"]
    assert not result["gates"]["both_assets_improve"]
    assert not result["substantial_predictive_gain_confirmed"]
