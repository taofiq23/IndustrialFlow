import pandas as pd

from src.industrialflow.transform import FEATURE_COLUMNS, engineer_features


def test_engineer_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="s"),
            "equipment_id": ["unit-01"] * 5,
            "temperature_c": [50.0, 52.0, 54.0, 56.0, 58.0],
            "vibration_mm_s": [3.0, 3.2, 3.4, 3.6, 3.8],
            "pressure_kpa": [150.0] * 5,
        }
    )

    result = engineer_features(df)

    for column in FEATURE_COLUMNS:
        assert column in result.columns


def test_rate_of_change_matches_manual_diff():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="s"),
            "equipment_id": ["unit-01"] * 3,
            "temperature_c": [50.0, 55.0, 53.0],
            "vibration_mm_s": [3.0, 3.0, 3.0],
            "pressure_kpa": [150.0] * 3,
        }
    )

    result = engineer_features(df)

    assert result["temperature_rate_of_change"].tolist() == [0.0, 5.0, -2.0]


def test_features_computed_independently_per_equipment():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="s"),
            "equipment_id": ["unit-01", "unit-02", "unit-01", "unit-02"],
            "temperature_c": [50.0, 100.0, 60.0, 100.0],
            "vibration_mm_s": [3.0, 3.0, 3.0, 3.0],
            "pressure_kpa": [150.0] * 4,
        }
    )

    result = engineer_features(df)

    # unit-02's constant 100.0 readings should show zero rate of change,
    # regardless of unit-01's readings interleaved in the same batch.
    unit_2 = result[result["equipment_id"] == "unit-02"]
    assert (unit_2["temperature_rate_of_change"] == 0.0).all()
