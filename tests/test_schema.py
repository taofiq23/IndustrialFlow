import numpy as np
import pandas as pd

from src.industrialflow.schema import validate_batch


def _base_df(n=5):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="s"),
            "equipment_id": ["unit-01"] * n,
            "temperature_c": [60.0] * n,
            "vibration_mm_s": [4.0] * n,
            "pressure_kpa": [150.0] * n,
        }
    )


def test_all_valid_rows_pass_through_untouched():
    df = _base_df()
    valid, quarantined = validate_batch(df)

    assert len(valid) == len(df)
    assert len(quarantined) == 0


def test_null_temperature_is_quarantined():
    df = _base_df()
    df.loc[2, "temperature_c"] = np.nan

    valid, quarantined = validate_batch(df)

    assert len(valid) == 4
    assert len(quarantined) == 1
    assert "temperature_c" in quarantined.iloc[0]["validation_error"]


def test_out_of_range_pressure_is_quarantined():
    df = _base_df()
    df.loc[1, "pressure_kpa"] = 999.0

    valid, quarantined = validate_batch(df)

    assert len(valid) == 4
    assert len(quarantined) == 1


def test_negative_vibration_is_quarantined():
    df = _base_df()
    df.loc[0, "vibration_mm_s"] = -2.0

    valid, quarantined = validate_batch(df)

    assert len(valid) == 4
    assert len(quarantined) == 1


def test_multiple_bad_rows_all_quarantined_independently():
    df = _base_df(n=6)
    df.loc[0, "temperature_c"] = np.nan
    df.loc[3, "pressure_kpa"] = 999.0

    valid, quarantined = validate_batch(df)

    assert len(valid) == 4
    assert len(quarantined) == 2
