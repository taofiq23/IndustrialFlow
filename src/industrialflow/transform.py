"""Feature engineering: derives rolling per-equipment statistics from
validated sensor readings - the inputs to the risk-scoring model.
"""

import pandas as pd

ROLLING_WINDOW = 5

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
    "temperature_rolling_mean",
    "vibration_rolling_mean",
    "temperature_rate_of_change",
    "vibration_rate_of_change",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["equipment_id", "timestamp"]).copy()
    grouped = df.groupby("equipment_id", group_keys=False)

    df["temperature_rolling_mean"] = grouped["temperature_c"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["vibration_rolling_mean"] = grouped["vibration_mm_s"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["temperature_rate_of_change"] = grouped["temperature_c"].diff().fillna(0.0)
    df["vibration_rate_of_change"] = grouped["vibration_mm_s"].diff().fillna(0.0)

    return df
