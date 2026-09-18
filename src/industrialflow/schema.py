"""Data validation with Pandera. Bad rows are never silently dropped or
allowed to crash the pipeline - they're quarantined with the specific
reason they failed, so someone can actually go look at what went wrong.
"""

from typing import Tuple

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from .config import PRESSURE_RANGE_KPA, TEMPERATURE_RANGE_C, VIBRATION_RANGE_MM_S

sensor_schema = DataFrameSchema(
    {
        "timestamp": Column(pa.DateTime, nullable=False),
        "equipment_id": Column(str, nullable=False),
        "temperature_c": Column(float, Check.in_range(*TEMPERATURE_RANGE_C), nullable=False),
        "vibration_mm_s": Column(float, Check.in_range(*VIBRATION_RANGE_MM_S), nullable=False),
        "pressure_kpa": Column(float, Check.in_range(*PRESSURE_RANGE_KPA), nullable=False),
    },
    strict=False,
    coerce=False,
)


def validate_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (valid_df, quarantined_df). quarantined_df has an extra
    'validation_error' column describing why each row was rejected.
    """
    try:
        sensor_schema.validate(df, lazy=True)
        empty = df.iloc[0:0].copy()
        empty["validation_error"] = pd.Series(dtype=str)
        return df.copy(), empty
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        bad_indices = set(failure_cases["index"].dropna().astype(int))

        reasons = (
            failure_cases[failure_cases["index"].notna()]
            .astype({"index": int})
            .groupby("index")
            .apply(lambda g: "; ".join(f"{c}: {chk}" for c, chk in zip(g["column"], g["check"])))
        )

        quarantined = df.loc[df.index.isin(bad_indices)].copy()
        quarantined["validation_error"] = quarantined.index.map(reasons)

        valid = df.loc[~df.index.isin(bad_indices)].copy()
        return valid, quarantined
