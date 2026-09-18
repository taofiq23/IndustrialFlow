"""Statistical drift detection: compares each new batch's feature
distributions against a reference baseline using a two-sample
Kolmogorov-Smirnov test - a real, standard technique for detecting
distribution shift in production ML monitoring, not an ad hoc threshold
on the mean.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd
from scipy.stats import ks_2samp

from .config import DRIFT_P_VALUE_THRESHOLD

MONITORED_COLUMNS = ["temperature_c", "vibration_mm_s", "pressure_kpa"]


@dataclass
class ColumnDriftResult:
    column: str
    ks_statistic: float
    p_value: float
    drifted: bool

    def to_dict(self) -> dict:
        return asdict(self)


def detect_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: Optional[List[str]] = None
) -> List[ColumnDriftResult]:
    columns = columns or MONITORED_COLUMNS
    results = []
    for column in columns:
        reference_values = reference_df[column].dropna()
        current_values = current_df[column].dropna()
        statistic, p_value = ks_2samp(reference_values, current_values)
        statistic, p_value = float(statistic), float(p_value)
        results.append(
            ColumnDriftResult(
                column=column,
                ks_statistic=round(statistic, 4),
                p_value=round(p_value, 6),
                # bool(...) matters here: comparing numpy floats yields
                # numpy.bool_, which fails both `is True/False` identity
                # checks and json.dumps() in the audit log.
                drifted=bool(p_value < DRIFT_P_VALUE_THRESHOLD),
            )
        )
    return results


def any_drifted(results: List[ColumnDriftResult]) -> bool:
    return any(r.drifted for r in results)
