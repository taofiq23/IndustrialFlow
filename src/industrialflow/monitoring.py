"""Statistical drift detection: compares each new batch's feature
distributions against a reference baseline using a two-sample
Kolmogorov-Smirnov test - a real, standard technique for detecting
distribution shift in production ML monitoring, not an ad hoc threshold
on the mean.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp

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


ALARM_RULES = ("uncorrected", "bonferroni", "bh", "fisher", "k_of_n")


def alarm_from_pvalues(p_values, rule: str = "uncorrected", alpha: float = DRIFT_P_VALUE_THRESHOLD, min_columns: int = 2) -> np.ndarray:
    """Batch-level alarm decision from per-column p-values (last axis = columns).

    "uncorrected" is what any_drifted() does: alarm if any column has p < alpha,
    so the batch-level false-alarm rate is 1-(1-alpha)^k, not alpha.
    "bonferroni" tests each column at alpha/k. "bh" alarms if any Benjamini-Hochberg
    rejection occurs at level alpha (Simes' test of the global null). "fisher" combines
    the p-values into one test at level alpha. "k_of_n" alarms only if at least
    min_columns columns each have p < alpha (alpha is then a per-column threshold).
    """
    p = np.asarray(p_values, dtype=float)
    k = p.shape[-1]
    if rule == "uncorrected":
        return (p < alpha).any(axis=-1)
    if rule == "bonferroni":
        return (p < alpha / k).any(axis=-1)
    if rule == "bh":
        sorted_p = np.sort(p, axis=-1)
        thresholds = alpha * np.arange(1, k + 1) / k
        return (sorted_p <= thresholds).any(axis=-1)
    if rule == "fisher":
        statistic = -2.0 * np.log(np.clip(p, 1e-300, 1.0)).sum(axis=-1)
        return chi2.sf(statistic, df=2 * k) < alpha
    if rule == "k_of_n":
        return (p < alpha).sum(axis=-1) >= min_columns
    raise ValueError(f"unknown alarm rule {rule!r}; expected one of {ALARM_RULES}")


def decide_alarm(results: List[ColumnDriftResult], rule: str = "uncorrected", alpha: float = DRIFT_P_VALUE_THRESHOLD, min_columns: int = 2) -> bool:
    return bool(alarm_from_pvalues([r.p_value for r in results], rule, alpha, min_columns))
