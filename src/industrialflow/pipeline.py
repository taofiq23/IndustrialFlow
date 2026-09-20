"""Orchestrates one pipeline run: validate -> quarantine bad rows ->
engineer features -> score -> check drift against a reference batch ->
record a full audit entry. Bad data never crashes the run or silently
passes through - it's quarantined with a reason, and the run continues
on whatever data is actually valid.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from . import audit
from .config import DRIFT_P_VALUE_THRESHOLD
from .model import score_batch
from .monitoring import ColumnDriftResult, any_drifted, decide_alarm, detect_drift
from .schema import validate_batch
from .transform import engineer_features


@dataclass
class RunResult:
    run_id: str
    n_rows_in: int
    n_rows_valid: int
    n_rows_quarantined: int
    quarantined_df: pd.DataFrame
    scored_df: pd.DataFrame
    drift_results: List[ColumnDriftResult]
    drift_detected: bool


def run_pipeline(
    raw_batch: pd.DataFrame,
    model,
    reference_df: Optional[pd.DataFrame] = None,
    drift_rule: str = "uncorrected",
    drift_alpha: Optional[float] = None,
) -> RunResult:
    """drift_rule picks how per-column KS p-values become one batch-level alarm (see
    monitoring.ALARM_RULES). The default, "uncorrected", alarms if any column has p < 0.01
    and therefore false-alarms on about 3% of clean batches with three columns; drift_alpha
    (default: the configured threshold) is the batch-level level for the other rules.
    """
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    valid_df, quarantined_df = validate_batch(raw_batch)
    featured_df = engineer_features(valid_df) if len(valid_df) else valid_df
    if len(featured_df):
        featured_df = featured_df.assign(failure_risk_score=score_batch(model, featured_df))

    drift_results = detect_drift(reference_df, featured_df) if reference_df is not None and len(featured_df) else []
    if drift_rule == "uncorrected" and drift_alpha is None:
        drift_detected = any_drifted(drift_results)
    else:
        drift_detected = bool(drift_results) and decide_alarm(
            drift_results, drift_rule, DRIFT_P_VALUE_THRESHOLD if drift_alpha is None else drift_alpha
        )

    quarantine_reasons = (
        quarantined_df["validation_error"].value_counts().to_dict() if len(quarantined_df) else {}
    )

    audit.record_run(
        run_id,
        {
            "started_at": started_at,
            "n_rows_in": len(raw_batch),
            "n_rows_valid": len(valid_df),
            "n_rows_quarantined": len(quarantined_df),
            "quarantine_reasons": quarantine_reasons,
            "drift_detected": drift_detected,
            "drift_details": [{**r.to_dict(), "alarm_rule": drift_rule} for r in drift_results],
            "mean_risk_score": float(featured_df["failure_risk_score"].mean()) if len(featured_df) else None,
            "n_high_risk": int((featured_df["failure_risk_score"] > 0.5).sum()) if len(featured_df) else 0,
        },
    )

    return RunResult(
        run_id=run_id,
        n_rows_in=len(raw_batch),
        n_rows_valid=len(valid_df),
        n_rows_quarantined=len(quarantined_df),
        quarantined_df=quarantined_df,
        scored_df=featured_df,
        drift_results=drift_results,
        drift_detected=drift_detected,
    )
