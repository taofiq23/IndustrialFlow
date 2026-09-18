"""Generates synthetic industrial sensor batches - clearly synthetic, but
structured like real equipment telemetry (temperature/vibration/pressure
per unit per timestamp), including a small rate of bad rows (nulls,
out-of-range values) so the validation stage has real work to do, and an
optional systematic drift shift to test whether monitoring catches it.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .config import EQUIPMENT_IDS, PRESSURE_RANGE_KPA, TEMPERATURE_RANGE_C, VIBRATION_RANGE_MM_S


def generate_batch(
    n_rows: int = 200,
    seed: int = 0,
    bad_row_rate: float = 0.05,
    temperature_drift_c: float = 0.0,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """temperature_drift_c shifts the whole batch's temperature readings by
    a fixed amount - simulating something like a sensor recalibration or a
    change in ambient conditions, for testing drift detection.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    start_time = start_time or datetime.now(timezone.utc)

    rows = []
    for i in range(n_rows):
        equipment_id = rng.choice(EQUIPMENT_IDS)
        timestamp = start_time + timedelta(seconds=i * 5)

        temperature = np_rng.normal(60.0, 12.0) + temperature_drift_c
        vibration = max(0.0, np_rng.normal(4.0, 1.5))
        pressure = np_rng.normal(150.0, 20.0)

        rows.append(
            {
                "timestamp": timestamp,
                "equipment_id": equipment_id,
                "temperature_c": temperature,
                "vibration_mm_s": vibration,
                "pressure_kpa": pressure,
            }
        )

    df = pd.DataFrame(rows)

    # Inject a realistic slice of bad rows: nulls and out-of-range values,
    # exactly the kind of thing a validation stage needs to catch rather
    # than silently pass downstream.
    n_bad = int(n_rows * bad_row_rate)
    bad_indices = rng.sample(range(n_rows), k=min(n_bad, n_rows))
    for idx in bad_indices:
        fault_type = rng.choice(["null_temperature", "out_of_range_pressure", "negative_vibration"])
        if fault_type == "null_temperature":
            df.loc[idx, "temperature_c"] = np.nan
        elif fault_type == "out_of_range_pressure":
            df.loc[idx, "pressure_kpa"] = PRESSURE_RANGE_KPA[1] + rng.uniform(50, 200)
        elif fault_type == "negative_vibration":
            df.loc[idx, "vibration_mm_s"] = -abs(np_rng.normal(2.0, 1.0))

    return df
