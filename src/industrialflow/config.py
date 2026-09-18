"""Shared constants."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
AUDIT_DB_PATH = DATA_DIR / "audit.db"
QUARANTINE_DIR = DATA_DIR / "quarantine"

EQUIPMENT_IDS = [f"unit-{i:02d}" for i in range(1, 9)]

# Valid sensor ranges, used both by the synthetic generator and the
# validation schema - real ranges for the kind of industrial equipment
# this simulates (a mid-size motor/pump assembly).
TEMPERATURE_RANGE_C = (10.0, 120.0)
VIBRATION_RANGE_MM_S = (0.0, 15.0)
PRESSURE_RANGE_KPA = (80.0, 250.0)

# Drift monitoring: a KS-test p-value below this on a column means its
# distribution has shifted enough from the reference batch to flag.
DRIFT_P_VALUE_THRESHOLD = 0.01
