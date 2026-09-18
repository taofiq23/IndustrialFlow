import numpy as np
import pandas as pd

from src.industrialflow.monitoring import detect_drift


def _df(values, column="temperature_c"):
    return pd.DataFrame({column: values})


def test_same_distribution_is_not_flagged_as_drifted():
    rng = np.random.default_rng(0)
    reference = _df(rng.normal(60, 10, size=500))
    current = _df(rng.normal(60, 10, size=500))

    results = detect_drift(reference, current, columns=["temperature_c"])

    assert results[0].drifted is False


def test_shifted_distribution_is_flagged_as_drifted():
    rng = np.random.default_rng(0)
    reference = _df(rng.normal(60, 10, size=500))
    current = _df(rng.normal(90, 10, size=500))  # +30 shift

    results = detect_drift(reference, current, columns=["temperature_c"])

    assert results[0].drifted is True
    assert results[0].p_value < 0.01


def test_detect_drift_checks_each_requested_column_independently():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {
            "temperature_c": rng.normal(60, 10, size=300),
            "vibration_mm_s": rng.normal(4, 1, size=300),
        }
    )
    current = pd.DataFrame(
        {
            "temperature_c": rng.normal(60, 10, size=300),  # unchanged
            "vibration_mm_s": rng.normal(10, 1, size=300),  # shifted
        }
    )

    results = detect_drift(reference, current, columns=["temperature_c", "vibration_mm_s"])
    by_column = {r.column: r for r in results}

    assert by_column["temperature_c"].drifted is False
    assert by_column["vibration_mm_s"].drifted is True
