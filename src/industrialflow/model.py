"""A lightweight failure-risk model, trained on synthetic labeled data
where risk is a known rule-plus-noise function of temperature and
vibration - so real accuracy can be measured against ground truth
rather than assumed. The model is intentionally simple; this project is
about pipeline rigor, not modeling sophistication.
"""

import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import PRESSURE_RANGE_KPA, TEMPERATURE_RANGE_C, VIBRATION_RANGE_MM_S
from .transform import FEATURE_COLUMNS


def _true_risk_label(temperature_c: float, vibration_mm_s: float, rng: random.Random) -> int:
    base = 1 if (temperature_c > 85 and vibration_mm_s > 6) else 0
    if rng.random() < 0.05:  # label noise, so the task isn't trivially separable
        base = 1 - base
    return base


def generate_training_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    temperature = np_rng.uniform(*TEMPERATURE_RANGE_C, size=n)
    vibration = np_rng.uniform(*VIBRATION_RANGE_MM_S, size=n)
    pressure = np_rng.uniform(*PRESSURE_RANGE_KPA, size=n)

    df = pd.DataFrame(
        {
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
            "pressure_kpa": pressure,
            # No meaningful rolling/rate-of-change signal for i.i.d. rows,
            # so these mirror the raw reading - the model still sees the
            # same feature vector shape it gets in production.
            "temperature_rolling_mean": temperature,
            "vibration_rolling_mean": vibration,
            "temperature_rate_of_change": np.zeros(n),
            "vibration_rate_of_change": np.zeros(n),
        }
    )
    df["failure_risk"] = [_true_risk_label(t, v, rng) for t, v in zip(df["temperature_c"], df["vibration_mm_s"])]
    return df


def train_risk_model(seed: int = 42) -> Tuple[RandomForestClassifier, dict]:
    data = generate_training_data(seed=seed)
    X = data[FEATURE_COLUMNS]
    y = data["failure_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "n_test": len(y_test),
    }
    return model, metrics


def score_batch(model: RandomForestClassifier, features_df: pd.DataFrame) -> pd.Series:
    if len(features_df) == 0:
        return pd.Series(dtype=float, name="failure_risk_score")
    probabilities = model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1]
    return pd.Series(probabilities, index=features_df.index, name="failure_risk_score")
