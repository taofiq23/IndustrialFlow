from src.industrialflow.model import generate_training_data, score_batch, train_risk_model
from src.industrialflow.transform import FEATURE_COLUMNS


def test_training_data_has_both_classes():
    data = generate_training_data(n=1000)
    assert set(data["failure_risk"].unique()) == {0, 1}


def test_model_achieves_reasonable_accuracy_on_held_out_data():
    _, metrics = train_risk_model()

    # The labeling rule is fairly separable (temperature + vibration
    # thresholds) with only 5% label noise, so a real model should do
    # well - this is a floor, not a claim of perfection.
    assert metrics["accuracy"] > 0.85
    assert metrics["n_test"] > 0


def test_score_batch_returns_probabilities_in_range():
    model, _ = train_risk_model()
    data = generate_training_data(n=20, seed=1)

    scores = score_batch(model, data[FEATURE_COLUMNS])

    assert len(scores) == 20
    assert scores.between(0.0, 1.0).all()


def test_score_batch_handles_empty_dataframe():
    model, _ = train_risk_model()
    empty = generate_training_data(n=0)

    scores = score_batch(model, empty[FEATURE_COLUMNS])

    assert len(scores) == 0
