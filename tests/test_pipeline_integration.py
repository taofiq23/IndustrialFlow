from src.industrialflow import audit
from src.industrialflow.data_generator import generate_batch
from src.industrialflow.model import train_risk_model
from src.industrialflow.pipeline import run_pipeline
from src.industrialflow.schema import validate_batch
from src.industrialflow.transform import engineer_features


def test_pipeline_quarantines_bad_rows_and_scores_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", tmp_path / "audit.db")
    model, _ = train_risk_model()

    batch = generate_batch(n_rows=100, seed=7, bad_row_rate=0.1)
    result = run_pipeline(batch, model, reference_df=None)

    assert result.n_rows_in == 100
    assert result.n_rows_valid + result.n_rows_quarantined == 100
    assert result.n_rows_quarantined > 0  # bad_row_rate=0.1 guarantees some bad rows
    assert len(result.scored_df) == result.n_rows_valid
    assert result.scored_df["failure_risk_score"].between(0.0, 1.0).all()

    recorded = audit.get_run(result.run_id, db_path=tmp_path / "audit.db")
    assert recorded is not None
    assert recorded["n_rows_quarantined"] == result.n_rows_quarantined


def test_pipeline_detects_injected_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", tmp_path / "audit.db")
    model, _ = train_risk_model()

    reference_batch = generate_batch(n_rows=300, seed=1)
    reference_valid, _ = validate_batch(reference_batch)
    reference_features = engineer_features(reference_valid)

    normal_batch = generate_batch(n_rows=200, seed=100)
    normal_result = run_pipeline(normal_batch, model, reference_df=reference_features)

    drifted_batch = generate_batch(n_rows=200, seed=200, temperature_drift_c=25.0)
    drifted_result = run_pipeline(drifted_batch, model, reference_df=reference_features)

    assert normal_result.drift_detected is False
    assert drifted_result.drift_detected is True
