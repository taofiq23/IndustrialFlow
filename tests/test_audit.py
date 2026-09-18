from pathlib import Path

from src.industrialflow import audit


def _metadata(**overrides):
    base = {
        "started_at": "2026-01-01T00:00:00+00:00",
        "n_rows_in": 100,
        "n_rows_valid": 95,
        "n_rows_quarantined": 5,
        "quarantine_reasons": {"temperature_c: out_of_range": 5},
        "drift_detected": False,
        "drift_details": [],
        "mean_risk_score": 0.12,
        "n_high_risk": 3,
    }
    base.update(overrides)
    return base


def test_record_and_get_run(tmp_path: Path):
    db_path = tmp_path / "audit.db"
    audit.record_run("run-1", _metadata(), db_path=db_path)

    result = audit.get_run("run-1", db_path=db_path)

    assert result is not None
    assert result["n_rows_in"] == 100
    assert result["n_rows_quarantined"] == 5


def test_get_run_returns_none_for_unknown_id(tmp_path: Path):
    db_path = tmp_path / "audit.db"
    audit.record_run("run-1", _metadata(), db_path=db_path)

    assert audit.get_run("does-not-exist", db_path=db_path) is None


def test_query_runs_orders_most_recent_first(tmp_path: Path):
    db_path = tmp_path / "audit.db"
    audit.record_run("run-1", _metadata(started_at="2026-01-01T00:00:00+00:00"), db_path=db_path)
    audit.record_run("run-2", _metadata(started_at="2026-01-02T00:00:00+00:00"), db_path=db_path)

    runs = audit.query_runs(db_path=db_path)

    assert [r["run_id"] for r in runs] == ["run-2", "run-1"]
