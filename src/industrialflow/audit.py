"""SQLite-backed audit trail: every pipeline run is recorded with full
provenance - row counts in/out, why rows were quarantined, drift status,
risk scores - so any run's history can be reconstructed later, not just
its final output.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from .config import AUDIT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    n_rows_in INTEGER NOT NULL,
    n_rows_valid INTEGER NOT NULL,
    n_rows_quarantined INTEGER NOT NULL,
    quarantine_reasons TEXT NOT NULL,
    drift_detected INTEGER NOT NULL,
    drift_details TEXT NOT NULL,
    mean_risk_score REAL,
    n_high_risk INTEGER
);
"""


@contextmanager
def _connect(db_path: Optional[Path] = None):
    # Resolved at call time (not baked into a default argument at import
    # time) so tests can monkeypatch module.AUDIT_DB_PATH and actually
    # have it take effect, instead of silently writing to the real file.
    db_path = db_path or AUDIT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_run(run_id: str, metadata: dict, db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, started_at, n_rows_in, n_rows_valid, n_rows_quarantined,
                quarantine_reasons, drift_detected, drift_details, mean_risk_score, n_high_risk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                metadata["started_at"],
                metadata["n_rows_in"],
                metadata["n_rows_valid"],
                metadata["n_rows_quarantined"],
                json.dumps(metadata["quarantine_reasons"]),
                int(metadata["drift_detected"]),
                json.dumps(metadata["drift_details"]),
                metadata.get("mean_risk_score"),
                metadata.get("n_high_risk"),
            ),
        )


def query_runs(db_path: Optional[Path] = None, limit: int = 50) -> List[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_run(run_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
