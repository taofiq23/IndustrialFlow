"""Simulates a production run: processes a sequence of batches (mostly
normal operating conditions, then batches with an injected sensor drift)
through the real pipeline, and reports what the audit log and drift
monitor actually caught - not what they're supposed to catch in theory.

Usage:
    python -m src.industrialflow.simulate
"""

import json
from pathlib import Path

from .data_generator import generate_batch
from .model import train_risk_model
from .pipeline import run_pipeline
from .schema import validate_batch
from .transform import engineer_features

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "run_results"


def _summarize(label: str, result) -> dict:
    temp_drift = next((r for r in result.drift_results if r.column == "temperature_c"), None)
    return {
        "batch": label,
        "run_id": result.run_id,
        "n_rows_in": result.n_rows_in,
        "n_rows_valid": result.n_rows_valid,
        "n_rows_quarantined": result.n_rows_quarantined,
        "drift_detected": result.drift_detected,
        "temperature_drift_p_value": temp_drift.p_value if temp_drift else None,
        "mean_risk_score": round(float(result.scored_df["failure_risk_score"].mean()), 4)
        if len(result.scored_df)
        else None,
    }


def render_markdown(model_metrics: dict, report_rows: list) -> str:
    lines = [
        "# IndustrialFlow Simulation Report",
        "",
        f"Risk model held-out accuracy: {model_metrics['accuracy']:.1%}, F1: {model_metrics['f1']:.3f} "
        f"(n={model_metrics['n_test']})",
        "",
        "| Batch | Rows in | Valid | Quarantined | Temp drift p-value | Drift flagged | Mean risk score |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report_rows:
        p_value = row["temperature_drift_p_value"]
        lines.append(
            f"| {row['batch']} | {row['n_rows_in']} | {row['n_rows_valid']} | {row['n_rows_quarantined']} | "
            f"{p_value:.2e} | {'YES' if row['drift_detected'] else 'no'} | {row['mean_risk_score']:.3f} |"
        )
    return "\n".join(lines)


def main():
    print("Training the risk model on synthetic labeled data...")
    model, metrics = train_risk_model()
    print(f"Model held-out accuracy: {metrics['accuracy']:.1%}, F1: {metrics['f1']:.3f} (n={metrics['n_test']})")

    reference_batch = generate_batch(n_rows=300, seed=1)
    reference_valid, _ = validate_batch(reference_batch)
    reference_features = engineer_features(reference_valid)

    report_rows = []

    print("\nProcessing 8 normal batches...")
    for i in range(8):
        batch = generate_batch(n_rows=200, seed=100 + i)
        result = run_pipeline(batch, model, reference_df=reference_features)
        report_rows.append(_summarize(f"normal_{i + 1}", result))

    print("Processing 2 batches with an injected sensor drift (+25C offset)...")
    for i in range(2):
        batch = generate_batch(n_rows=200, seed=200 + i, temperature_drift_c=25.0)
        result = run_pipeline(batch, model, reference_df=reference_features)
        report_rows.append(_summarize(f"drifted_{i + 1}", result))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps({"model_metrics": metrics, "runs": report_rows}, indent=2))

    markdown = render_markdown(metrics, report_rows)
    (RESULTS_DIR / "RESULTS.md").write_text(markdown)
    print("\n" + markdown)


if __name__ == "__main__":
    main()
