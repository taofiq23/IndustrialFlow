# IndustrialFlow

A production data/ML pipeline built around a claim the MLOps literature backs up repeatedly: **data pipeline failures - schema drift, sensor recalibration, silent data-quality degradation - are the leading cause of production ML failures, more common than actual model bugs.** Yet most teams test their application code far more rigorously than the data flowing through it. This pipeline validates every row against a real schema (bad rows are quarantined with a specific reason, never silently dropped or allowed to crash the run), records a full audit trail for every run, and monitors for statistical drift using a real Kolmogorov-Smirnov test - not a hand-wave.

Domain: simulated industrial sensor data (temperature/vibration/pressure) feeding a predictive-maintenance risk model - a genuine, common industrial ML use case.

## The result

A simulated production run: 8 batches of normal sensor data, then 2 batches with an injected sensor drift (a systematic +25°C offset, like a sensor recalibration or an ambient-condition change).

| Batch | Rows in | Valid | Quarantined | Temp drift p-value | Drift flagged | Mean risk score |
|---|---|---|---|---|---|---|
| normal_1 | 200 | 190 | 10 | 0.611 | no | 0.090 |
| normal_2 | 200 | 190 | 10 | 0.089 | no | 0.086 |
| normal_3 | 200 | 190 | 10 | 0.355 | no | 0.093 |
| normal_4 | 200 | 190 | 10 | 0.611 | no | 0.089 |
| normal_5 | 200 | 190 | 10 | 0.229 | no | 0.089 |
| normal_6 | 200 | 190 | 10 | 0.519 | **YES** | 0.083 |
| normal_7 | 200 | 190 | 10 | 0.405 | no | 0.083 |
| normal_8 | 200 | 190 | 10 | 0.355 | no | 0.085 |
| drifted_1 | 200 | 189 | 11 | **0.000** | YES | 0.113 |
| drifted_2 | 200 | 190 | 10 | **0.000** | YES | 0.111 |

Risk model held-out accuracy: 97.0%, F1: 0.924 (n=600). Full output in [run_results/RESULTS.md](run_results/RESULTS.md) and [run_results/results.json](run_results/results.json).

### What this actually shows

- **The drifted batches are caught cleanly and precisely.** Temperature's p-value drops to effectively 0 in both drifted batches, while a per-column breakdown (queryable from the audit log) shows vibration (p=0.88) and pressure (p=0.13) correctly staying unflagged - the monitor identifies *which* sensor drifted, not just "something is different."
- **`normal_6`'s flag is a real false positive, and it's worth explaining rather than hiding.** Its temperature p-value (0.52) is nowhere near the 0.01 threshold - the flag actually came from `pressure_kpa` (p=0.0004), by chance, confirmed by querying that run's full drift detail from the audit log. Testing 3 independent columns at a 1% significance level each gives roughly a 3% false-positive chance *per batch* (`1 - 0.99^3`), and across 8 normal batches, at least one false alarm has around a 22% chance of happening (`1 - 0.97^8`) - so seeing exactly one here isn't a bug, it's the expected behavior of uncorrected multiple hypothesis testing. A production version of this monitor would want a multiple-testing correction (e.g., Bonferroni) or a "flag only if 2+ columns agree" rule; this project reports the raw, honest behavior instead of quietly re-tuning the threshold until the demo looked clean.
- **Every batch quarantines a consistent ~5% of rows**, matching the generator's injected `bad_row_rate=0.05` - and the audit log records *why*, e.g. `"vibration_mm_s: in_range(0.0, 15.0)": 5` rows, not just a count. That's the difference between "something failed" and an actually debuggable audit trail.
- **The risk model's 97% accuracy is measured against a known synthetic ground-truth rule** (temperature > 85°C AND vibration > 6mm/s, with 5% label noise so the task isn't trivially separable) - a real, verifiable number for this task, not a claim about real-world predictive maintenance accuracy.

## Architecture

```
raw sensor batch
      │
      ▼
 ┌─────────────┐   quarantined rows + reason
 │  Validate   │──────────────────────────────► quarantine
 │ (Pandera)   │
 └──────┬──────┘
        │ valid rows only
        ▼
 ┌─────────────┐
 │  Transform   │  rolling stats, rate of change (per equipment)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │    Score     │  RandomForest risk model
 └──────┬──────┘
        ▼
 ┌─────────────┐        ┌───────────────────┐
 │Drift check  │───────►│  Audit log (SQLite) │  every run: row counts,
 │(KS-test vs  │        │                     │  quarantine reasons, drift
 │ reference)  │        └───────────────────┘  results, risk scores
 └─────────────┘
```

## Quickstart

```bash
git clone <this-repo-url>
cd IndustrialFlow
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Reproduce the simulation and report above
python -m src.industrialflow.simulate

# Query the audit trail directly
python -c "from src.industrialflow import audit; import json; print(json.dumps(audit.query_runs(limit=5), indent=2))"
```

## Tests

```bash
pytest
```

Covers schema validation (each fault type - null, out-of-range, negative - is quarantined independently and correctly), feature engineering (including that rolling/rate-of-change features are computed per-equipment, not leaked across units sharing a batch), the risk model (real accuracy on held-out synthetic data), the audit log (SQLite read/write/query), drift detection (same-distribution samples correctly pass, shifted ones correctly fail, using real statistical assertions, not mocked results), and a full pipeline integration test.

## Project structure

```
IndustrialFlow/
├── src/industrialflow/
│   ├── data_generator.py  # synthetic sensor batches with injected bad rows + optional drift
│   ├── schema.py           # Pandera validation -> (valid, quarantined_with_reason)
│   ├── transform.py         # per-equipment feature engineering
│   ├── model.py              # risk model, trained on synthetic ground-truth data
│   ├── audit.py               # SQLite audit trail
│   ├── monitoring.py           # KS-test drift detection
│   ├── pipeline.py              # orchestrates validate -> transform -> score -> audit -> drift check
│   └── simulate.py               # the real run behind the report above
├── tests/
└── run_results/                   # a real committed run (see above)
```

## Limitations

- Sensor data is synthetic by design - it exists to make validation, drift detection, and the audit trail exercisable and verifiable against known ground truth, not to represent real industrial telemetry.
- The risk model's labeling rule is a simplified, known synthetic function; its 97% accuracy is a real, measured number for *this* task, not a claim about real-world predictive-maintenance performance.
- Drift detection here tests each monitored column independently at a fixed p-value threshold with no multiple-testing correction - which is exactly what produced the `normal_6` false positive discussed above, left in deliberately rather than tuned away.
- The audit log is a single local SQLite file for this demo; a real production deployment would want a proper time-series-capable store and a way to correlate audit entries with the actual upstream data files, not just row counts.

## License

MIT - see [LICENSE](LICENSE).
