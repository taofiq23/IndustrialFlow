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
- **`normal_6`'s flag is a real false alarm, and the explanation originally given here was incomplete.** Its temperature p-value (0.52) is nowhere near the 0.01 threshold - the flag came from `pressure_kpa` (p=0.0004), confirmed by querying that run's full drift detail from the audit log. Testing 3 columns at 1% each gives a ~3% false-alarm chance *per batch* (`1 - 0.99^3`), which this README first blamed entirely on chance. The [simulation study](#drift-monitoring-study) showed that isn't the whole story: this simulation reuses one fixed 300-row reference batch, and against that particular reference the pressure column rejects **6.75%** of clean batches (27 of 400, 95% CI 4.7-9.6%) instead of 1%, so the monitor flags **8.0%** of clean batches instead of ~3%. Multiple testing raises the false-alarm rate about 3x; this unrepresentative reference raises it a further ~2.7x, and no correction rule fixes that. Reproduce with `python -m src.industrialflow.drift_study released`.
- **Every batch quarantines a consistent ~5% of rows**, matching the generator's injected `bad_row_rate=0.05` - and the audit log records *why*, e.g. `"vibration_mm_s: in_range(0.0, 15.0)": 5` rows, not just a count. That's the difference between "something failed" and an actually debuggable audit trail.
- **The risk model's 97% accuracy is measured against a known synthetic ground-truth rule** (temperature > 85°C AND vibration > 6mm/s, with 5% label noise so the task isn't trivially separable) - a real, verifiable number for this task, not a claim about real-world predictive maintenance accuracy.

## Drift-monitoring study

The `normal_6` false alarm prompted a simulation study of how KS-based drift monitors behave, written up in [paper/paper.pdf](paper/paper.pdf) (LaTeX source alongside). Everything below is measured by simulation with this pipeline's own column distributions, sample sizes and alarm logic; the paper has the full tables and confidence intervals.

| Question | Finding |
|---|---|
| How often does the default monitor false-alarm? | On **2.9%** of clean batches (about one in 34), not the 1% the per-column threshold suggests. |
| Which alarm rule is best at equal (1%) batch-level size? | Bonferroni / Benjamini-Hochberg for single-sensor drift; Fisher for small simultaneous drift in all sensors; "2-of-3 columns" is nearly blind to single-sensor faults. |
| Does reusing one fixed reference matter? | Yes: with a 100-row reference, an alarm is followed by another with probability 20% (vs 2.9% average); the released 285-row reference is still 2.3x worse than independence. Use a larger reference (1,000+ rows) and audit it on held-out clean data. |
| What about autocorrelated sensor readings? | Naive KS false-alarms on 17% of clean batches at lag-1 autocorrelation 0.6 (60% at 0.9). A block-permutation test restores near-nominal size up to 0.6. |
| Correlated sensor columns? | Fisher's rule becomes anti-conservative (4.2% at correlation 0.9); Bonferroni/BH stay valid. |

The alarm rule is now a parameter; the default keeps the historical "any column p < 0.01" behavior:

```python
from src.industrialflow.pipeline import run_pipeline

run_pipeline(batch, model, reference_df)                                  # default: "uncorrected"
run_pipeline(batch, model, reference_df, drift_rule="bonferroni", drift_alpha=0.001)
# rules: "uncorrected", "bonferroni", "bh", "fisher", "k_of_n"  (see monitoring.ALARM_RULES)
```

The rule used is recorded in each run's audit-log drift details. To reproduce the study (about 30 minutes on one CPU core; results are seeded and land in `study_results/`):

```bash
pip install -r requirements-study.txt
python -m src.industrialflow.drift_study rules        # false alarms and power by alarm rule
python -m src.industrialflow.drift_study power_n      # power vs batch size
python -m src.industrialflow.drift_study reference    # fixed-reference effect
python -m src.industrialflow.drift_study autocorr     # autocorrelated readings + remedies
python -m src.industrialflow.drift_study correlated   # correlated columns
python -m src.industrialflow.drift_study released     # this simulation's own reference batch
python -m src.industrialflow.study_figures            # rebuild paper/ figures and tables
```

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
- The default drift alarm ("uncorrected") tests each monitored column independently at a fixed p-value threshold with no multiple-testing correction, left as the default so the simulation above stays reproducible; the corrected rules are available via `drift_rule` and their behavior is characterized in the [study](#drift-monitoring-study). The study itself is a simulation with Gaussian marginals and step-change drift, not an evaluation on real plant data.
- The audit log is a single local SQLite file for this demo; a real production deployment would want a proper time-series-capable store and a way to correlate audit entries with the actual upstream data files, not just row counts.

## License

MIT - see [LICENSE](LICENSE).
