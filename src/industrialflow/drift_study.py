"""Simulation study of KS-based drift monitoring, using IndustrialFlow's own
column distributions, sample sizes and alarm rules.

    python -m src.industrialflow.drift_study rules        # false alarms and power by alarm rule
    python -m src.industrialflow.drift_study power_n      # power vs batch size
    python -m src.industrialflow.drift_study reference    # effect of reusing one fixed reference batch
    python -m src.industrialflow.drift_study autocorr     # effect of autocorrelated sensor readings
    python -m src.industrialflow.drift_study correlated   # effect of correlation between sensor columns
    python -m src.industrialflow.drift_study released     # the reference batch used by simulate.py

Every experiment writes a JSON file to study_results/ and is seeded, so a rerun
reproduces the same numbers.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy
from scipy import stats
from scipy.optimize import brentq
from scipy.signal import lfilter

from .monitoring import alarm_from_pvalues

STUDY_DIR = Path(__file__).resolve().parent.parent.parent / "study_results"

# Sizes seen in the released simulation: a 300-row reference and 200-row batches,
# each with 5% of rows quarantined by validation before drift testing.
N_REFERENCE = 285
N_BATCH = 190
ALPHA = 0.01

# (mean, sd, clipped_at_zero) exactly as in data_generator.generate_batch
COLUMNS = {
    "temperature_c": (60.0, 12.0, False),
    "vibration_mm_s": (4.0, 1.5, True),
    "pressure_kpa": (150.0, 20.0, False),
}


def sample(rng, reps: int, n: int, column: str, shift_sd: float = 0.0, scale: float = 1.0, phi: float = 0.0) -> np.ndarray:
    """(reps, n) readings of one column. Drift is a mean shift of shift_sd standard
    deviations and/or a multiplication of the standard deviation by scale. phi > 0 makes each
    row a stationary AR(1) series with lag-1 autocorrelation phi and unchanged marginal law.
    """
    mean, sd, clip = COLUMNS[column]
    if phi == 0.0:
        z = rng.standard_normal((reps, n))
    else:
        e = rng.standard_normal((reps, n + 1))
        z = lfilter([1.0], [1.0, -phi], np.sqrt(1.0 - phi**2) * e[:, 1:], axis=1, zi=phi * e[:, :1])[0]
    x = mean + sd * (shift_sd + scale * z)
    return np.maximum(x, 0.0) if clip else x


def ks_rows(reference: np.ndarray, batches: np.ndarray):
    """Two-sample KS statistic and p-value per row, using the same scipy call as monitoring.detect_drift."""
    reps = len(batches)
    d, p = np.empty(reps), np.empty(reps)
    for i in range(reps):
        result = stats.ks_2samp(reference[i], batches[i])
        d[i], p[i] = result.statistic, result.pvalue
    return d, p


def wilson(k: int, n: int) -> List[float]:
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(ci.low), float(ci.high)]


def rate(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool).ravel()
    return {"rate": float(mask.mean()), "ci": wilson(mask.sum(), len(mask)), "n": int(len(mask))}


def calibrated_k_of_n_alpha(target: float, k: int = 3, min_columns: int = 2) -> float:
    """Per-column alpha at which 'at least min_columns of k independent columns' has size `target`."""
    def size(a):
        return sum(stats.binom.pmf(j, k, a) for j in range(min_columns, k + 1))
    return float(brentq(lambda a: size(a) - target, 1e-9, 0.9))


def rule_configs() -> List[dict]:
    return [
        {"name": "uncorrected (released)", "rule": "uncorrected", "alpha": ALPHA},
        {"name": "Bonferroni", "rule": "bonferroni", "alpha": ALPHA},
        {"name": "Benjamini-Hochberg", "rule": "bh", "alpha": ALPHA},
        {"name": "Fisher combination", "rule": "fisher", "alpha": ALPHA},
        {"name": "2-of-3 columns (per-column 0.01)", "rule": "k_of_n", "alpha": ALPHA},
        {"name": "2-of-3 columns (size-calibrated)", "rule": "k_of_n", "alpha": calibrated_k_of_n_alpha(ALPHA)},
    ]


def alarms(p: np.ndarray, config: dict) -> np.ndarray:
    return alarm_from_pvalues(p, config["rule"], config["alpha"], min_columns=2)


SCENARIOS = [
    ("no drift", {}),
    ("temperature +0.25 sd", {"temperature_c": {"shift_sd": 0.25}}),
    ("temperature +0.5 sd", {"temperature_c": {"shift_sd": 0.5}}),
    ("temperature +1.0 sd", {"temperature_c": {"shift_sd": 1.0}}),
    ("temperature sd x1.5", {"temperature_c": {"scale": 1.5}}),
    ("all columns +0.15 sd", {c: {"shift_sd": 0.15} for c in COLUMNS}),
    ("all columns +0.25 sd", {c: {"shift_sd": 0.25} for c in COLUMNS}),
    ("all columns +0.35 sd", {c: {"shift_sd": 0.35} for c in COLUMNS}),
]


def _p_matrix(rng, reps: int, n_ref: int, n_batch: int, drift: Dict[str, dict]) -> np.ndarray:
    p = np.empty((reps, len(COLUMNS)))
    for j, column in enumerate(COLUMNS):
        reference = sample(rng, reps, n_ref, column)
        batches = sample(rng, reps, n_batch, column, **drift.get(column, {}))
        p[:, j] = ks_rows(reference, batches)[1]
    return p


def run_rules(reps: int = 10_000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    configs = rule_configs()
    out = {"reps": reps, "seed": seed, "n_reference": N_REFERENCE, "n_batch": N_BATCH, "alpha": ALPHA, "configs": configs, "scenarios": []}
    for label, drift in SCENARIOS:
        start = time.perf_counter()
        p = _p_matrix(rng, reps, N_REFERENCE, N_BATCH, drift)
        out["scenarios"].append(
            {
                "scenario": label,
                "per_column_rejection": {c: rate(p[:, j] < ALPHA) for j, c in enumerate(COLUMNS)},
                "rules": {cfg["name"]: rate(alarms(p, cfg)) for cfg in configs},
            }
        )
        print(f"[rules] {label}: uncorrected={out['scenarios'][-1]['rules']['uncorrected (released)']['rate']:.4f} ({time.perf_counter() - start:.0f}s)", flush=True)
    _save("rules.json", out)


def run_power_vs_n(reps: int = 5_000, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    configs = [c for c in rule_configs() if c["rule"] in ("uncorrected", "bonferroni", "fisher")]
    out = {"reps": reps, "seed": seed, "shift_sd": 0.5, "rows": []}
    for n in (50, 100, 190, 500):
        p = _p_matrix(rng, reps, int(1.5 * n), n, {"temperature_c": {"shift_sd": 0.5}})
        out["rows"].append({"n_batch": n, "n_reference": int(1.5 * n), "rules": {cfg["name"]: rate(alarms(p, cfg)) for cfg in configs}})
        print(f"[power_n] n={n} done", flush=True)
    _save("power_n.json", out)


def run_reference(n_references: int = 300, n_batches: int = 200, seed: int = 2) -> None:
    """Reuse one fixed reference sample for many null batches, as the released pipeline does."""
    rng = np.random.default_rng(seed)
    configs = rule_configs()
    out = {"n_references": n_references, "n_batches": n_batches, "seed": seed, "n_batch": N_BATCH, "by_reference_size": []}
    for m in (100, N_REFERENCE, 1000, 5000):
        start = time.perf_counter()
        p = np.empty((n_references, n_batches, len(COLUMNS)))
        for r in range(n_references):
            for j, column in enumerate(COLUMNS):
                reference = sample(rng, 1, m, column)
                batches = sample(rng, n_batches, N_BATCH, column)
                p[r, :, j] = ks_rows(np.repeat(reference, n_batches, axis=0), batches)[1]
        entry = {"n_reference": m, "rules": {}}
        for cfg in configs:
            a = alarms(p, cfg)  # (n_references, n_batches)
            pooled = float(a.mean())
            per_ref = a.mean(axis=1)
            prev, nxt = a[:, :-1], a[:, 1:]
            first = np.where(a.any(axis=1), a.argmax(axis=1) + 1, n_batches + 1)
            entry["rules"][cfg["name"]] = {
                "pooled_alarm_rate": pooled,
                "per_reference_rate_mean": float(per_ref.mean()),
                "per_reference_rate_sd": float(per_ref.std(ddof=1)),
                "per_reference_rate_p05": float(np.percentile(per_ref, 5)),
                "per_reference_rate_p95": float(np.percentile(per_ref, 95)),
                "per_reference_rate_max": float(per_ref.max()),
                "share_of_references_above_2x_pooled": float((per_ref > 2 * pooled).mean()) if pooled > 0 else None,
                "p_alarm_given_previous_alarm": float((prev & nxt).sum() / prev.sum()) if prev.sum() else None,
                "two_consecutive_alarm_rate": float((prev & nxt).mean()),
                "two_consecutive_if_independent": pooled**2,
                "share_with_alarm_within_50_batches": float((first <= 50).mean()),
                "share_with_alarm_within_50_if_independent": float(1 - (1 - pooled) ** 50),
                "share_never_alarming_in_run": float((first > n_batches).mean()),
            }
        out["by_reference_size"].append(entry)
        rates = np.asarray(alarms(p, configs[0]).mean(axis=1))
        entry["uncorrected_per_reference_rates"] = [round(float(x), 4) for x in rates]
        print(f"[reference] m={m} pooled uncorrected={entry['rules']['uncorrected (released)']['pooled_alarm_rate']:.4f} ({time.perf_counter() - start:.0f}s)", flush=True)
    _save("reference.json", out)


def lag1_autocorr(x: np.ndarray) -> np.ndarray:
    d = x - x.mean(axis=1, keepdims=True)
    return (d[:, :-1] * d[:, 1:]).sum(axis=1) / (d**2).sum(axis=1)


def _thin_ks(reference: np.ndarray, batches: np.ndarray, phi_hat: np.ndarray):
    p = np.empty(len(batches))
    for i in range(len(batches)):
        step = int(min(20, max(1, np.ceil((1 + phi_hat[i]) / (1 - phi_hat[i])))))
        p[i] = stats.ks_2samp(reference[i, ::step], batches[i, ::step]).pvalue
    return p


def _ess_p(d: np.ndarray, n_ref: int, n_batch: int, phi_hat: np.ndarray) -> np.ndarray:
    factor = (1 - phi_hat) / (1 + phi_hat)
    n_eff, m_eff = n_batch * factor, n_ref * factor
    return stats.kstwobign.sf(np.sqrt(n_eff * m_eff / (n_eff + m_eff)) * d)


BLOCK_LENGTH = 20
N_PERMUTATIONS = 499


def block_permutation_ks_p(reference: np.ndarray, batch: np.ndarray, rng, block: int = BLOCK_LENGTH, n_perm: int = N_PERMUTATIONS) -> float:
    """Permutation p-value for the two-sample KS statistic that shuffles whole contiguous blocks, not
    single rows, between the two samples. Blocks keep the short-range dependence of the series intact,
    so under 'no drift' the blocks are (approximately) exchangeable even when consecutive rows are not.
    Each series is trimmed to a whole number of blocks.
    """
    n_ref_blocks, n_batch_blocks = len(reference) // block, len(batch) // block
    values = np.concatenate([reference[: n_ref_blocks * block], batch[: n_batch_blocks * block]])
    n_blocks = n_ref_blocks + n_batch_blocks
    n_ref, n_batch = n_ref_blocks * block, n_batch_blocks * block
    order = np.argsort(values, kind="stable")

    def ks_from_batch_mask(mask: np.ndarray) -> np.ndarray:  # mask: (..., n_values) True where value is in the "batch" group
        sorted_mask = mask[..., order]
        batch_cdf = np.cumsum(sorted_mask, axis=-1) / n_batch
        ref_cdf = np.cumsum(~sorted_mask, axis=-1) / n_ref
        return np.abs(batch_cdf - ref_cdf).max(axis=-1)

    observed_mask = np.zeros(n_blocks, dtype=bool)
    observed_mask[n_ref_blocks:] = True
    observed = ks_from_batch_mask(np.repeat(observed_mask, block))

    chosen = np.argsort(rng.random((n_perm, n_blocks)), axis=1)[:, :n_batch_blocks]
    permuted_mask = np.zeros((n_perm, n_blocks), dtype=bool)
    np.put_along_axis(permuted_mask, chosen, True, axis=1)
    permuted = ks_from_batch_mask(np.repeat(permuted_mask, block, axis=1))
    return float((1 + np.sum(permuted >= observed - 1e-12)) / (1 + n_perm))


def run_autocorr(null_reps: int = 8_000, power_reps: int = 3_000, seed: int = 3) -> None:
    """Temperature column only. Both the reference and the batch are contiguous stationary AR(1) segments."""
    rng = np.random.default_rng(seed)
    perm_rng = np.random.default_rng(seed + 100)  # separate stream: adding permutation draws must not change the simulated data
    out = {"null_reps": null_reps, "power_reps": power_reps, "seed": seed, "alpha": ALPHA, "rows": []}
    for phi in (0.0, 0.3, 0.6, 0.8, 0.9):
        entry = {"phi": phi, "null": {}, "power": {}}
        for label, shift, reps in (("null", 0.0, null_reps), ("shift_0.5", 0.5, power_reps), ("shift_1.0", 1.0, power_reps)):
            start = time.perf_counter()
            reference = sample(rng, reps, N_REFERENCE, "temperature_c", phi=phi)
            batches = sample(rng, reps, N_BATCH, "temperature_c", shift_sd=shift, phi=phi)
            d, p_naive = ks_rows(reference, batches)
            phi_hat = np.clip(0.5 * (lag1_autocorr(reference) + lag1_autocorr(batches)), 0.0, 0.98)
            p_thin = _thin_ks(reference, batches, phi_hat)
            p_ess = _ess_p(d, N_REFERENCE, N_BATCH, phi_hat)
            p_block = np.array([block_permutation_ks_p(reference[i], batches[i], perm_rng) for i in range(reps)])
            target = entry["null"] if label == "null" else entry["power"].setdefault(label, {})
            target["naive"] = rate(p_naive < ALPHA)
            target["thinned"] = rate(p_thin < ALPHA)
            target["ess_adjusted"] = rate(p_ess < ALPHA)
            target["block_permutation"] = rate(p_block < ALPHA)
            if label == "null":
                target["mean_phi_hat"] = float(phi_hat.mean())
            print(f"[autocorr] phi={phi} {label}: naive={target['naive']['rate']:.4f} thinned={target['thinned']['rate']:.4f} ess={target['ess_adjusted']['rate']:.4f} block={target['block_permutation']['rate']:.4f} ({time.perf_counter() - start:.0f}s)", flush=True)
        out["rows"].append(entry)
    _save("autocorrelation.json", out)


def _correlated_columns(rng, reps: int, n: int, chol: np.ndarray, shift_sd: float) -> np.ndarray:
    """(reps, n, k) readings whose columns share a Gaussian copula with the given Cholesky factor."""
    z = rng.standard_normal((reps, n, chol.shape[0])) @ chol.T + shift_sd
    out = np.empty_like(z)
    for j, (mean, sd, clip) in enumerate(COLUMNS.values()):
        x = mean + sd * z[:, :, j]
        out[:, :, j] = np.maximum(x, 0.0) if clip else x
    return out


def run_correlated(reps: int = 10_000, seed: int = 4) -> None:
    """Contemporaneous correlation between the three sensor columns (rows remain independent)."""
    rng = np.random.default_rng(seed)
    configs = rule_configs()
    out = {"reps": reps, "seed": seed, "alpha": ALPHA, "configs": configs, "rows": []}
    for rho in (0.0, 0.3, 0.6, 0.9):
        corr = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
        chol = np.linalg.cholesky(corr)
        entry = {"rho": rho, "scenarios": {}}
        for label, shift in (("no drift", 0.0), ("all columns +0.25 sd", 0.25)):
            start = time.perf_counter()
            reference = _correlated_columns(rng, reps, N_REFERENCE, chol, 0.0)
            batches = _correlated_columns(rng, reps, N_BATCH, chol, shift)
            p = np.column_stack([ks_rows(reference[:, :, j], batches[:, :, j])[1] for j in range(3)])
            entry["scenarios"][label] = {cfg["name"]: rate(alarms(p, cfg)) for cfg in configs}
            print(f"[correlated] rho={rho} {label}: uncorrected={entry['scenarios'][label]['uncorrected (released)']['rate']:.4f} fisher={entry['scenarios'][label]['Fisher combination']['rate']:.4f} ({time.perf_counter() - start:.0f}s)", flush=True)
        out["rows"].append(entry)
    _save("correlated.json", out)


def run_released_reference(n_batches: int = 400, reference_seed: int = 1, batch_seed_start: int = 5000) -> None:
    """False-alarm behavior of the real pipeline code path against the reference batch that
    simulate.py uses (generate_batch(n_rows=300, seed=1)), on clean batches from the real generator."""
    import pandas as pd  # noqa: F401  (imported lazily: the other experiments do not need the pipeline stack)

    from .data_generator import generate_batch
    from .monitoring import MONITORED_COLUMNS, decide_alarm, detect_drift
    from .schema import validate_batch
    from .transform import engineer_features

    reference = engineer_features(validate_batch(generate_batch(n_rows=300, seed=reference_seed))[0])
    rule_names = ["uncorrected", "bonferroni", "bh", "fisher"]
    alarm_counts = {r: 0 for r in rule_names}
    column_counts = {c: 0 for c in MONITORED_COLUMNS}
    for i in range(n_batches):
        current = engineer_features(validate_batch(generate_batch(n_rows=200, seed=batch_seed_start + i))[0])
        results = detect_drift(reference, current)
        for rule in rule_names:
            alarm_counts[rule] += decide_alarm(results, rule, ALPHA)
        for r in results:
            column_counts[r.column] += bool(r.drifted)
    out = {
        "reference_rows": int(len(reference)),
        "reference_seed": reference_seed,
        "n_clean_batches": n_batches,
        "batch_seed_start": batch_seed_start,
        "per_column_rejection": {c: {"count": k, "rate": k / n_batches, "ci": wilson(k, n_batches)} for c, k in column_counts.items()},
        "batch_alarm": {r: {"count": k, "rate": k / n_batches, "ci": wilson(k, n_batches)} for r, k in alarm_counts.items()},
    }
    print(json.dumps(out["batch_alarm"]), flush=True)
    _save("released_reference.json", out)


def _save(name: str, payload: dict) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"environment": {"numpy": np.__version__, "scipy": scipy.__version__}, **payload}
    (STUDY_DIR / name).write_text(json.dumps(payload, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["rules", "power_n", "reference", "autocorr", "correlated", "released", "all"])
    args = parser.parse_args(argv)
    runners = {
        "rules": run_rules,
        "power_n": run_power_vs_n,
        "reference": run_reference,
        "autocorr": run_autocorr,
        "correlated": run_correlated,
        "released": run_released_reference,
    }
    for name in runners if args.experiment == "all" else [args.experiment]:
        runners[name]()


if __name__ == "__main__":
    main()
