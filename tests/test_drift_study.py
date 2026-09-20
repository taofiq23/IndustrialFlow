import numpy as np
import pytest

from src.industrialflow.data_generator import generate_batch
from src.industrialflow.drift_study import (
    COLUMNS,
    _correlated_columns,
    _ess_p,
    block_permutation_ks_p,
    calibrated_k_of_n_alpha,
    ks_rows,
    lag1_autocorr,
    sample,
)
from src.industrialflow.monitoring import ColumnDriftResult, alarm_from_pvalues, decide_alarm


def test_uncorrected_alarms_on_single_marginal_column_but_corrections_do_not():
    p = [0.005, 0.5, 0.5]
    assert alarm_from_pvalues(p, "uncorrected", 0.01)
    assert not alarm_from_pvalues(p, "bonferroni", 0.01)
    assert not alarm_from_pvalues(p, "bh", 0.01)
    assert not alarm_from_pvalues(p, "fisher", 0.01)
    assert not alarm_from_pvalues(p, "k_of_n", 0.01)


def test_every_rule_alarms_on_strong_multi_column_evidence():
    p = [0.0001, 0.0002, 0.3]
    for rule in ("uncorrected", "bonferroni", "bh", "fisher", "k_of_n"):
        assert alarm_from_pvalues(p, rule, 0.01), rule


def test_rules_vectorize_over_leading_axes():
    p = np.array([[0.5, 0.5, 0.5], [0.0001, 0.0001, 0.0001]])
    assert alarm_from_pvalues(p, "fisher", 0.01).tolist() == [False, True]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        alarm_from_pvalues([0.5, 0.5], "made_up")


def test_decide_alarm_matches_array_api():
    results = [ColumnDriftResult("a", 0.1, 0.005, True), ColumnDriftResult("b", 0.1, 0.5, False)]
    assert decide_alarm(results, "uncorrected", 0.01) is True
    assert decide_alarm(results, "bonferroni", 0.01) is False


def test_calibrated_k_of_n_alpha_gives_requested_size():
    a = calibrated_k_of_n_alpha(0.01)
    assert 3 * a**2 - 2 * a**3 == pytest.approx(0.01, abs=1e-6)


def test_sampler_marginals_match_the_real_data_generator():
    real = generate_batch(n_rows=20_000, seed=7, bad_row_rate=0.0)
    rng = np.random.default_rng(0)
    for column in COLUMNS:
        fast = sample(rng, 1, 20_000, column)[0]
        assert fast.mean() == pytest.approx(real[column].mean(), rel=0.02)
        assert fast.std() == pytest.approx(real[column].std(), rel=0.05)
        assert fast.min() >= 0.0 if column == "vibration_mm_s" else True


def test_ar1_sampler_has_requested_autocorrelation_and_marginal_variance():
    rng = np.random.default_rng(1)
    x = sample(rng, 400, 500, "temperature_c", phi=0.8)
    assert lag1_autocorr(x).mean() == pytest.approx(0.8, abs=0.03)
    assert x.std() == pytest.approx(12.0, rel=0.05)


def test_null_false_alarm_rates_follow_theory_for_independent_data():
    rng = np.random.default_rng(2)
    reps = 1500
    p = np.empty((reps, 3))
    for j, column in enumerate(COLUMNS):
        p[:, j] = ks_rows(sample(rng, reps, 285, column), sample(rng, reps, 190, column))[1]
    uncorrected = alarm_from_pvalues(p, "uncorrected", 0.01).mean()
    bonferroni = alarm_from_pvalues(p, "bonferroni", 0.01).mean()
    assert 0.015 < uncorrected < 0.05
    assert bonferroni < 0.02


def test_correlated_columns_have_requested_correlation():
    rng = np.random.default_rng(3)
    chol = np.linalg.cholesky(np.full((3, 3), 0.6) + 0.4 * np.eye(3))
    flat = _correlated_columns(rng, 50, 2000, chol, 0.0).reshape(-1, 3)
    assert np.corrcoef(flat[:, 0], flat[:, 2])[0, 1] == pytest.approx(0.6, abs=0.03)


def test_block_permutation_p_value_is_uniform_ish_under_null_and_small_under_shift():
    rng = np.random.default_rng(5)
    perm_rng = np.random.default_rng(6)
    ref = sample(rng, 60, 285, "temperature_c")
    same = sample(rng, 60, 190, "temperature_c")
    shifted = sample(rng, 60, 190, "temperature_c", shift_sd=1.0)
    null_p = np.array([block_permutation_ks_p(ref[i], same[i], perm_rng, n_perm=199) for i in range(60)])
    shift_p = np.array([block_permutation_ks_p(ref[i], shifted[i], perm_rng, n_perm=199) for i in range(60)])
    assert 0.3 < null_p.mean() < 0.7
    assert (shift_p < 0.05).mean() > 0.8


def test_block_permutation_p_value_is_bounded_and_deterministic_given_rng():
    rng = np.random.default_rng(7)
    a, b = sample(rng, 1, 285, "temperature_c")[0], sample(rng, 1, 190, "temperature_c")[0]
    p1 = block_permutation_ks_p(a, b, np.random.default_rng(1), n_perm=99)
    p2 = block_permutation_ks_p(a, b, np.random.default_rng(1), n_perm=99)
    assert p1 == p2 and 1 / 100 <= p1 <= 1.0


def test_ess_adjustment_widens_p_values_when_autocorrelated():
    d = np.array([0.2])
    independent = _ess_p(d, 285, 190, np.array([0.0]))
    correlated = _ess_p(d, 285, 190, np.array([0.8]))
    assert correlated[0] > independent[0]
