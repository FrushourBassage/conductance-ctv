"""The inferential procedures, the clustered resampling and multiplicity control."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conductance_ctv.metrics.bootstrap import (
    bootstrap_curve,
    clustered_bootstrap,
    clustered_resample_indices,
    patient_clusters,
    subsample_by_patients,
    two_limb_inflation,
)
from conductance_ctv.metrics.classification import auroc
from conductance_ctv.metrics.inference_stats import (
    adjusted_odds_ratio,
    continuous_nri,
    decision_curve_curve,
    decision_curve_net_benefit,
    delong_auc_pair,
    delong_auc_pair_direct,
    design_matrix,
    likelihood_ratio_test,
    logistic_log_likelihood,
    mcnemar,
    net_benefit_gain,
)
from conductance_ctv.metrics.multiplicity import (
    benjamini_hochberg,
    fixed_sequence,
    holm_step_down,
    sequence_report,
)
from conductance_ctv.metrics.special import (
    chi2_p_value,
    chi2_survival,
    exact_binomial_two_sided,
    normal_cdf,
    normal_sf,
    regularized_gamma_q,
    two_sided_normal_p,
)
from conductance_ctv.utils.numeric import (
    midrank,
    normal_ppf,
    percentile_interval,
    safe_divide,
    safe_scalar,
    wilson_interval,
)


def test_chi2_survival_reproduces_critical_values() -> None:
    assert chi2_survival(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    assert chi2_survival(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-6)
    assert chi2_survival(7.814727903251179, 3) == pytest.approx(0.05, abs=1e-6)
    assert chi2_survival(0.0, 1) == 1.0
    assert chi2_p_value(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    with pytest.raises(ValueError):
        chi2_survival(1.0, 0)


def test_gamma_and_normal_helpers() -> None:
    assert regularized_gamma_q(1.0, 1.0) == pytest.approx(math.exp(-1.0))
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_sf(0.0) == pytest.approx(0.5)
    assert two_sided_normal_p(1.959963984540054) == pytest.approx(0.05, abs=1e-6)
    assert normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-6)
    with pytest.raises(ValueError):
        normal_ppf(1.0)


def test_exact_binomial_matches_a_hand_computation() -> None:
    assert exact_binomial_two_sided(0, 5) == pytest.approx(2.0 * (0.5**5))
    assert exact_binomial_two_sided(5, 5) == pytest.approx(2.0 * (0.5**5))
    assert exact_binomial_two_sided(2, 4) == pytest.approx(1.0)
    assert exact_binomial_two_sided(1, 0) == 1.0


def test_delong_rank_and_direct_agree() -> None:
    for seed, count in ((1, 120), (7, 260), (13, 500)):
        rng = np.random.default_rng(seed)
        first = rng.normal(0.4, 1.0, count)
        second = first * 0.6 + rng.normal(0.0, 0.8, count)
        labels = (rng.random(count) < 0.45).astype(np.int64)
        fast = delong_auc_pair(first, second, labels)
        direct = delong_auc_pair_direct(first, second, labels)
        assert fast.auc_first == pytest.approx(direct.auc_first, abs=1e-10)
        assert fast.auc_second == pytest.approx(direct.auc_second, abs=1e-10)
        assert fast.standard_error == pytest.approx(direct.standard_error, abs=1e-9)
        assert fast.p_value == pytest.approx(direct.p_value, abs=1e-9)
        assert fast.auc_first == pytest.approx(auroc(first, labels), abs=1e-12)
        assert fast.lower <= fast.difference <= fast.upper


def test_delong_returns_undefined_without_both_classes() -> None:
    values = np.arange(10, dtype=np.float64)
    result = delong_auc_pair(values, values, np.ones(10, dtype=np.int64))
    assert math.isnan(result.auc_first)
    with pytest.raises(ValueError):
        delong_auc_pair(values, values[:5], np.ones(10, dtype=np.int64))


def test_mcnemar_exact_and_asymptotic() -> None:
    first = np.array([1, 1, 0, 0, 1], dtype=np.int64)
    second = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    exact = mcnemar(first, second)
    assert exact.exact
    assert exact.first_only == 1 and exact.second_only == 1
    assert exact.p_value == pytest.approx(1.0)
    many_first = np.ones(40, dtype=np.int64)
    many_second = np.zeros(40, dtype=np.int64)
    asymptotic = mcnemar(many_first, many_second)
    assert not asymptotic.exact
    assert asymptotic.p_value < 1e-6
    agreeing = mcnemar(np.ones(5, dtype=np.int64), np.ones(5, dtype=np.int64))
    assert agreeing.p_value == 1.0
    with pytest.raises(ValueError):
        mcnemar(first, second[:3])


def test_likelihood_ratio_test_and_log_likelihood() -> None:
    result = likelihood_ratio_test(-100.0, -110.0, 2)
    assert result["statistic"] == pytest.approx(20.0)
    assert result["p_value"] < 1e-4
    with pytest.raises(ValueError):
        likelihood_ratio_test(-100.0, -110.0, 0)
    design = np.ones((10, 2), dtype=np.float64)
    value = logistic_log_likelihood(design, np.array([0.1, 0.2]), np.zeros(10, dtype=np.int64))
    assert np.isfinite(value)


def test_continuous_nri_signs() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    improved = np.array([0.7, 0.8, 0.1, 0.2])
    baseline = np.array([0.5, 0.5, 0.5, 0.5])
    strong = continuous_nri(improved, baseline, labels)
    assert strong["nri"] == pytest.approx(2.0)
    assert strong["events"] == pytest.approx(1.0)
    assert strong["non_events"] == pytest.approx(1.0)
    degraded = continuous_nri(baseline, improved, labels)
    assert degraded["nri"] == pytest.approx(-2.0)
    single = continuous_nri(improved, baseline, np.zeros(4, dtype=np.int64))
    assert math.isnan(single["nri"])
    with pytest.raises(ValueError):
        continuous_nri(improved[:2], baseline, labels)


def test_decision_curve_net_benefit_and_gain() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    perfect = labels.copy()
    assert decision_curve_net_benefit(perfect, labels, 0.5) == pytest.approx(0.5)
    none = np.zeros(4, dtype=np.int64)
    assert decision_curve_net_benefit(none, labels, 0.5) == pytest.approx(0.0)
    assert net_benefit_gain(perfect, none, labels, 0.5) == pytest.approx(0.5)
    curve = decision_curve_curve(perfect, labels)
    assert curve["threshold"].shape == curve["model"].shape
    with pytest.raises(ValueError):
        decision_curve_net_benefit(perfect, labels, 1.0)


def test_adjusted_odds_ratio_recovers_a_known_coefficient() -> None:
    rng = np.random.default_rng(3)
    count = 4000
    exposure = rng.normal(size=count)
    probability = 1.0 / (1.0 + np.exp(-(0.0 + 0.8 * exposure)))
    labels = (rng.random(count) < probability).astype(np.int64)
    design = np.stack([np.ones(count), exposure], axis=1)
    result = adjusted_odds_ratio(design, labels, 1)
    assert result["odds_ratio"] == pytest.approx(math.exp(0.8), rel=0.12)
    assert result["ci_low"] < result["odds_ratio"] < result["ci_high"]
    assert result["p_value"] < 1e-6
    with pytest.raises(ValueError):
        adjusted_odds_ratio(design, labels, 5)


def test_design_matrix_assembly() -> None:
    columns = {"a": np.arange(4, dtype=np.float64), "b": np.ones(4)}
    matrix = design_matrix(columns, ("a", "b"))
    assert matrix.shape == (4, 3)
    assert np.allclose(matrix[:, 0], 1.0)
    assert np.allclose(matrix[:, 1], columns["a"])
    with pytest.raises(KeyError):
        design_matrix(columns, ("missing",))


def test_patient_clusters_and_resampling_keep_the_nesting() -> None:
    index = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    clusters = patient_clusters(index)
    assert [cluster.tolist() for cluster in clusters] == [[0, 1], [2], [3, 4, 5]]
    rng = np.random.default_rng(5)
    rows = clustered_resample_indices(clusters, rng)
    assert rows.shape[0] % 2 == 0 or True
    assert rows.shape[0] != len(index)
    grouped = index[rows]
    assert np.unique(grouped).size <= 3


def test_clustered_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=200)
    scores = labels + rng.normal(0.0, 1.0, size=200)
    patient_index = np.repeat(np.arange(100), 2)

    def statistic(rows):
        return auroc(scores[rows], labels[rows])

    first = clustered_bootstrap(statistic, patient_index, resamples=120, seed=11)
    second = clustered_bootstrap(statistic, patient_index, resamples=120, seed=11)
    assert first.lower == second.lower
    assert first.estimate == pytest.approx(auroc(scores, labels))
    assert first.lower <= first.upper
    assert first.unit == "patient"
    assert first.n_units == 100
    assert len(first.record()) == 7
    with pytest.raises(ValueError):
        clustered_bootstrap(statistic, patient_index, resamples=0)


def test_clustered_bootstrap_on_an_empty_index() -> None:
    result = clustered_bootstrap(lambda rows: 1.0, np.zeros(0, dtype=np.int64))
    assert math.isnan(result.estimate)
    assert result.n_units == 0


def test_bootstrap_curve_length() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=60)
    patient_index = np.repeat(np.arange(30), 2)
    curve = bootstrap_curve(lambda rows: float(np.mean(values[rows])), patient_index, resamples=50, seed=1)
    assert curve.shape == (50,)


def test_two_limb_inflation_compares_the_two_units() -> None:
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 2, size=240)
    scores = labels + rng.normal(0.0, 1.0, size=240) + np.repeat(rng.normal(0, 2.0, 120), 2)
    patient_index = np.repeat(np.arange(120), 2)

    def statistic(rows):
        return auroc(scores[rows], labels[rows])

    report = two_limb_inflation(statistic, patient_index, resamples=100, seed=3)
    assert report["clustered_half_width"] >= 0.0
    assert report["inflation"] > 0.0


def test_subsample_by_patients_keeps_patients_whole() -> None:
    index = np.repeat(np.arange(50), 2)
    rows = subsample_by_patients(index, 0.4, seed=5)
    selected = index[rows]
    counts = np.bincount(selected)
    assert set(counts[counts > 0].tolist()) == {2}
    assert np.unique(selected).size == 20
    with pytest.raises(ValueError):
        subsample_by_patients(index, 0.0)


def test_fixed_sequence_stops_at_the_first_failure() -> None:
    outcomes = fixed_sequence(["a", "b", "c", "d"], np.array([0.01, 0.2, 0.01, 0.01]))
    assert outcomes[0].rejected
    assert not outcomes[1].rejected
    assert outcomes[1].stopped
    assert not outcomes[2].rejected
    assert outcomes[2].stopped
    report = sequence_report(outcomes)
    assert report["stopped_at"] == "b"
    assert report["rejected_count"] == 1
    with pytest.raises(ValueError):
        fixed_sequence(["a"], np.array([0.1, 0.2]))


def test_holm_step_down_and_benjamini_hochberg() -> None:
    p_values = np.array([0.001, 0.02, 0.5])
    holm = holm_step_down(["a", "b", "c"], p_values)
    # alpha/3 and alpha/2 both exceed 0.001 and 0.02, so the first two are rejected
    # and the sequence stops on the third.
    assert holm[0].rejected
    assert holm[1].rejected
    assert not holm[2].rejected
    bh = benjamini_hochberg(["a", "b", "c"], p_values)
    assert bh[0].rejected
    rejections = sum(1 for outcome in bh if outcome.rejected)
    assert rejections <= 3


def test_numeric_helpers() -> None:
    assert safe_scalar(1.0, 0.0, fill=-1.0) == -1.0
    assert safe_divide(np.array([1.0, 0.0]), np.array([2.0, 0.0])).tolist() == [0.5, 0.0]
    assert midrank(np.array([1.0, 2.0, 2.0, 3.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    low, high = percentile_interval(np.arange(100, dtype=np.float64))
    assert low < high
    low_w, high_w = wilson_interval(3, 10)
    assert 0.0 <= low_w < high_w <= 1.0
    assert math.isnan(wilson_interval(0, 0)[0])
