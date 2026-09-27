"""Correlated-comparison and association tests.

Four inferential procedures are used by the reported tables: the DeLong test for
two correlated areas under the curve on the same cases, the exact or asymptotic
McNemar test for paired calls, a likelihood-ratio test for the association models,
and the continuous net reclassification improvement with a decision-curve net
benefit. Two implementations of DeLong are provided: a rank-based one used in
production and a direct one over case pairs used to check it.

Ref: Sec. 6.8 (DeLong, McNemar, likelihood-ratio test, net reclassification and
decision-curve analysis); Table 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.metrics.special import chi2_survival, exact_binomial_two_sided, two_sided_normal_p
from conductance_ctv.utils.numeric import midrank, safe_scalar
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray


def _midrank(values: FloatArray) -> FloatArray:
    return midrank(values)


@dataclass(frozen=True)
class DeLongResult:
    """Difference of two correlated areas with its interval and p-value."""

    auc_first: float
    auc_second: float
    difference: float
    standard_error: float
    z: float
    p_value: float
    lower: float
    upper: float

    def record(self) -> dict[str, float]:
        return {
            "auc_first": self.auc_first,
            "auc_second": self.auc_second,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "z": self.z,
            "p_value": self.p_value,
            "ci_low": self.lower,
            "ci_high": self.upper,
        }


def _placements(scores: FloatArray, positive: BoolArray) -> tuple[FloatArray, FloatArray]:
    """Placement values of the DeLong covariance, one per case in each group.

    Each placement is the case's rank in the pooled sample with its rank inside
    its own group removed, so it measures how far the case sits into the other
    group. Using the within-group midrank rather than the case's position in the
    group is what makes the two placement vectors centred.
    """
    positive_values = scores[positive]
    negative_values = scores[~positive]
    pooled_ranks = _midrank(np.concatenate([positive_values, negative_values]))
    n_positive = positive_values.shape[0]
    n_negative = negative_values.shape[0]
    positive_ranks = pooled_ranks[:n_positive]
    negative_ranks = pooled_ranks[n_positive:]
    v10 = (positive_ranks - _midrank(positive_values)) / n_negative
    v01 = 1.0 - (negative_ranks - _midrank(negative_values)) / n_positive
    return v10, v01


def delong_auc_pair(first: FloatArray, second: FloatArray, labels: IntArray, alpha: float = 0.05) -> DeLongResult:
    """Compare two correlated areas under the curve with the DeLong covariance.

    Both classifiers are scored on the same cases, so their placements share the
    case-level randomness and the covariance of the two placement vectors has to
    enter the variance of the difference.
    """
    scores_first = np.asarray(first, dtype=np.float64)
    scores_second = np.asarray(second, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores_first.shape != scores_second.shape or scores_first.shape[0] != targets.shape[0]:
        raise ValueError("the two score vectors and the labels must have equal length")
    positive = targets == 1
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(targets.shape[0] - n_positive)
    if n_positive < 2 or n_negative < 2:
        return DeLongResult(*(float("nan"),) * 8)

    v10_first, v01_first = _placements(scores_first, positive)
    v10_second, v01_second = _placements(scores_second, positive)
    auc_first = float(v10_first.mean())
    auc_second = float(v10_second.mean())

    s10 = np.cov(np.stack([v10_first, v10_second])) / n_positive
    s01 = np.cov(np.stack([v01_first, v01_second])) / n_negative
    covariance = s10 + s01
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    standard_error = float(np.sqrt(max(variance, 0.0)))
    difference = auc_first - auc_second
    z = safe_scalar(difference, standard_error, 0.0)
    p_value = two_sided_normal_p(z)
    half = _normal_quantile(1.0 - alpha / 2.0) * standard_error
    return DeLongResult(
        auc_first=auc_first,
        auc_second=auc_second,
        difference=difference,
        standard_error=standard_error,
        z=z,
        p_value=p_value,
        lower=difference - half,
        upper=difference + half,
    )


def _normal_quantile(probability: float) -> float:
    from conductance_ctv.utils.numeric import normal_ppf

    return normal_ppf(probability)


def delong_auc_pair_direct(
    first: FloatArray, second: FloatArray, labels: IntArray, alpha: float = 0.05
) -> DeLongResult:
    """DeLong computed straight from the definition, one pair of cases at a time.

    This is the form the test suite compares the rank-based implementation
    against; it is quadratic in the sample size and is not used in the pipeline.
    """
    scores_first = np.asarray(first, dtype=np.float64)
    scores_second = np.asarray(second, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    if positive.size < 2 or negative.size < 2:
        return DeLongResult(*(float("nan"),) * 8)
    first_grid = scores_first[positive][:, None] - scores_first[negative][None, :]
    second_grid = scores_second[positive][:, None] - scores_second[negative][None, :]
    first_psi = np.where(first_grid > 0, 1.0, np.where(first_grid == 0, 0.5, 0.0))
    second_psi = np.where(second_grid > 0, 1.0, np.where(second_grid == 0, 0.5, 0.0))
    auc_first = float(first_psi.mean())
    auc_second = float(second_psi.mean())
    v10_first = first_psi.mean(axis=1)
    v10_second = second_psi.mean(axis=1)
    v01_first = first_psi.mean(axis=0)
    v01_second = second_psi.mean(axis=0)
    s10 = np.cov(np.stack([v10_first, v10_second])) / positive.size
    s01 = np.cov(np.stack([v01_first, v01_second])) / negative.size
    covariance = s10 + s01
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    standard_error = float(np.sqrt(max(variance, 0.0)))
    difference = auc_first - auc_second
    z = safe_scalar(difference, standard_error, 0.0)
    half = _normal_quantile(1.0 - alpha / 2.0) * standard_error
    return DeLongResult(
        auc_first=auc_first,
        auc_second=auc_second,
        difference=difference,
        standard_error=standard_error,
        z=z,
        p_value=two_sided_normal_p(z),
        lower=difference - half,
        upper=difference + half,
    )


@dataclass(frozen=True)
class McNemarResult:
    """Discordant-pair counts and the paired test of two call sets."""

    first_only: int
    second_only: int
    statistic: float
    p_value: float
    exact: bool

    def record(self) -> dict[str, float]:
        return {
            "first_only": float(self.first_only),
            "second_only": float(self.second_only),
            "statistic": self.statistic,
            "p_value": self.p_value,
            "exact": float(self.exact),
        }


def mcnemar(first_correct: IntArray, second_correct: IntArray, exact_below: int = 25) -> McNemarResult:
    """McNemar test on two call sets, with the exact form for few discordant pairs."""
    first = np.asarray(first_correct, dtype=np.int64)
    second = np.asarray(second_correct, dtype=np.int64)
    if first.shape != second.shape:
        raise ValueError("the two call vectors must have equal length")
    first_only = int(np.count_nonzero((first == 1) & (second == 0)))
    second_only = int(np.count_nonzero((first == 0) & (second == 1)))
    discordant = first_only + second_only
    if discordant == 0:
        return McNemarResult(first_only, second_only, 0.0, 1.0, discordant < exact_below)
    if discordant < exact_below:
        p_value = exact_binomial_two_sided(min(first_only, second_only), discordant, 0.5)
        return McNemarResult(first_only, second_only, 0.0, float(p_value), True)
    statistic = (abs(first_only - second_only) - 1.0) ** 2 / discordant
    return McNemarResult(
        first_only,
        second_only,
        float(statistic),
        chi2_survival(float(statistic), 1),
        False,
    )


def likelihood_ratio_test(
    log_likelihood_full: float, log_likelihood_reduced: float, added_parameters: int
) -> dict[str, float]:
    """Likelihood-ratio test of a nested pair, as a chi-square survival function.

    Ref: Table 7 P (LR) column.
    """
    if added_parameters < 1:
        raise ValueError("a nested comparison adds at least one parameter")
    statistic = 2.0 * (float(log_likelihood_full) - float(log_likelihood_reduced))
    statistic = max(statistic, 0.0)
    return {
        "statistic": statistic,
        "degrees_of_freedom": float(added_parameters),
        "p_value": chi2_survival(statistic, added_parameters),
    }


def logistic_log_likelihood(design: FloatArray, coefficients: FloatArray, labels: IntArray) -> float:
    """Log-likelihood of a logistic model, used by the association comparison."""
    features = np.asarray(design, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    linear = features @ weights
    return float(np.sum(targets * linear - np.logaddexp(0.0, linear)))


def continuous_nri(probability_new: FloatArray, probability_old: FloatArray, labels: IntArray) -> dict[str, float]:
    """Continuous net reclassification improvement.

    For events the improvement counts upward movement, for non-events downward
    movement, and the statistic is the difference of the two net proportions.

    Ref: Table 7 and the caption of the association table.
    """
    new = np.asarray(probability_new, dtype=np.float64)
    old = np.asarray(probability_old, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if new.shape != old.shape or new.shape[0] != targets.shape[0]:
        raise ValueError("the two probability vectors and the labels must have equal length")
    delta = new - old
    events = targets == 1
    non_events = ~events
    if not bool(events.any()) or not bool(non_events.any()):
        return {"nri": float("nan"), "events": float("nan"), "non_events": float("nan")}
    event_component = float(np.mean(delta[events] > 0) - np.mean(delta[events] < 0))
    non_event_component = float(np.mean(delta[non_events] < 0) - np.mean(delta[non_events] > 0))
    return {
        "nri": event_component + non_event_component,
        "events": event_component,
        "non_events": non_event_component,
    }


def decision_curve_net_benefit(calls: IntArray, labels: IntArray, threshold: float) -> float:
    """Net benefit of a decision rule at a pre-specified threshold probability.

    ``net benefit = TP / N - FP / N * pt / (1 - pt)``.
    """
    predicted = np.asarray(calls, dtype=np.int64)
    targets = np.asarray(labels, dtype=np.int64)
    if not 0.0 < threshold < 1.0:
        raise ValueError("the threshold probability must lie strictly inside (0, 1)")
    total = float(predicted.shape[0])
    if total == 0.0:
        return float("nan")
    true_positive = float(np.count_nonzero((predicted == 1) & (targets == 1)))
    false_positive = float(np.count_nonzero((predicted == 1) & (targets == 0)))
    return true_positive / total - false_positive / total * (threshold / (1.0 - threshold))


def net_benefit_gain(calls_new: IntArray, calls_old: IntArray, labels: IntArray, threshold: float) -> float:
    """Difference in net benefit between two decision rules at one threshold."""
    return decision_curve_net_benefit(calls_new, labels, threshold) - decision_curve_net_benefit(
        calls_old, labels, threshold
    )


def decision_curve_curve(
    calls: IntArray, labels: IntArray, thresholds: FloatArray | None = None
) -> dict[str, FloatArray]:
    """Net benefit across a grid of thresholds, for the decision-curve figure."""
    grid = (
        np.linspace(0.05, 0.95, 19, dtype=np.float64)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64)
    )
    values = np.array([decision_curve_net_benefit(calls, labels, float(value)) for value in grid])
    targets = np.asarray(labels, dtype=np.int64)
    prevalence = float(np.mean(targets))
    treated_all = prevalence - (1.0 - prevalence) * (grid / (1.0 - grid))
    return {"threshold": grid, "model": values, "treat_all": treated_all, "treat_none": np.zeros_like(grid)}


def adjusted_odds_ratio(
    design: FloatArray, labels: IntArray, column: int, iterations: int = 100, ridge: float = 1e-6
) -> dict[str, float]:
    """Adjusted odds ratio and Wald interval for one column of a logistic design.

    Fitted by Newton-Raphson on the log-likelihood, which is the estimator the
    paper's association table is built on; a small ridge keeps the Hessian
    invertible when a column is nearly collinear.
    """
    features = np.asarray(design, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] <= column:
        raise ValueError("the requested column is outside the design matrix")
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(iterations):
        linear = features @ weights
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -60.0, 60.0)))
        gradient = features.T @ (targets - probability) - ridge * weights
        variance = probability * (1.0 - probability)
        hessian = features.T @ (features * variance[:, None]) + ridge * np.eye(features.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights = weights + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    linear = features @ weights
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -60.0, 60.0)))
    variance = probability * (1.0 - probability)
    hessian = features.T @ (features * variance[:, None]) + ridge * np.eye(features.shape[1])
    covariance = np.linalg.inv(hessian)
    standard_error = float(np.sqrt(max(covariance[column, column], 0.0)))
    coefficient = float(weights[column])
    z = safe_scalar(coefficient, standard_error, 0.0)
    half = _normal_quantile(0.975) * standard_error
    return {
        "odds_ratio": float(np.exp(coefficient)),
        "ci_low": float(np.exp(coefficient - half)),
        "ci_high": float(np.exp(coefficient + half)),
        "z": z,
        "p_value": two_sided_normal_p(z),
        "coefficient": coefficient,
        "standard_error": standard_error,
    }


def log_likelihood_of_design(design: FloatArray, labels: IntArray, ridge: float = 1e-6) -> float:
    """Log-likelihood of the fitted logistic model, for the likelihood-ratio test."""
    features = np.asarray(design, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(100):
        linear = features @ weights
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -60.0, 60.0)))
        gradient = features.T @ (targets - probability) - ridge * weights
        variance = probability * (1.0 - probability)
        hessian = features.T @ (features * variance[:, None]) + ridge * np.eye(features.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights = weights + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return logistic_log_likelihood(features, weights, np.asarray(labels, dtype=np.int64))


def design_matrix(columns: dict[str, FloatArray], names: tuple[str, ...], intercept: bool = True) -> FloatArray:
    """Assemble a design matrix from named columns, with an intercept column first."""
    parts: list[FloatArray] = []
    if intercept:
        length = len(next(iter(columns.values()))) if columns else 0
        parts.append(np.ones(length, dtype=np.float64))
    for name in names:
        if name not in columns:
            raise KeyError(f"missing design column {name!r}")
        parts.append(np.asarray(columns[name], dtype=np.float64))
    return np.stack(parts, axis=1)
