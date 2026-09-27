"""Discrimination metrics for the four-type assignment.

The primary endpoint is the macro-averaged area under the receiver operating
characteristic curve over the four types, each type scored one-versus-rest, with
ties handled by midranks. Per-type values are reported independently because the
rarest and hardest type carries the decision, and the agreement with the
two-reader consensus is reported as Cohen's kappa.

Ref: Table 3 and its caption; Sec. 6.7 (the primary criterion); Sec. 6.8.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from conductance_ctv.data.schema import confusion_counts
from conductance_ctv.utils.numeric import midrank, safe_scalar
from conductance_ctv.utils.types import FloatArray, IntArray


def auroc(scores: FloatArray, labels: IntArray) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic with midranks.

    Ties receive their average rank, so a score with many repeated values is
    neither credited nor penalised for them.
    """
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.shape[0] != targets.shape[0]:
        raise ValueError("scores and labels must agree on the length")
    positive = targets == 1
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(targets.shape[0] - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = midrank(values)
    rank_sum = float(np.sum(ranks[positive]))
    statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return statistic / (n_positive * n_negative)


def per_type_scores(probabilities: FloatArray, classes: int = 4) -> FloatArray:
    """One-versus-rest scores from a probability matrix."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != classes:
        raise ValueError(f"a probability matrix with {classes} columns is required")
    return values


def per_type_auroc(probabilities: FloatArray, labels: IntArray, classes: int = 4) -> FloatArray:
    """One-versus-rest AUC of each type, in type order."""
    values = per_type_scores(probabilities, classes)
    targets = np.asarray(labels, dtype=np.int64)
    out = np.full(classes, np.nan, dtype=np.float64)
    for index in range(classes):
        out[index] = auroc(values[:, index], (targets == index + 1).astype(np.int64))
    return out


def macro_auroc(probabilities: FloatArray, labels: IntArray, classes: int = 4) -> float:
    """Macro-averaged one-versus-rest AUC, the primary endpoint of Sec. 6.7."""
    values = per_type_auroc(probabilities, labels, classes)
    return float(np.nanmean(values))


def balanced_accuracy(predictions: IntArray, labels: IntArray, classes: int = 4) -> float:
    """Mean per-class recall, which is what an unbalanced four-type task needs."""
    guesses = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(labels, dtype=np.int64)
    recalls: list[float] = []
    for index in range(1, classes + 1):
        mask = targets == index
        if int(np.count_nonzero(mask)) == 0:
            continue
        recalls.append(safe_scalar(float(np.count_nonzero(guesses[mask] == index)), float(np.count_nonzero(mask))))
    return float(np.mean(recalls)) if recalls else float("nan")


def f1_per_class(predictions: IntArray, labels: IntArray, classes: int = 4) -> FloatArray:
    matrix = confusion_counts(np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64), classes)
    out = np.zeros(classes, dtype=np.float64)
    for index in range(classes):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        out[index] = safe_scalar(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative)
    return out


def macro_f1(predictions: IntArray, labels: IntArray, classes: int = 4) -> float:
    return float(np.mean(f1_per_class(predictions, labels, classes)))


def cohen_kappa(predictions: IntArray, labels: IntArray, classes: int = 4) -> float:
    """Cohen's kappa between a predicted call and a reference call."""
    matrix = confusion_counts(np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64), classes)
    total = float(matrix.sum())
    if total <= 0.0:
        return float("nan")
    observed = float(np.trace(matrix)) / total
    row = matrix.sum(axis=1) / total
    column = matrix.sum(axis=0) / total
    expected = float(np.sum(row * column))
    return safe_scalar(observed - expected, 1.0 - expected, 0.0)


def kappa_standard_error(predictions: IntArray, labels: IntArray, classes: int = 4) -> float:
    """Asymptotic standard error of Cohen's kappa under the multinomial model."""
    matrix = confusion_counts(np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64), classes)
    total = float(matrix.sum())
    if total <= 0.0:
        return float("nan")
    proportions = matrix / total
    row = proportions.sum(axis=1)
    column = proportions.sum(axis=0)
    agreement = float(np.trace(proportions))
    expected = float(np.sum(row * column))
    if abs(1.0 - expected) < 1e-12:
        return float("nan")
    term_a = agreement * (1.0 - agreement) / ((1.0 - expected) ** 2)
    term_b = (
        2.0
        * (1.0 - agreement)
        * (float(np.sum(proportions * np.outer(row, column))) - agreement * expected)
        / ((1.0 - expected) ** 3)
    )
    term_c = (
        ((1.0 - agreement) ** 2)
        * (float(np.sum(np.outer(row, column).T * np.outer(row, column))) - expected**2)
        / ((1.0 - expected) ** 4)
    )
    variance = (term_a + term_b - term_c) / total
    return float(math.sqrt(max(variance, 0.0)))


def kappa_interval(
    predictions: IntArray, labels: IntArray, classes: int = 4, z: float = 1.959963984540054
) -> tuple[float, float]:
    value = cohen_kappa(predictions, labels, classes)
    error = kappa_standard_error(predictions, labels, classes)
    if not np.isfinite(value) or not np.isfinite(error):
        return (float("nan"), float("nan"))
    return (value - z * error, value + z * error)


def sensitivity_specificity(calls: IntArray, truth: IntArray) -> tuple[float, float]:
    """Sensitivity and specificity of a binary call."""
    predicted = np.asarray(calls, dtype=np.int64)
    actual = np.asarray(truth, dtype=np.int64)
    positive = actual == 1
    negative = ~positive
    sensitivity = safe_scalar(float(np.count_nonzero(predicted[positive] == 1)), float(np.count_nonzero(positive)))
    specificity = safe_scalar(float(np.count_nonzero(predicted[negative] == 0)), float(np.count_nonzero(negative)))
    return sensitivity, specificity


def calibration_slope(probabilities: FloatArray, labels: IntArray, iterations: int = 100) -> float:
    """Slope of the calibration line of the management-relevant call.

    The call is post-thrombotic or occlusive against the rest, its predicted risk
    is the sum of those two probabilities, and the slope comes from a logistic
    regression of the observed call on the logit of the predicted risk. A slope
    of one means the probabilities need no further stretching; below one means
    they are over-confident.

    Ref: Table 4 (calibration slope row) and Table 6.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    risk = np.clip(values[:, 2] + values[:, 3], 1e-6, 1.0 - 1e-6)
    observed = (targets >= 3).astype(np.float64)
    predictor = np.log(risk / (1.0 - risk))
    design = np.stack([np.ones_like(predictor), predictor], axis=1)
    weights = np.zeros(2, dtype=np.float64)
    for _ in range(iterations):
        linear = np.clip(design @ weights, -60.0, 60.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (observed - fitted)
        variance = fitted * (1.0 - fitted)
        hessian = design.T @ (design * variance[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        weights = weights + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(weights[1])


def brier_score(probabilities: FloatArray, labels: IntArray, classes: int = 4) -> float:
    """Multi-class Brier score against the one-hot reference."""
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros_like(values)
    one_hot[np.arange(targets.shape[0]), np.clip(targets - 1, 0, classes - 1)] = 1.0
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def expected_calibration_error(probabilities: FloatArray, labels: IntArray, bins: int = 10) -> float:
    """Top-label expected calibration error over equal-width confidence bins."""
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1) + 1
    correct = (predicted == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(values.shape[0])
    error = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = float(np.count_nonzero(mask))
        if count == 0.0:
            continue
        error += count / total * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(error)


def confusion_table(predictions: IntArray, labels: IntArray, classes: int = 4) -> list[dict[str, float | int]]:
    matrix = confusion_counts(np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64), classes)
    return [
        {"reference": row + 1, "predicted": column + 1, "count": int(matrix[row, column])}
        for row in range(classes)
        for column in range(classes)
    ]


def high_risk_flag(probabilities: FloatArray, threshold: float) -> IntArray:
    """The management-relevant call: post-thrombotic or occlusive."""
    values = np.asarray(probabilities, dtype=np.float64)
    return (values[:, 2] + values[:, 3] >= threshold).astype(np.int64)


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation of two sequences, ties sharing a midrank.

    Used as an internal agreement diagnostic: it says whether two profiles order
    their samples the same way even when their scales differ, which is what a
    discretised measurement of a closed-form profile should do.
    """
    left = midrank(np.asarray(first, dtype=np.float64))
    right = midrank(np.asarray(second, dtype=np.float64))
    if left.shape[0] < 2:
        return float("nan")
    left_centred = left - left.mean()
    right_centred = right - right.mean()
    denominator = float(np.sqrt(np.sum(left_centred**2) * np.sum(right_centred**2)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(left_centred * right_centred) / denominator)
