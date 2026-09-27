"""Agreement between an automated measurement and a manual one.

Agreement is reported as an intraclass correlation of the two-way random-effects
absolute-agreement single-measurement form, ICC(2, 1), together with the
Bland-Altman bias and the limits of agreement. The absolute-area descriptors and
the dimensionless ones are reported on the same scale so the two families can be
compared across sites.

Ref: Sec. 3.2 and Table 5; Sec. 6.8 (ICC(2, 1) and Bland-Altman).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.utils.numeric import safe_scalar
from conductance_ctv.utils.types import FloatArray

_LOA_Z = 1.959963984540054


@dataclass(frozen=True)
class ICCResult:
    """ICC(2, 1) with the mean squares it was formed from."""

    value: float
    between_subjects: float
    between_raters: float
    residual: float
    subjects: int
    raters: int

    def record(self) -> dict[str, float]:
        return {
            "icc": self.value,
            "ms_between_subjects": self.between_subjects,
            "ms_between_raters": self.between_raters,
            "ms_residual": self.residual,
            "subjects": float(self.subjects),
            "raters": float(self.raters),
        }


def icc_2_1(ratings: FloatArray) -> ICCResult:
    """Two-way random-effects, absolute-agreement, single-measurement ICC.

    ``ratings`` is indexed ``[subject, rater]``. The estimator is

        (MSR - MSE) / (MSR + (k - 1) MSE + k (MSC - MSE) / n)

    with ``MSR`` the between-subject mean square, ``MSC`` the between-rater mean
    square and ``MSE`` the residual mean square.
    """
    values = np.asarray(ratings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("ratings must have shape (subjects, raters)")
    n_subjects, n_raters = values.shape
    if n_subjects < 2 or n_raters < 2:
        raise ValueError("ICC(2, 1) needs at least two subjects and two raters")
    if not np.all(np.isfinite(values)):
        raise ValueError("ICC(2, 1) needs a complete ratings matrix")
    grand = float(values.mean())
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    between_subjects = float(n_raters * np.sum((row_means - grand) ** 2) / (n_subjects - 1))
    between_raters = float(n_subjects * np.sum((column_means - grand) ** 2) / (n_raters - 1))
    total = float(np.sum((values - grand) ** 2))
    residual = float(
        (total - n_raters * np.sum((row_means - grand) ** 2) - n_subjects * np.sum((column_means - grand) ** 2))
        / ((n_subjects - 1) * (n_raters - 1))
    )
    denominator = between_subjects + (n_raters - 1) * residual + n_raters * (between_raters - residual) / n_subjects
    value = safe_scalar(between_subjects - residual, denominator, float("nan"))
    return ICCResult(
        value=float(value),
        between_subjects=between_subjects,
        between_raters=between_raters,
        residual=residual,
        subjects=int(n_subjects),
        raters=int(n_raters),
    )


def concordance_correlation(ratings: FloatArray) -> float:
    """Lin's concordance correlation coefficient, a scale-sensitive alternative."""
    values = np.asarray(ratings, dtype=np.float64)
    if values.shape[1] != 2:
        raise ValueError("the concordance correlation is defined for two raters")
    first = values[:, 0]
    second = values[:, 1]
    mean_first = float(first.mean())
    mean_second = float(second.mean())
    covariance = float(np.mean((first - mean_first) * (second - mean_second)))
    var_first = float(np.mean((first - mean_first) ** 2))
    var_second = float(np.mean((second - mean_second) ** 2))
    denominator = var_first + var_second + (mean_first - mean_second) ** 2
    return safe_scalar(2.0 * covariance, denominator, 0.0)


@dataclass(frozen=True)
class BlandAltman:
    """Bias and limits of agreement of a paired measurement."""

    bias: float
    standard_deviation: float
    lower: float
    upper: float
    samples: int
    crosses_zero: bool = False

    def record(self) -> dict[str, float]:
        return {
            "bias": self.bias,
            "sd": self.standard_deviation,
            "loa_low": self.lower,
            "loa_high": self.upper,
            "samples": float(self.samples),
        }


def bland_altman(automated: FloatArray, manual: FloatArray, paired: bool = True) -> BlandAltman:
    """Bias and 95% limits of agreement between two measurement series.

    ``paired`` uses the differences of matched pairs, which is the form the
    manuscript reports; the unpaired form compares the two series' means and
    pooled standard deviation instead.
    """
    first = np.asarray(automated, dtype=np.float64)
    second = np.asarray(manual, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("the two series must have the same length")
    if first.size == 0:
        return BlandAltman(float("nan"), float("nan"), float("nan"), float("nan"), 0, False)
    if paired:
        difference = first - second
        bias = float(difference.mean())
        deviation = float(difference.std(ddof=1)) if difference.size > 1 else 0.0
    else:
        bias = float(first.mean() - second.mean())
        deviation = float(np.sqrt(0.5 * (first.var(ddof=1) + second.var(ddof=1)))) if first.size > 1 else 0.0
    lower = bias - _LOA_Z * deviation
    upper = bias + _LOA_Z * deviation
    return BlandAltman(
        bias=bias,
        standard_deviation=deviation,
        lower=lower,
        upper=upper,
        samples=int(first.size),
        crosses_zero=bool(lower <= 0.0 <= upper),
    )


def agreement_table(automated: dict[str, FloatArray], manual: dict[str, FloatArray]) -> dict[str, dict[str, float]]:
    """ICC and Bland-Altman of every descriptor present in both dictionaries."""
    out: dict[str, dict[str, float]] = {}
    for name in automated:
        if name not in manual:
            continue
        first = np.asarray(automated[name], dtype=np.float64)
        second = np.asarray(manual[name], dtype=np.float64)
        if first.shape != second.shape or first.size < 2:
            continue
        matrix = np.stack([first, second], axis=1)
        icc = icc_2_1(matrix)
        limits = bland_altman(first, second)
        out[name] = {**icc.record(), **limits.record()}
    return out


def relative_degradation(development: float, external: float) -> float:
    """Relative change of a metric between the development and external sites.

    Ref: Table 4 change column.
    """
    if not np.isfinite(development) or abs(development) < 1e-12:
        return float("nan")
    return float(abs(external - development) / abs(development))


def family_of(descriptor: str) -> str:
    """Whether a descriptor is dimensionless or length-scaled.

    The dimensionless families are the two ratios; every other reported
    descriptor carries a length scale and is expected to degrade on transport.

    Ref: Sec. 3.3 and Table 4.
    """
    dimensionless = {"resistance_ratio", "shape_factor"}
    return "dimensionless" if descriptor in dimensionless else "absolute"
