"""Small numerical helpers used by the metric and hydraulic modules.

Ref: Sec. 6.8 (statistical analysis) and Sec. 6.5 (descriptor arithmetic).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from conductance_ctv.utils.types import FloatArray

EPS = 1e-12


def as_triple(values: Sequence[float]) -> tuple[float, float, float]:
    """Three floats in a fixed-length tuple, checked where the tuple is built.

    Volumes and spacings are three-axis throughout, so the length is a boundary
    condition rather than something every consumer should re-test.
    """
    items = [float(value) for value in values]
    if len(items) != 3:
        raise ValueError(f"expected three values, got {len(items)}")
    return items[0], items[1], items[2]


def as_quad(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Four floats in a fixed-length tuple, checked where the tuple is built.

    The four-type prevalence vector is the other fixed-length quantity the
    pipeline passes between modules, so it gets the same boundary check.
    """
    items = [float(value) for value in values]
    if len(items) != 4:
        raise ValueError(f"expected four values, got {len(items)}")
    return items[0], items[1], items[2], items[3]


def as_triple_int(values: Sequence[int]) -> tuple[int, int, int]:
    """Three ints in a fixed-length tuple, checked where the tuple is built."""
    items = [int(value) for value in values]
    if len(items) != 3:
        raise ValueError(f"expected three values, got {len(items)}")
    return items[0], items[1], items[2]


def safe_divide(numerator: FloatArray, denominator: FloatArray, fill: float = 0.0) -> FloatArray:
    """Elementwise division that maps |denominator| <= EPS to ``fill``."""
    out = np.full_like(np.asarray(numerator, dtype=np.float64), fill, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=np.abs(denominator) > EPS)
    return out


def safe_scalar(numerator: float, denominator: float, fill: float = 0.0) -> float:
    if abs(denominator) <= EPS:
        return fill
    return float(numerator) / float(denominator)


def midrank(values: FloatArray) -> FloatArray:
    """Average ranks with ties sharing the mean rank (1-based)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < sorted_values.shape[0]:
        stop = index + 1
        while stop < sorted_values.shape[0] and sorted_values[stop] == sorted_values[index]:
            stop += 1
        ranks[order[index:stop]] = 0.5 * (index + stop - 1) + 1.0
        index = stop
    return ranks


def percentile_interval(samples: FloatArray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, used for every clustered bootstrap in the paper."""
    tail = (1.0 - level) / 2.0
    low = float(np.quantile(samples, tail))
    high = float(np.quantile(samples, 1.0 - tail))
    return low, high


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def normal_ppf(probability: float) -> float:
    """Inverse standard normal CDF via the Acklam rational approximation."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly inside (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1 - plow
    if probability < plow:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > phigh:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
