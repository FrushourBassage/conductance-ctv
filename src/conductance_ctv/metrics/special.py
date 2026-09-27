"""Distribution functions the inference tests need.

Implemented here rather than imported so the release carries one dependency set
and so the tail probabilities used in the significance tests are readable next to
the tests that use them: a likelihood-ratio statistic is a chi-square survival
function, and a DeLong statistic is a standard-normal one.

Ref: Sec. 6.8 (DeLong, McNemar, likelihood-ratio testing).
"""

from __future__ import annotations

import math

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def log_gamma(value: float) -> float:
    return math.lgamma(value)


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution."""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def normal_sf(value: float) -> float:
    """Standard normal survival function."""
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def two_sided_normal_p(z_statistic: float) -> float:
    return float(min(1.0, 2.0 * normal_sf(abs(z_statistic))))


def regularized_gamma_q(shape: float, value: float) -> float:
    """Regularised upper incomplete gamma ``Q(shape, value)``.

    A continued fraction is used above the shape and a series below it, which is
    the split that keeps both branches convergent.
    """
    if shape <= 0.0:
        raise ValueError("the gamma shape must be positive")
    if value < 0.0:
        raise ValueError("the gamma argument must be non-negative")
    if value == 0.0:
        return 1.0
    if value < shape + 1.0:
        return 1.0 - _gamma_p_series(shape, value)
    return _gamma_q_fraction(shape, value)


def _gamma_p_series(shape: float, value: float) -> float:
    term = 1.0 / shape
    total = term
    denominator = shape
    for _ in range(1000):
        denominator += 1.0
        term *= value / denominator
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-value + shape * math.log(value) - log_gamma(shape))


def _gamma_q_fraction(shape: float, value: float) -> float:
    tiny = 1e-300
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 1000):
        an = -index * (index - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-value + shape * math.log(value) - log_gamma(shape)) * h


def chi2_survival(statistic: float, degrees: int) -> float:
    """Survival function of a chi-square variable with integer degrees of freedom."""
    if degrees < 1:
        raise ValueError("the degrees of freedom must be at least one")
    if statistic <= 0.0:
        return 1.0
    return float(regularized_gamma_q(0.5 * degrees, 0.5 * statistic))


def chi2_p_value(statistic: float, degrees: int) -> float:
    return chi2_survival(statistic, degrees)


def log_binomial_coefficient(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return log_gamma(n + 1) - log_gamma(k + 1) - log_gamma(n - k + 1)


def exact_binomial_two_sided(k: int, n: int, probability: float = 0.5) -> float:
    """Exact two-sided binomial p-value by summing the observed and rarer tails."""
    if n <= 0:
        return 1.0
    observed = math.exp(log_binomial_coefficient(n, k) + k * math.log(probability) + (n - k) * math.log1p(-probability))
    total = 0.0
    for value in range(n + 1):
        term = math.exp(
            log_binomial_coefficient(n, value) + value * math.log(probability) + (n - value) * math.log1p(-probability)
        )
        if term <= observed * (1.0 + 1e-12):
            total += term
    return float(min(1.0, total))


def normal_log_pdf(value: float) -> float:
    return -0.5 * value * value - _LOG_SQRT_2PI
