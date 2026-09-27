"""Patient-clustered resampling for every reported interval.

The resampling unit is the patient, not the limb and not the segment: a patient's
two limbs are correlated and so are the three segments of one limb, so an
interval built by resampling limbs independently would be too narrow. Every
resample draws patients with replacement and propagates the draw to their limbs,
exactly as Algorithm 3 describes.

Ref: Sec. 6.8 ("bootstrap confidence intervals for patient clusters in a way that
the dependency between the two limbs of a patient as well as different sections
of the limb is taken into account"); Algorithm 3 lines 3-6.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from conductance_ctv.utils.numeric import percentile_interval
from conductance_ctv.utils.types import FloatArray, IntArray

BOOTSTRAP_RESAMPLES = 2000


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its percentile interval and the resampling unit."""

    estimate: float
    lower: float
    upper: float
    resamples: int
    unit: str
    n_units: int

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def record(self) -> dict[str, float | str]:
        return {
            "estimate": self.estimate,
            "ci_low": self.lower,
            "ci_high": self.upper,
            "half_width": self.half_width,
            "resamples": float(self.resamples),
            "unit": self.unit,
            "n_units": float(self.n_units),
        }


def patient_clusters(patient_index: IntArray) -> list[IntArray]:
    """Row indices of every patient, in first-appearance order."""
    values = np.asarray(patient_index, dtype=np.int64)
    clusters: list[IntArray] = []
    for patient in dict.fromkeys(values.tolist()):
        clusters.append(np.flatnonzero(values == patient).astype(np.int64))
    return clusters


def clustered_resample_indices(clusters: list[IntArray], rng: np.random.Generator) -> IntArray:
    """Draw patients with replacement and concatenate their rows."""
    order = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[int(index)] for index in order])


def clustered_bootstrap(
    statistic: Callable[[IntArray], float],
    patient_index: IntArray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
    unit: str = "patient",
) -> BootstrapResult:
    """Percentile interval of a statistic under patient-clustered resampling."""
    if resamples < 1:
        raise ValueError("at least one resample is required")
    clusters = patient_clusters(patient_index)
    if not clusters:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, unit, 0)
    every_row = np.arange(int(np.asarray(patient_index).shape[0]), dtype=np.int64)
    estimate = float(statistic(every_row))
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        rows = clustered_resample_indices(clusters, rng)
        samples[index] = float(statistic(rows))
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return BootstrapResult(estimate, float("nan"), float("nan"), resamples, unit, len(clusters))
    low, high = percentile_interval(samples, level)
    return BootstrapResult(estimate, low, high, resamples, unit, len(clusters))


def paired_clustered_bootstrap(
    statistic: Callable[[IntArray], float],
    patient_index: IntArray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
    unit: str = "patient",
) -> BootstrapResult:
    """Clustered bootstrap of a paired difference, which is the same draw for both arms.

    The two arms of a paired comparison are evaluated on the identical resampled
    rows, so their difference carries the correlation and its interval is
    narrower than an interval built from two independent resamples.
    """
    return clustered_bootstrap(statistic, patient_index, resamples, level, seed, unit)


def bootstrap_curve(
    statistic: Callable[[IntArray], float],
    patient_index: IntArray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> FloatArray:
    """The resampled statistics themselves, for a figure or a further transform."""
    clusters = patient_clusters(patient_index)
    rng = np.random.default_rng(seed)
    out = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        out[index] = float(statistic(clustered_resample_indices(clusters, rng)))
    return out


def two_limb_inflation(
    statistic: Callable[[IntArray], float], patient_index: IntArray, resamples: int = 400, seed: int = 0
) -> dict[str, float]:
    """Compare the clustered interval with a limb-level one.

    Reporting the ratio makes explicit how much of the interval comes from the
    correlation between the two limbs of a patient; a ratio near one means the
    cohort has few two-limbed patients.

    Ref: Sec. 6.2 and the analysis-unit paragraph of Sec. 6.8.
    """
    rows = np.arange(int(np.asarray(patient_index).shape[0]), dtype=np.int64)
    clustered = clustered_bootstrap(statistic, patient_index, resamples, seed=seed)
    rng = np.random.default_rng(seed + 1)
    limb_values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw = rng.integers(0, rows.shape[0], size=rows.shape[0])
        limb_values[index] = float(statistic(rows[draw]))
    limb_values = limb_values[np.isfinite(limb_values)]
    limb_low, limb_high = percentile_interval(limb_values, 0.95)
    limb_width = 0.5 * (limb_high - limb_low)
    return {
        "clustered_half_width": clustered.half_width,
        "limb_half_width": float(limb_width),
        "inflation": float(clustered.half_width / limb_width) if limb_width > 1e-12 else float("nan"),
    }


def subsample_by_patients(patient_index: IntArray, fraction: float, seed: int = 0) -> IntArray:
    """A patient-level subsample, used by the data-efficiency ablations.

    Ref: Table 2 (initialisation at 10%, 25% and 100% of the corpus).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("the fraction must lie in (0, 1]")
    clusters = patient_clusters(patient_index)
    count = max(1, int(round(fraction * len(clusters))))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(clusters), size=count, replace=False)
    return np.sort(np.concatenate([clusters[int(index)] for index in chosen]))
