"""Intensity and geometry transforms for the segmentation and type inputs.

The augmentation budget is deliberately small: a venous-phase acquisition is
already anisotropic and the appearance branch is not supposed to encode an
appearance signal that a change of scanner would move, so the transforms that
survive are the ones a different scanner or contrast protocol could plausibly
produce.

Ref: Sec. 3.3 (transport of the dimensionless descriptors while absolute and
appearance quantities move) and Sec. 6.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from conductance_ctv.utils.types import BoolArray, FloatArray

HU_WINDOW_LOW = -120.0
HU_WINDOW_HIGH = 520.0


def window_intensity(volume: FloatArray, low: float = HU_WINDOW_LOW, high: float = HU_WINDOW_HIGH) -> FloatArray:
    """Clip to a venous window and rescale to ``[0, 1]``."""
    if high <= low:
        raise ValueError("the window must have positive width")
    return np.clip((np.asarray(volume, dtype=np.float64) - low) / (high - low), 0.0, 1.0)


def unwindow_intensity(scaled: FloatArray, low: float = HU_WINDOW_LOW, high: float = HU_WINDOW_HIGH) -> FloatArray:
    """Inverse of :func:`window_intensity`, used by the report writers."""
    return np.asarray(scaled, dtype=np.float64) * (high - low) + low


def normalise_robust(volume: FloatArray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> FloatArray:
    """Percentile normalisation, which is what makes a vendor gain difference a nuisance."""
    values = np.asarray(volume, dtype=np.float64)
    if values.size == 0:
        return values
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high - low < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def random_flip_axes(volume: FloatArray, axes: tuple[int, ...], rng: np.random.Generator) -> FloatArray:
    out = np.array(volume, dtype=np.float64)
    for axis in axes:
        if float(rng.random()) < 0.5:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def random_contrast_phase(
    volume: FloatArray, rng: np.random.Generator, gain_range: tuple[float, float] = (0.9, 1.1)
) -> FloatArray:
    """Apply a contrast-protocol difference: a gain and a small offset."""
    gain = float(rng.uniform(*gain_range))
    offset = float(rng.normal(0.0, 4.0))
    return np.asarray(volume, dtype=np.float64) * gain + offset


def blur_through_plane(volume: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Separable Gaussian blur along the first axis, the coarse one in a CTV."""
    if sigma <= 0.0:
        return np.array(volume, dtype=np.float64)
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(np.asarray(volume, dtype=np.float64), ((radius, radius), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(np.asarray(volume, dtype=np.float64))
    for position, weight in enumerate(kernel):
        out += weight * padded[position : position + volume.shape[0]]
    return out


def add_noise(volume: FloatArray, rng: np.random.Generator, sigma: float = 6.0) -> FloatArray:
    return np.asarray(volume, dtype=np.float64) + rng.normal(0.0, sigma, size=np.shape(volume))


@dataclass(frozen=True)
class SegmentationAugment:
    """Augmentation applied to a training pair of image and mask."""

    flip_axes: tuple[int, ...] = (1, 2)
    phase_gain_range: tuple[float, float] = (0.92, 1.08)
    noise_sigma: float = 5.0
    through_plane_sigma: float = 0.0

    def __call__(self, image: FloatArray, mask: BoolArray, rng: np.random.Generator) -> tuple[FloatArray, BoolArray]:
        result_image = np.array(image, dtype=np.float64)
        result_mask = np.array(mask, dtype=bool)
        for axis in self.flip_axes:
            if float(rng.random()) < 0.5:
                result_image = np.ascontiguousarray(np.flip(result_image, axis=axis))
                result_mask = np.ascontiguousarray(np.flip(result_mask, axis=axis))
        result_image = random_contrast_phase(result_image, rng, self.phase_gain_range)
        if self.through_plane_sigma > 0.0:
            result_image = blur_through_plane(result_image, self.through_plane_sigma)
        result_image = add_noise(result_image, rng, self.noise_sigma)
        return result_image, result_mask


@dataclass(frozen=True)
class DescriptorStandardiser:
    """Per-column centring and scaling fitted on the training partition only."""

    mean: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    scale: FloatArray = field(default_factory=lambda: np.ones(0, dtype=np.float64))

    def fit(self, features: FloatArray) -> DescriptorStandardiser:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return DescriptorStandardiser(mean=mean, scale=scale)

    def transform(self, features: FloatArray) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if self.mean.shape[0] not in (0, values.shape[1]):
            raise ValueError("the standardiser was fitted on a different column count")
        if self.mean.shape[0] == 0:
            return values.copy()
        return (values - self.mean) / self.scale

    def inverse_transform(self, features: FloatArray) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if self.mean.shape[0] == 0:
            return values.copy()
        return values * self.scale + self.mean

    def record(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def replace_non_finite(features: FloatArray, value: float = 0.0) -> FloatArray:
    """Map the infinities of an occluded span to a finite sentinel before scaling.

    An occluded segment has infinite series resistance by construction, so it
    cannot enter a standardised design matrix unchanged. The sentinel is applied
    after the standardiser is fitted, and the occlusion is carried separately by
    the record's occlusion-length column.

    Ref: Sec. 6.5 ("a segment with the absence of opacified lumen has G = 0").
    """
    values = np.array(features, dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = float(value)
    return values
