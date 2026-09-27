"""Analytic cross-section constructors.

These build the occupancy arrays whose conductance is known in closed form, so
the finite-difference solve of Eq. (1) can be checked against the disc, the
ellipse and the multi-channel bounds of Proposition 1 on the same objects the
pipeline consumes. They are also the shape vocabulary of the cohort generator in
:mod:`conductance_ctv.data.fixture`.

Ref: Sec. 6.5, Proposition 1 and the axis-ratio range quoted in Sec. 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from conductance_ctv.hydraulics.conductance import ellipse_conductance
from conductance_ctv.hydraulics.grid import CrossSection
from conductance_ctv.utils.types import FloatArray

_DEFAULT_MARGIN = 4.0


def grid_shape_for_extent(extent_uv: tuple[float, float], spacing: float) -> tuple[int, int]:
    """Number of cells needed to cover an extent at a given isotropic spacing."""
    if spacing <= 0.0:
        raise ValueError("spacing must be strictly positive")
    extents = [max(float(value), spacing) for value in extent_uv]
    return (
        max(2, int(math.ceil(extents[0] / spacing)) + 1),
        max(2, int(math.ceil(extents[1] / spacing)) + 1),
    )


def _centred_axes(shape: tuple[int, int], spacing: float) -> tuple[FloatArray, FloatArray]:
    rows = (np.arange(shape[0], dtype=np.float64) - (shape[0] - 1) / 2.0) * spacing
    cols = (np.arange(shape[1], dtype=np.float64) - (shape[1] - 1) / 2.0) * spacing
    return np.meshgrid(rows, cols, indexing="ij")


def disc_section(radius: float, spacing: float = 1.0, margin: float = _DEFAULT_MARGIN) -> CrossSection:
    """A single circular lumen of the given radius (mm)."""
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive")
    extent = 2.0 * radius + 2.0 * margin
    shape = grid_shape_for_extent((extent, extent), spacing)
    rows, cols = _centred_axes(shape, spacing)
    occupancy = (rows * rows + cols * cols) <= radius * radius
    return CrossSection(occupancy, (spacing, spacing))


def ellipse_section(
    semi_major: float,
    semi_minor: float,
    spacing: float = 1.0,
    margin: float = _DEFAULT_MARGIN,
    centre: tuple[float, float] = (0.0, 0.0),
) -> CrossSection:
    """An elliptical lumen; the slit-like compression of Sec. 1 is the case b << a."""
    if semi_major <= 0.0 or semi_minor <= 0.0:
        raise ValueError("semi-axes must be strictly positive")
    a = max(semi_major, semi_minor)
    b = min(semi_major, semi_minor)
    extent = (2.0 * a + 2.0 * margin, 2.0 * b + 2.0 * margin)
    shape = grid_shape_for_extent(extent, spacing)
    rows, cols = _centred_axes(shape, spacing)
    occupancy = ((rows - centre[0]) / a) ** 2 + ((cols - centre[1]) / b) ** 2 <= 1.0
    return CrossSection(occupancy, (spacing, spacing))


def rectangle_section(
    half_width: float,
    half_height: float,
    spacing: float = 1.0,
    margin: float = _DEFAULT_MARGIN,
) -> CrossSection:
    """An axis-aligned rectangle, used as the staircase-free convergence anchor.

    A rectangle is representable exactly on the grid, so its conductance differs
    from the separable Fourier series only by the truncation error of the
    five-point stencil.
    """
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("half-extents must be strictly positive")
    extent = (2.0 * half_width + 2.0 * margin, 2.0 * half_height + 2.0 * margin)
    shape = grid_shape_for_extent(extent, spacing)
    rows, cols = _centred_axes(shape, spacing)
    occupancy = (np.abs(rows) <= half_width) & (np.abs(cols) <= half_height)
    return CrossSection(occupancy, (spacing, spacing))


def disc_channel_section(
    radii: Sequence[float],
    centres: Sequence[tuple[float, float]],
    spacing: float = 1.0,
    margin: float = _DEFAULT_MARGIN,
) -> CrossSection:
    """A union of circular channels of the given radii and centres, in mm."""
    if len(radii) != len(centres):
        raise ValueError("radii and centres must have the same length")
    if not radii:
        raise ValueError("at least one channel is required")
    radius_values = [float(value) for value in radii]
    if any(value <= 0.0 for value in radius_values):
        raise ValueError("channel radii must be strictly positive")
    ys = [float(point[0]) for point in centres]
    xs = [float(point[1]) for point in centres]
    reach = max(radius_values)
    extent = (
        2.0 * (max(abs(value) for value in ys) + reach) + 2.0 * margin,
        2.0 * (max(abs(value) for value in xs) + reach) + 2.0 * margin,
    )
    shape = grid_shape_for_extent(extent, spacing)
    rows, cols = _centred_axes(shape, spacing)
    occupancy = np.zeros(shape, dtype=bool)
    for radius, y, x in zip(radius_values, ys, xs, strict=True):
        occupancy |= (rows - y) ** 2 + (cols - x) ** 2 <= radius * radius
    return CrossSection(occupancy, (spacing, spacing))


def septum_section(
    total_area: float,
    channels: int,
    spacing: float = 1.0,
    margin: float = _DEFAULT_MARGIN,
    separation: float = 1.0,
) -> CrossSection:
    """``channels`` equal discs of combined area ``total_area``, laid out apart.

    This is the configuration that attains the lower bound of Proposition 1, so
    ``G`` equals ``A^2 / (8 pi k)`` up to discretisation.
    """
    if channels < 1:
        raise ValueError("channels must be at least one")
    if total_area <= 0.0:
        raise ValueError("total area must be strictly positive")
    radius = math.sqrt(total_area / (math.pi * channels))
    pitch = 2.0 * radius + separation
    span = (channels - 1) * pitch
    centres = [(0.0, -0.5 * span + index * pitch) for index in range(channels)]
    return disc_channel_section([radius] * channels, centres, spacing=spacing, margin=margin)


def equal_area_radius(total_area: float, channels: int) -> float:
    """Radius of each of ``channels`` equal discs of combined area ``total_area``."""
    if channels < 1 or total_area <= 0.0:
        return 0.0
    return math.sqrt(total_area / (math.pi * channels))


def ellipse_reference(semi_major: float, semi_minor: float) -> dict[str, float]:
    """Closed-form area, conductance and shape factor of an ellipse."""
    a, b = max(semi_major, semi_minor), min(semi_major, semi_minor)
    area = math.pi * a * b
    conductance = ellipse_conductance(a, b)
    return {
        "area": area,
        "conductance": conductance,
        "shape_factor": area * area / (8.0 * math.pi * conductance),
    }


def slit_axis_ratios(count: int, low: float = 0.15, high: float = 0.30) -> FloatArray:
    """The axis-ratio range quoted for slit-like compression in the manuscript."""
    if count < 2:
        raise ValueError("a range needs at least two samples")
    return np.linspace(low, high, count, dtype=np.float64)
