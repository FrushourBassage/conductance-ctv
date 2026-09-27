"""Conductance, the Polya-Szego bounds and the hydraulic shape factor.

For a section solved by Eq. (1) the conductance is ``G = integral of u dA``. The
Polya-Szego inequality states that a simply connected section of area ``A``
satisfies ``G <= A^2 / (8 pi)`` with equality exactly for the disc, and that a
section made of ``k`` disc channels of combined area ``A`` satisfies
``G >= A^2 / (8 pi k)``. The symmetric ratio of the two is the shape factor

    kappa = A^2 / (8 pi G),

which is one at a circular lumen, is at least one for every section, and grows
as the lumen departs from circularity. Because it is a ratio of two integrals of
the same section it does not carry a length scale, which is the property that
makes it transportable across scanners and contrast protocols.

Ref: Sec. 6.5, Proposition 1 and the descriptor list; Sec. 3.3; Sec. 4.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conductance_ctv.hydraulics.dirichlet import (
    DEFAULT_MAX_SWEEPS,
    DEFAULT_TOLERANCE,
    conductance_of_section,
    residual_norm,
)
from conductance_ctv.hydraulics.grid import CrossSection
from conductance_ctv.utils.types import FloatArray

_EIGHT_PI = 8.0 * math.pi
_PI_SIXTH = math.pi**6


def disc_conductance(radius: float) -> float:
    """``G = pi a^4 / 8`` for a disc of radius ``a`` (Hagen-Poiseuille)."""
    if radius <= 0.0:
        return 0.0
    return math.pi * radius**4 / 8.0


def ellipse_conductance(semi_major: float, semi_minor: float) -> float:
    """``G = pi a^3 b^3 / (4 (a^2 + b^2))`` for an ellipse with semi-axes a, b."""
    if semi_major <= 0.0 or semi_minor <= 0.0:
        return 0.0
    a, b = semi_major, semi_minor
    return math.pi * a**3 * b**3 / (4.0 * (a * a + b * b))


def rectangle_conductance(half_width: float, half_height: float, terms: int = 64) -> float:
    """Conductance of a ``2a x 2b`` rectangle by its double Fourier series.

    Separating Eq. (1) on an axis-aligned rectangle gives

        G = 1024 a^3 b^3 / pi^6 * sum over odd m, n of 1 / (m^2 n^2 (m^2 b^2 + n^2 a^2)).

    The rectangle boundary is exact on a grid, so this reference carries no
    staircase error and isolates the truncation error of the five-point stencil.
    """
    if half_width <= 0.0 or half_height <= 0.0:
        return 0.0
    a, b = float(half_width), float(half_height)
    odd = np.arange(1, 2 * terms, 2, dtype=np.float64)
    mm, nn = np.meshgrid(odd, odd, indexing="ij")
    denominator = mm**2 * nn**2 * (mm**2 * b * b + nn**2 * a * a)
    series = float(np.sum(1.0 / denominator))
    return 1024.0 * a**3 * b**3 * series / _PI_SIXTH


def disc_channel_conductance(total_area: float, channels: int) -> float:
    """``A^2 / (8 pi k)``, the attainability bound for ``k`` equal disc channels."""
    if total_area <= 0.0 or channels < 1:
        return 0.0
    return total_area * total_area / (_EIGHT_PI * channels)


def polya_szego_upper(area: float) -> float:
    """``A^2 / (8 pi)``: the largest conductance a section of area ``A`` can have."""
    if area <= 0.0:
        return 0.0
    return area * area / _EIGHT_PI


def polya_szego_channel_lower(area: float, channels: int) -> float:
    """``A^2 / (8 pi k)``: the conductance floor for ``k`` equal disc channels."""
    return disc_channel_conductance(area, channels)


def shape_factor(area: float, conductance: float) -> float:
    """``kappa = A^2 / (8 pi G)``; ``nan`` when the section carries no flow."""
    if conductance <= 0.0:
        return float("nan")
    return area * area / (_EIGHT_PI * conductance)


def ellipse_shape_factor(semi_major: float, semi_minor: float) -> float:
    """``(a^2 + b^2) / (2 a b)``: the closed-form shape factor of an ellipse."""
    if semi_major <= 0.0 or semi_minor <= 0.0:
        return float("nan")
    a, b = semi_major, semi_minor
    return (a * a + b * b) / (2.0 * a * b)


def axis_ratio_from_shape_factor(kappa: float) -> float:
    """Invert ``kappa = (1 + r^2) / (2 r)`` for the axis ratio ``r <= 1``."""
    if not np.isfinite(kappa) or kappa < 1.0:
        return float("nan")
    return float(kappa - math.sqrt(max(0.0, kappa * kappa - 1.0)))


def shape_factor_from_axis_ratio(ratio: float) -> float:
    """``kappa`` of an ellipse whose semi-axis ratio is ``ratio <= 1``."""
    if ratio <= 0.0:
        return float("nan")
    return (1.0 + ratio * ratio) / (2.0 * ratio)


def resistance_floor(area: float, conductance: float) -> float:
    """``G / (A^2 / (8 pi))``: the share of the attainable conductance realised."""
    upper = polya_szego_upper(area)
    if upper <= 0.0:
        return 0.0
    return float(conductance) / upper


def resolution_floor_area(spacing_uv: tuple[float, float], min_cells: int = 2) -> float:
    """Smallest component area that counts as a channel.

    A component must span at least ``min_cells`` cells along its longer grid axis,
    otherwise it is indistinguishable from a single-cell threshold artefact. The
    resulting floor is the area of the disc inscribed in that span.

    Ref: Algorithm 1 line 4 ("keep components above the resolution floor").
    """
    span = max(min_cells * max(spacing_uv), max(spacing_uv))
    radius = 0.5 * span
    return math.pi * radius * radius


@dataclass(frozen=True)
class ChannelReport:
    """Channel decomposition of a section, as used by Algorithm 1 line 4."""

    count: int
    areas: tuple[float, ...]
    below_floor: int
    floor_area: float

    @property
    def dominant_area(self) -> float:
        if not self.areas:
            return 0.0
        return float(max(self.areas))


def component_areas(section: CrossSection) -> tuple[float, ...]:
    """Effective area of every 4-connected component of a section, in mm^2.

    Each component is measured with the same half-cell convention as the whole
    section, so a component's area is consistent with the conductance it carries.
    """
    if section.is_empty:
        return ()
    cropped = section.cropped(pad=0)
    labels, count = cropped.label_map()
    if count == 0:
        return ()
    areas: list[float] = []
    for label in range(1, count + 1):
        component = labels == label
        rows = np.flatnonzero(component.any(axis=1))
        cols = np.flatnonzero(component.any(axis=0))
        window = component[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        piece = CrossSection(window, cropped.spacing_uv)
        areas.append(piece.effective_area)
    return tuple(sorted(areas, reverse=True))


def channel_report(section: CrossSection, floor_area: float | None = None) -> ChannelReport:
    """Count the 4-connected components of a section above the resolution floor."""
    if section.is_empty:
        return ChannelReport(0, (), 0, 0.0 if floor_area is None else floor_area)
    floor = resolution_floor_area(section.spacing_uv) if floor_area is None else float(floor_area)
    areas = component_areas(section)
    if not areas:
        return ChannelReport(0, (), 0, floor)
    keep = [value for value in areas if value >= floor]
    return ChannelReport(
        count=len(keep),
        areas=tuple(keep),
        below_floor=len(areas) - len(keep),
        floor_area=floor,
    )


def is_resolution_limited(kappa: float, channels: int) -> bool:
    """Whether the shape factor and the channel count disagree.

    For ``k`` equal disc channels ``kappa`` equals ``k``. A section whose shape
    factor exceeds its counted channel count is more resistive than that bound
    permits, which is what sub-resolution septa produce: the irregularity is
    still carried by ``kappa`` while the count has already collapsed.

    Ref: Sec. 4.1 ("the conflict between them denotes that the resolution is low").
    """
    if not np.isfinite(kappa):
        return False
    return bool(kappa > float(channels) + 0.5)


@dataclass(frozen=True)
class SectionHydraulics:
    """Everything Eq. (1) yields for one perpendicular section."""

    area: float
    occupancy_area: float
    interface_edges: int
    conductance: float
    shape_factor: float
    channels: int
    iterations: int
    residual: float
    converged: bool
    resolution_limited: bool
    channel_areas: tuple[float, ...]
    below_floor: int


def analyse_section(
    section: CrossSection,
    tolerance: float = DEFAULT_TOLERANCE,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
    floor_area: float | None = None,
) -> SectionHydraulics:
    """Solve Eq. (1) on a section and derive its hydraulic signature."""
    occupancy_area = section.area
    area = section.effective_area
    edges = section.interface_edges
    if section.is_empty:
        return SectionHydraulics(
            area=0.0,
            occupancy_area=0.0,
            interface_edges=0,
            conductance=0.0,
            shape_factor=float("nan"),
            channels=0,
            iterations=0,
            residual=0.0,
            converged=True,
            resolution_limited=False,
            channel_areas=(),
            below_floor=0,
        )
    conductance, solution = conductance_of_section(section, tolerance=tolerance, max_sweeps=max_sweeps)
    kappa = shape_factor(area, conductance)
    report = channel_report(section, floor_area=floor_area)
    return SectionHydraulics(
        area=area,
        occupancy_area=occupancy_area,
        interface_edges=edges,
        conductance=conductance,
        shape_factor=kappa,
        channels=report.count,
        iterations=solution.iterations,
        residual=residual_norm(solution, section.cropped(pad=1), source=1.0),
        converged=solution.converged,
        resolution_limited=is_resolution_limited(kappa, report.count),
        channel_areas=report.areas,
        below_floor=report.below_floor,
    )


def conductance_curve(
    sections: list[CrossSection],
    tolerance: float = DEFAULT_TOLERANCE,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per-section ``(area, conductance, shape_factor, channels)`` along a path."""
    areas = np.zeros(len(sections), dtype=np.float64)
    conductances = np.zeros(len(sections), dtype=np.float64)
    kappas = np.full(len(sections), np.nan, dtype=np.float64)
    channels = np.zeros(len(sections), dtype=np.float64)
    for index, section in enumerate(sections):
        result = analyse_section(section, tolerance=tolerance, max_sweeps=max_sweeps)
        areas[index] = result.area
        conductances[index] = result.conductance
        kappas[index] = result.shape_factor
        channels[index] = float(result.channels)
    return areas, conductances, kappas, channels


def sections_from_mask_stack(stack: FloatArray, spacing_uv: tuple[float, float]) -> list[CrossSection]:
    """Wrap a stack of binary section planes as :class:`CrossSection` objects."""
    return [CrossSection(np.asarray(plane, dtype=bool), spacing_uv) for plane in stack]
