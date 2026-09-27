"""Unit-source Dirichlet problem on a lumen cross-section.

The flow problem in a slowly varying duct reduces, for steady fully developed
laminar flow, to a Poisson problem with a constant unit source and a homogeneous
Dirichlet condition on the lumen boundary:

    -div(grad u) = 1  in Omega,   u = 0 on the boundary of Omega.

The conductance of the section is the integral of the solution,

    G(Omega) = integral over Omega of u dA,

which reproduces the Hagen-Poiseuille conductance pi a^4 / 8 for a disc of radius
a and is the quantity bounded by the Polya-Szego inequality for a general
section.

Ref: Sec. 6.5, Eq. (1); Algorithm 1 lines 5-6.

The solve is a red-black successive over-relaxation sweep. It is chosen over a
direct sparse factorisation because the same masked stencil is reused for a few
hundred sections per volume and the sweep vectorises over the whole section,
while the anisotropic in-plane spacing of a resampled section is carried by the
same stencil coefficients. The source is a field rather than the constant one of
Eq. (1) so that the solver can be exercised against an analytic solution on a
staircase-free domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conductance_ctv.hydraulics.grid import CrossSection
from conductance_ctv.utils.types import FloatArray

DEFAULT_TOLERANCE = 1e-11
DEFAULT_MAX_SWEEPS = 20_000
_MIN_OMEGA = 1.0
_MAX_OMEGA = 1.95


@dataclass(frozen=True)
class DirichletSolution:
    """Solution field of Eq. (1) with the convergence bookkeeping kept visible."""

    field: FloatArray
    iterations: int
    delta: float
    converged: bool

    def integral(self, cell_area: float) -> float:
        """``G = integral of u dA`` for the solved field."""
        return float(self.field.sum()) * float(cell_area)


def optimal_omega(shape: tuple[int, int]) -> float:
    """Spectral-radius-matched relaxation factor for a rectangular domain."""
    span = max(max(shape), 2)
    raw = 2.0 / (1.0 + math.sin(math.pi / span))
    return float(min(_MAX_OMEGA, max(_MIN_OMEGA, raw)))


def as_source_field(section: CrossSection, source: float | FloatArray) -> FloatArray:
    """Broadcast a scalar or field source onto the section, zero outside it."""
    occupancy = section.occupancy
    if np.isscalar(source):
        field = np.zeros(occupancy.shape, dtype=np.float64)
        field[occupancy] = float(source)  # type: ignore[arg-type]
        return field
    field = np.asarray(source, dtype=np.float64)
    if field.shape != occupancy.shape:
        raise ValueError("a source field must match the section shape")
    return np.where(occupancy, field, 0.0)


def apply_laplacian(field: FloatArray, occupancy: np.ndarray, spacing_uv: tuple[float, float]) -> FloatArray:
    """Masked five-point Laplacian with the homogeneous Dirichlet condition.

    Cells outside the occupancy act as u = 0, exactly as the boundary condition
    prescribes, so they enter the stencil as zero-valued neighbours.
    """
    dx, dy = spacing_uv
    padded = np.zeros((field.shape[0] + 2, field.shape[1] + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = field
    centre = padded[1:-1, 1:-1]
    lap = (padded[:-2, 1:-1] + padded[2:, 1:-1] - 2.0 * centre) / (dx * dx) + (
        padded[1:-1, :-2] + padded[1:-1, 2:] - 2.0 * centre
    ) / (dy * dy)
    lap[~occupancy] = 0.0
    return -lap


def residual_norm(solution: DirichletSolution, section: CrossSection, source: float | FloatArray = 1.0) -> float:
    """Maximum absolute residual of the Poisson problem over the section interior."""
    if section.is_empty:
        return 0.0
    rhs = as_source_field(section, source)
    residual = apply_laplacian(solution.field, section.occupancy, section.spacing_uv) - rhs
    residual[~section.occupancy] = 0.0
    return float(np.max(np.abs(residual)))


def solve_poisson(
    section: CrossSection,
    source: float | FloatArray = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
    omega: float | None = None,
    initial: FloatArray | None = None,
) -> DirichletSolution:
    """Solve ``-div(grad u) = source`` with ``u = 0`` outside the occupancy.

    An empty section returns an all-zero field with ``converged=True``: a
    non-opacified lumen carries no flow and its conductance is exactly zero by
    definition (Sec. 6.5).
    """
    occupancy = section.occupancy
    if not bool(occupancy.any()):
        return DirichletSolution(np.zeros(occupancy.shape, dtype=np.float64), 0, 0.0, True)

    rhs = as_source_field(section, source)
    dx, dy = section.spacing_uv
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    diagonal = 2.0 * (ax + ay)
    relaxation = optimal_omega(occupancy.shape) if omega is None else float(omega)

    field = np.zeros(occupancy.shape, dtype=np.float64) if initial is None else np.array(initial, dtype=np.float64)
    field[~occupancy] = 0.0

    parity = (np.add.outer(np.arange(occupancy.shape[0]), np.arange(occupancy.shape[1])) % 2).astype(bool)
    converged = False
    sweeps = 0
    delta = float("inf")

    for _ in range(max_sweeps):
        sweeps += 1
        delta = 0.0
        for colour in (False, True):
            east = np.zeros_like(field)
            west = np.zeros_like(field)
            north = np.zeros_like(field)
            south = np.zeros_like(field)
            east[:, :-1] = field[:, 1:]
            west[:, 1:] = field[:, :-1]
            north[:-1, :] = field[1:, :]
            south[1:, :] = field[:-1, :]
            gauss = (rhs + ax * (east + west) + ay * (north + south)) / diagonal
            active = occupancy & (parity == colour)
            if not bool(active.any()):
                continue
            updated = field[active] + relaxation * (gauss[active] - field[active])
            delta = max(delta, float(np.max(np.abs(updated - field[active]))))
            field[active] = updated
        if delta < tolerance:
            converged = True
            break

    return DirichletSolution(field, sweeps, delta, converged)


def solve_unit_source(
    section: CrossSection,
    tolerance: float = DEFAULT_TOLERANCE,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
) -> DirichletSolution:
    """Eq. (1) on the bounding-box crop of ``section``.

    Cropping removes the lumen-free margin that contributes nothing to the
    integral and keeps the sweep cost proportional to the lumen rather than to
    the requested output grid.
    """
    cropped = section.cropped(pad=1)
    return solve_poisson(cropped, source=1.0, tolerance=tolerance, max_sweeps=max_sweeps)


def conductance_of_section(
    section: CrossSection,
    tolerance: float = DEFAULT_TOLERANCE,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
) -> tuple[float, DirichletSolution]:
    """Return ``(G, solution)`` for a section, cropping before the solve."""
    cropped = section.cropped(pad=1)
    solution = solve_poisson(cropped, source=1.0, tolerance=tolerance, max_sweeps=max_sweeps)
    return solution.integral(cropped.cell_area), solution
