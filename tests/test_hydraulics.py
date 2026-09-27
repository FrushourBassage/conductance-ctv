"""The flow problem, its closed forms and the shape factor.

The solver is checked against two independent references: the separable Fourier
series of an axis-aligned rectangle, whose boundary the grid represents exactly,
and a manufactured solution whose truncation error is known to be second order.
Neither reference calls the implementation it checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conductance_ctv.hydraulics.conductance import (
    analyse_section,
    axis_ratio_from_shape_factor,
    channel_report,
    component_areas,
    disc_channel_conductance,
    disc_conductance,
    ellipse_conductance,
    ellipse_shape_factor,
    is_resolution_limited,
    polya_szego_upper,
    rectangle_conductance,
    resolution_floor_area,
    shape_factor,
    shape_factor_from_axis_ratio,
)
from conductance_ctv.hydraulics.dirichlet import optimal_omega, residual_norm, solve_poisson, solve_unit_source
from conductance_ctv.hydraulics.grid import CrossSection, section_diameter, upsample_section
from conductance_ctv.hydraulics.shapes import (
    disc_channel_section,
    disc_section,
    ellipse_reference,
    ellipse_section,
    rectangle_section,
    septum_section,
)


def test_disc_conductance_closed_form() -> None:
    assert disc_conductance(0.0) == 0.0
    assert disc_conductance(2.0) == pytest.approx(math.pi * 16.0 / 8.0, rel=1e-12)
    assert disc_conductance(3.0) > disc_conductance(2.0)


def test_ellipse_conductance_reduces_to_disc() -> None:
    assert ellipse_conductance(4.0, 4.0) == pytest.approx(disc_conductance(4.0), rel=1e-12)


def test_ellipse_shape_factor_matches_axis_ratio_range() -> None:
    assert shape_factor_from_axis_ratio(0.15) == pytest.approx(3.4083, abs=1e-3)
    assert shape_factor_from_axis_ratio(0.30) == pytest.approx(1.8167, abs=1e-3)
    assert axis_ratio_from_shape_factor(shape_factor_from_axis_ratio(0.2)) == pytest.approx(0.2, abs=1e-9)


def test_polya_szego_bounds_are_ordered() -> None:
    area = 500.0
    assert polya_szego_upper(area) == pytest.approx(area * area / (8.0 * math.pi), rel=1e-12)
    assert disc_channel_conductance(area, 1) == polya_szego_upper(area)
    for channels in range(1, 6):
        assert disc_channel_conductance(area, channels) <= polya_szego_upper(area)


def _effective_half_extent(section: CrossSection) -> float:
    """Half-extent of the domain the stencil actually discretises.

    The occupied cells reach half a cell beyond their outermost centres, so a
    square of ``n`` occupied cells along an axis is the domain of side
    ``(n + 1) h``; the Fourier reference has to be evaluated on that domain and
    not on the nominal one.
    """
    cells = round(math.sqrt(section.area / section.cell_area))
    step = section.spacing_uv[0]
    return 0.5 * (cells * step + step)


def test_solver_matches_rectangle_fourier_series() -> None:
    base = rectangle_section(12.0, 12.0, spacing=1.0)
    fine_section = upsample_section(base, 2)
    coarse = analyse_section(base)
    fine = analyse_section(fine_section)
    coarse_reference = rectangle_conductance(_effective_half_extent(base), _effective_half_extent(base))
    fine_reference = rectangle_conductance(_effective_half_extent(fine_section), _effective_half_extent(fine_section))
    coarse_error = abs(coarse.conductance - coarse_reference) / coarse_reference
    fine_error = abs(fine.conductance - fine_reference) / fine_reference
    assert coarse_error < 0.01
    assert fine_error < 0.005
    assert fine_error < coarse_error


def test_solver_is_second_order_on_a_manufactured_solution() -> None:
    errors: list[float] = []
    steps: list[float] = []
    for nodes in (8, 16, 32):
        step = 1.0 / (nodes + 1)
        steps.append(step)
        section = CrossSection(np.ones((nodes, nodes), dtype=bool), (step, step))
        axis = (np.arange(nodes, dtype=np.float64) + 1.0) * step
        grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
        exact = np.sin(math.pi * grid_x) * np.sin(math.pi * grid_y)
        source = 2.0 * math.pi**2 * exact
        solution = solve_poisson(section, source=source)
        errors.append(float(np.max(np.abs(solution.field - exact))))
    first_order = math.log(errors[0] / errors[1]) / math.log(steps[0] / steps[1])
    second_order = math.log(errors[1] / errors[2]) / math.log(steps[1] / steps[2])
    assert first_order > 1.85
    assert second_order > 1.9
    assert errors[-1] < 2e-3


def test_solver_residual_is_small_and_reported() -> None:
    section = disc_section(10.0, spacing=0.5)
    solution = solve_unit_source(section)
    cropped = section.cropped(pad=1)
    assert solution.converged
    assert residual_norm(solution, cropped, source=1.0) < 1e-6


def test_empty_section_carries_no_flow() -> None:
    section = CrossSection(np.zeros((8, 8), dtype=bool), (1.0, 1.0))
    analysis = analyse_section(section)
    assert analysis.conductance == 0.0
    assert analysis.channels == 0
    assert math.isnan(analysis.shape_factor)
    assert analysis.converged


def test_digital_disc_shape_factor_approaches_one() -> None:
    coarse = analyse_section(disc_section(20.0, spacing=0.5))
    fine = analyse_section(disc_section(20.0, spacing=0.25))
    assert coarse.shape_factor > 1.0
    assert abs(fine.shape_factor - 1.0) < abs(coarse.shape_factor - 1.0) + 1e-9
    assert fine.shape_factor < 1.03


def test_digital_ellipse_shape_factor_approaches_the_closed_form() -> None:
    reference = ellipse_reference(20.0, 6.0)
    errors = []
    for step in (1.0, 0.5, 0.25):
        analysis = analyse_section(ellipse_section(20.0, 6.0, spacing=step))
        errors.append(abs(analysis.shape_factor - reference["shape_factor"]) / reference["shape_factor"])
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.02


def test_septum_attains_the_channel_floor() -> None:
    for channels in (2, 3, 4):
        analysis = analyse_section(septum_section(1800.0, channels, spacing=0.5))
        floor = disc_channel_conductance(analysis.area, channels)
        assert analysis.conductance / floor == pytest.approx(1.0, rel=0.06)
        assert analysis.shape_factor == pytest.approx(float(channels), rel=0.07)
        assert analysis.channels == channels


def test_septum_shape_factor_error_shrinks_with_the_mask_resolution() -> None:
    errors = []
    for step in (1.0, 0.5, 0.25):
        analysis = analyse_section(septum_section(1800.0, 3, spacing=step))
        errors.append(abs(analysis.shape_factor - 3.0) / 3.0)
    assert errors[0] > errors[-1]
    assert errors[-1] < 0.03


def test_septation_raises_resistance_at_constant_area() -> None:
    single = analyse_section(disc_section(math.sqrt(1800.0 / math.pi), spacing=0.5))
    septated = analyse_section(septum_section(1800.0, 3, spacing=0.5))
    assert septated.conductance < single.conductance
    assert septated.shape_factor > single.shape_factor


def test_channel_report_and_resolution_floor() -> None:
    section = disc_channel_section([4.0, 4.0], [(0.0, 0.0), (0.0, 12.0)], spacing=0.5)
    report = channel_report(section)
    assert report.count == 2
    assert report.below_floor == 0
    floor = resolution_floor_area(section.spacing_uv, min_cells=2)
    assert floor > 0.0
    assert min(report.areas) >= floor


def test_component_areas_are_descending_and_positive() -> None:
    section = disc_channel_section([5.0, 2.0, 1.0], [(0.0, 0.0), (15.0, 0.0), (0.0, 15.0)], spacing=0.5)
    areas = component_areas(section)
    assert len(areas) == 3
    assert list(areas) == sorted(areas, reverse=True)
    assert all(value > 0.0 for value in areas)


def test_resolution_limited_when_shape_factor_exceeds_channels() -> None:
    assert is_resolution_limited(3.5, 1)
    assert not is_resolution_limited(1.0, 1)
    assert not is_resolution_limited(float("nan"), 1)


def test_effective_area_exceeds_occupancy_area() -> None:
    section = disc_section(10.0, spacing=1.0)
    assert section.effective_area > section.area
    difference = section.effective_area - section.area
    assert difference == pytest.approx(0.5 * section.interface_edges * section.cell_area, rel=1e-12)


def test_optimal_omega_stays_inside_the_convergent_band() -> None:
    for shape in ((4, 4), (16, 16), (64, 64), (256, 8)):
        omega = optimal_omega(shape)
        assert 1.0 < omega < 2.0


def test_section_diameter_of_a_disc_is_its_diameter() -> None:
    section = disc_section(8.0, spacing=0.25)
    assert section_diameter(section) == pytest.approx(16.0, abs=0.6)


def test_ellipse_section_area_and_reference_agree() -> None:
    section = ellipse_section(20.0, 6.0, spacing=0.25)
    reference = ellipse_reference(20.0, 6.0)
    assert section.area == pytest.approx(reference["area"], rel=0.02)


def test_ellipse_shape_factor_helper_matches_reference() -> None:
    reference = ellipse_reference(20.0, 6.0)
    assert ellipse_shape_factor(20.0, 6.0) == pytest.approx(reference["shape_factor"], rel=1e-12)
    assert shape_factor(reference["area"], reference["conductance"]) == pytest.approx(
        reference["shape_factor"], rel=1e-12
    )
