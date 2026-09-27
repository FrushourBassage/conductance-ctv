"""The resistance algebra, the centreline and the descriptor vector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conductance_ctv.hydraulics.centerline import (
    SEGMENT_NAMES,
    build_path,
    centroid_path,
    estimate_tangents,
    smooth_path,
)
from conductance_ctv.hydraulics.conductance import analyse_section
from conductance_ctv.hydraulics.descriptors import (
    DESCRIPTOR_FIELDS,
    absolute_columns,
    describe_sections,
    descriptor_feature_names,
    descriptor_matrix,
    dimensionless_columns,
    sample_sections,
    signature_from_mask,
)
from conductance_ctv.hydraulics.grid import CrossSection, PlaneBasis, orthonormal_frame, resample_arclength
from conductance_ctv.hydraulics.series import (
    SpanError,
    conductance_weighted_length,
    minimal_area_index,
    minimal_conductance_index,
    occlusion_length,
    patent_mask,
    propagated_resistance_error,
    reference_resistance,
    reference_span,
    resistance_ratio,
    resistance_sensitivity,
    series_resistance,
)
from conductance_ctv.hydraulics.shapes import disc_section, ellipse_section, septum_section


def test_series_resistance_matches_hand_computation() -> None:
    conductance = np.array([10.0, 5.0, 0.0, 20.0], dtype=np.float64)
    assert series_resistance(conductance, 2.0) == pytest.approx(2.0 / 10.0 + 2.0 / 5.0 + 2.0 / 20.0)
    assert occlusion_length(conductance, 2.0) == pytest.approx(2.0)
    assert list(patent_mask(conductance)) == [True, True, False, True]


def test_series_resistance_of_a_fully_occluded_span_is_infinite() -> None:
    assert math.isinf(series_resistance(np.zeros(4, dtype=np.float64), 1.0))


def test_series_resistance_rejects_an_empty_span() -> None:
    with pytest.raises(SpanError):
        series_resistance(np.zeros(0, dtype=np.float64), 1.0)


def test_reference_resistance_is_length_normalised() -> None:
    conductance = np.full(10, 4.0, dtype=np.float64)
    doubled = reference_resistance(conductance, 20.0, 1.0)
    single = reference_resistance(conductance, 10.0, 1.0)
    assert doubled == pytest.approx(2.0 * single)


def test_resistance_ratio_is_dimensionless_and_at_least_one_on_a_lesion() -> None:
    reference = reference_resistance(np.full(20, 8.0, dtype=np.float64), 20.0, 1.0)
    healthy = series_resistance(np.full(20, 8.0, dtype=np.float64), 1.0)
    lesioned = series_resistance(np.concatenate([np.full(19, 8.0), np.array([2.0])]), 1.0)
    assert resistance_ratio(healthy, reference) == pytest.approx(1.0, rel=1e-12)
    assert resistance_ratio(lesioned, reference) > 1.0
    assert math.isinf(resistance_ratio(math.inf, reference))


def test_reference_span_selection_modes() -> None:
    candidates = {"CIV": np.full(5, 1.0), "EIV": np.full(5, 4.0), "CFV": np.full(5, 2.0)}
    assert reference_span(candidates, 5.0, 1.0, mode="max") == "EIV"
    assert reference_span(candidates, 5.0, 1.0, mode="upstream", order=("CIV", "CFV")) == "CIV"
    assert reference_span(candidates, 5.0, 1.0, mode="explicit", order=("CFV",)) == "CFV"
    with pytest.raises(SpanError):
        reference_span({}, 5.0, 1.0)
    with pytest.raises(SpanError):
        reference_span(candidates, 0.0, 1.0)
    with pytest.raises(SpanError):
        reference_span(candidates, 5.0, 1.0, mode="nonsense")


def test_reference_span_rejects_an_all_closed_candidate_set() -> None:
    with pytest.raises(SpanError):
        reference_span({"CIV": np.zeros(4)}, 4.0, 1.0)


def test_conductance_weighted_length_shrinks_with_a_lesion() -> None:
    reference = 8.0
    healthy = conductance_weighted_length(np.full(10, 8.0, dtype=np.float64), 1.0, reference)
    lesioned = conductance_weighted_length(np.full(10, 2.0, dtype=np.float64), 1.0, reference)
    assert healthy == pytest.approx(10.0)
    assert lesioned < healthy


def test_conductance_weighted_length_is_not_the_span_length() -> None:
    # A profile whose mean equals the reference: a weighting by G / G_ref would
    # return the span length exactly, so only the resistance-equivalent form can
    # detect the burden the uneven profile carries.
    reference = 8.0
    uneven = np.array([4.0, 4.0, 12.0, 12.0], dtype=np.float64)
    assert float(np.mean(uneven)) == pytest.approx(reference)
    weighted = conductance_weighted_length(uneven, 1.0, reference)
    resistance = float(np.sum(1.0 / uneven))
    assert weighted == pytest.approx(4.0 * (4.0 / reference) / resistance)
    assert weighted == pytest.approx(3.0)
    assert weighted < 4.0


def test_conductance_weighted_length_without_a_patent_reference_is_zero() -> None:
    assert conductance_weighted_length(np.full(4, 3.0), 1.0, 0.0) == 0.0


def test_minimum_indices_are_reported_separately() -> None:
    conductance = np.array([9.0, 8.0, 50.0, 4.0, 7.0])
    area = np.array([100.0, 90.0, 5.0, 95.0, 80.0])
    assert minimal_conductance_index(conductance) == 3
    assert minimal_area_index(area) == 2
    assert minimal_conductance_index(conductance, patent=np.array([True, True, True, False, True])) == 4


def test_propagated_error_doubles_the_relative_area_error() -> None:
    # The linearised propagation the manuscript quotes is exactly twice the area error.
    assert propagated_resistance_error(100.0, 0.041) == pytest.approx(0.082)
    # The exact inverse-square sensitivity of 1 / A^2 to a relative area error e
    # is (2 e + e^2) / (1 + e)^2, which sits just below the linearised value.
    area_error = 0.041
    exact = (2.0 * area_error + area_error**2) / (1.0 + area_error) ** 2
    assert resistance_sensitivity(100.0, area_error) == pytest.approx(exact, rel=1e-9)
    assert resistance_sensitivity(100.0, area_error) < 0.082
    assert resistance_sensitivity(100.0, area_error) > 0.9 * 0.082


def test_propagated_error_of_a_degenerate_area_is_infinite() -> None:
    assert math.isinf(propagated_resistance_error(0.0, 0.1))


def test_orthonormal_frame_is_right_handed() -> None:
    tangent = np.array([0.2, 0.9, 0.3], dtype=np.float64)
    axis_u, axis_v = orthonormal_frame(tangent)
    assert float(np.dot(axis_u, axis_v)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.linalg.norm(np.cross(axis_u, axis_v))) == pytest.approx(1.0, abs=1e-12)
    assert float(np.dot(axis_u, tangent)) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        orthonormal_frame(np.zeros(3))


def test_plane_basis_rejects_a_degenerate_frame() -> None:
    with pytest.raises(ValueError):
        PlaneBasis(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_resample_arclength_uniform_grid() -> None:
    points = np.stack([np.arange(5, dtype=np.float64), np.zeros(5), np.zeros(5)], axis=1)
    resampled, values, grid = resample_arclength(points, np.arange(5, dtype=np.float64), 0.5)
    assert resampled.shape[0] == grid.shape[0]
    assert grid[1] - grid[0] == pytest.approx(0.5)
    assert values.shape[0] == grid.shape[0]
    with pytest.raises(ValueError):
        resample_arclength(points[:1], None, 1.0)


def test_smooth_path_preserves_shape_and_reduces_jitter() -> None:
    points = np.zeros((20, 3), dtype=np.float64)
    points[:, 0] = np.arange(20)
    points[10, 2] = 5.0
    smoothed = smooth_path(points, window=5)
    assert smoothed.shape == points.shape
    assert abs(smoothed[10, 2]) < 5.0
    assert smooth_path(points, window=1).shape == points.shape


def test_tangents_are_unit_vectors() -> None:
    points = np.stack([np.arange(6, dtype=np.float64), np.arange(6, dtype=np.float64), np.zeros(6)], axis=1)
    tangents = estimate_tangents(points)
    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)


def test_centroid_path_recovers_a_straight_tube() -> None:
    mask = np.zeros((24, 32, 32), dtype=bool)
    mask[:, 14:18, 16:20] = True
    path = centroid_path(mask, (1.0, 1.0, 1.0))
    assert path.shape == (24, 3)
    assert np.allclose(path[:, 1], 15.5)
    assert np.allclose(path[:, 2], 17.5)


def test_centroid_path_rejects_an_empty_mask() -> None:
    with pytest.raises(ValueError):
        centroid_path(np.zeros((4, 4, 4), dtype=bool), (1.0, 1.0, 1.0))


def test_build_path_resamples_at_the_requested_step() -> None:
    mask = np.zeros((32, 32, 32), dtype=bool)
    mask[:, 14:18, 14:18] = True
    path = build_path(mask, (1.0, 1.0, 1.0), step=2.0)
    assert path.step == pytest.approx(2.0, rel=1e-9)
    # The uniform grid ends within one step of the anatomical arc length.
    assert path.total_length == pytest.approx(31.0, rel=0.08)
    frame = path.frame(1)
    assert frame.origin.shape == (3,)


def test_segment_labels_cover_three_segments() -> None:
    mask = np.zeros((40, 32, 32), dtype=bool)
    mask[:, 14:18, 14:18] = True
    path = build_path(mask, (1.0, 1.0, 1.0), step=1.0)
    labels = path.segment_of((0.0, 0.40, 0.75, 1.0))
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    for index in range(3):
        start, stop = path.bounds_for(index, (0.0, 0.40, 0.75, 1.0))
        assert stop > start


def test_segment_fractions_must_span_zero_to_one() -> None:
    mask = np.zeros((12, 16, 16), dtype=bool)
    mask[:, 7:9, 7:9] = True
    path = build_path(mask, (1.0, 1.0, 1.0), step=1.0)
    with pytest.raises(ValueError):
        path.segment_of((0.1, 0.4, 0.75, 1.0))
    with pytest.raises(ValueError):
        path.segment_of((0.0, 0.5, 0.4, 1.0))


def test_sample_sections_follow_the_path() -> None:
    mask = np.zeros((24, 32, 32), dtype=bool)
    mask[:, 13:19, 13:19] = True
    path = build_path(mask, (1.0, 1.0, 1.0), step=2.0)
    sections = sample_sections(mask, (1.0, 1.0, 1.0), path, 1.0, 24.0)
    assert len(sections) == path.points.shape[0]
    assert all(section.occupancy.shape == (25, 25) for section in sections)
    assert all(not section.is_empty for section in sections)


def test_describe_sections_assembles_seven_fields_per_segment() -> None:
    results = [analyse_section(disc_section(6.0, spacing=0.5)) for _ in range(30)]
    arclength = np.arange(30, dtype=np.float64)
    labels = np.array([0] * 12 + [1] * 10 + [2] * 8, dtype=np.int64)
    signature = describe_sections(results, arclength, labels)
    assert len(signature.segments) == 3
    assert signature.vector().shape[0] == 21
    assert len(signature.descriptor_names()) == 21
    assert signature.segments[0].patent
    assert signature.diagnostics["occluded_samples"] == 0.0
    record = signature.record()
    assert "CIV.minimum_luminal_area" in record


def test_describe_sections_marks_an_occluded_segment() -> None:
    healthy = [analyse_section(disc_section(6.0, spacing=0.5)) for _ in range(10)]
    closed = [analyse_section(CrossSection(np.zeros((16, 16), dtype=bool), (0.5, 0.5))) for _ in range(5)]
    results = closed + healthy
    arclength = np.arange(15, dtype=np.float64)
    labels = np.array([0] * 5 + [1] * 10, dtype=np.int64)
    signature = describe_sections(results, arclength, labels)
    occluded = next(segment for segment in signature.segments if segment.name == "CIV")
    assert not occluded.patent
    assert math.isinf(occluded.series_resistance)
    assert occluded.occlusion_length == pytest.approx(5.0)
    assert signature.diagnostics["occluded_samples"] == 5.0


def test_signature_from_mask_runs_end_to_end() -> None:
    mask = np.zeros((48, 48, 48), dtype=bool)
    for z in range(48):
        radius = 7.0 - 2.0 * z / 47.0
        yy, xx = np.meshgrid(np.arange(48) - 23.5, np.arange(48) - 23.5, indexing="ij")
        mask[z] = (yy**2 + xx**2) <= radius**2
    signature = signature_from_mask(mask, (1.0, 1.0, 1.0), step=2.0)
    assert len(signature.segments) == 3
    assert all(segment.minimum_luminal_area > 0.0 for segment in signature.segments)
    assert all(segment.shape_factor >= 1.0 for segment in signature.segments)


def test_descriptor_feature_names_and_column_families() -> None:
    names = descriptor_feature_names()
    assert len(names) == 21
    assert names[0] == "CIV_minimum_luminal_area"
    dimensionless = dimensionless_columns()
    absolute = absolute_columns()
    assert not set(dimensionless) & set(absolute)
    assert len(dimensionless) == 6
    assert len(absolute) == 6
    for index in dimensionless:
        assert names[index].split("_", 1)[1] in {"resistance_ratio", "shape_factor"}


def test_descriptor_matrix_stacks_vectors() -> None:
    mask = np.zeros((32, 32, 32), dtype=bool)
    mask[:, 14:18, 14:18] = True
    first = signature_from_mask(mask, (1.0, 1.0, 1.0), step=2.0)
    matrix = descriptor_matrix([first, first])
    assert matrix.shape == (2, 21)
    assert descriptor_matrix([]).shape == (0, 21)
    assert DESCRIPTOR_FIELDS[1] == "minimum_conductance_arclength"


def test_segment_names_are_the_three_anatomical_segments() -> None:
    assert SEGMENT_NAMES == ("CIV", "EIV", "CFV")


def test_ellipse_and_septum_analyses_differ_in_the_expected_direction() -> None:
    slit = analyse_section(ellipse_section(12.0, 3.0, spacing=0.5))
    septated = analyse_section(septum_section(math.pi * 36.0, 3, spacing=0.5))
    assert slit.channels == 1
    assert septated.channels == 3
    assert septated.shape_factor > slit.shape_factor
