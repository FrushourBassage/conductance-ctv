"""The cohort fixture: schema, generation, profiles and the analytic truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conductance_ctv.data.fixture import (
    DEFAULT_GRID,
    analytic_descriptor_vector,
    baseline_radius,
    build_profile,
    draw_lesion_type,
    generate_cohort,
    generate_limb,
    render_intensity,
    render_mask,
)
from conductance_ctv.data.schema import (
    LESION_TYPE_COUNT,
    LESION_TYPE_DESCRIPTION,
    AcquisitionPhase,
    GridSpec,
    LesionType,
    SectionKind,
    Site,
    confusion_counts,
)
from conductance_ctv.data.splits import (
    accrual_halves,
    crossfit_folds,
    leave_one_site_out,
    patient_level_split,
    reference_type_counts,
    site_stratified_patients,
    stratum_of,
)


def test_grid_spec_rejects_a_degenerate_grid() -> None:
    with pytest.raises(ValueError):
        GridSpec(shape=(2, 4, 4), spacing_zyx=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        GridSpec(shape=(4, 4, 4), spacing_zyx=(0.0, 1.0, 1.0))


def test_grid_spec_extent_and_voxel_volume() -> None:
    grid = GridSpec(shape=(4, 5, 6), spacing_zyx=(1.0, 0.5, 2.0))
    assert grid.extent_zyx == (4.0, 2.5, 12.0)
    assert grid.voxel_volume == pytest.approx(1.0)


def test_lesion_type_descriptions_cover_four_types() -> None:
    assert LESION_TYPE_COUNT == 4
    for lesion_type in LesionType:
        assert lesion_type in LESION_TYPE_DESCRIPTION
    assert "post-thrombotic" in LESION_TYPE_DESCRIPTION[LesionType.POST_THROMBOTIC]


def test_baseline_radius_tapers_from_confluence_to_bifurcation() -> None:
    z = np.linspace(0.0, 100.0, 50)
    radius = baseline_radius(z, 100.0)
    assert radius[0] > radius[-1]
    assert float(np.max(np.abs(radius - radius.mean()))) < 2.0


def test_profile_section_kinds_per_type() -> None:
    rng = np.random.default_rng(1)
    expectations = {
        LesionType.COMPRESSIVE: SectionKind.ELLIPSE,
        LesionType.COMPRESSIVE_WITH_COLLATERALS: SectionKind.ELLIPSE,
        LesionType.POST_THROMBOTIC: SectionKind.SEPTUM,
        LesionType.OCCLUSIVE: SectionKind.OCCLUDED,
    }
    for lesion_type, expected in expectations.items():
        profile = build_profile(DEFAULT_GRID, lesion_type, _civ(), 0.2, rng)
        start, stop = profile.lesion_slice_range
        assert profile.kind[start] is expected
        assert profile.kind[0] is SectionKind.DISC or start == 0


def test_profile_truth_is_finite_and_ordered() -> None:
    rng = np.random.default_rng(2)
    for lesion_type in LesionType:
        profile = build_profile(DEFAULT_GRID, lesion_type, _civ(), 0.25, rng)
        area = profile.unit_area()
        conductance = profile.unit_conductance()
        assert np.all(area >= 0.0)
        assert np.all(conductance >= 0.0)
        assert np.all(conductance <= area * area / (8.0 * math.pi) + 1e-9)
        channels = profile.unit_channels()
        assert np.all(channels >= 0)
        if lesion_type is LesionType.OCCLUSIVE:
            assert float(np.min(conductance)) == 0.0


def test_profile_lesion_lies_inside_its_segment() -> None:
    profile = build_profile(DEFAULT_GRID, LesionType.COMPRESSIVE, _civ(), 0.2, np.random.default_rng(3))
    labels = profile.segment_of_slice()
    start, stop = profile.lesion_slice_range
    assert len(set(labels[start:stop].tolist())) == 1
    assert int(labels[start]) == 0
    assert len(labels) == DEFAULT_GRID.shape[0]


def test_analytic_descriptor_vector_matches_the_field_list() -> None:
    profile = build_profile(DEFAULT_GRID, LesionType.COMPRESSIVE, _civ(), 0.2, np.random.default_rng(4))
    vector = analytic_descriptor_vector(profile)
    assert vector.shape == (21,)
    assert np.all(np.isfinite(vector))
    area = profile.unit_area()
    assert vector[0] == pytest.approx(float(np.min(area)), rel=1e-9)


def test_occlusive_profile_has_closed_slices_and_a_finite_encoding() -> None:
    profile = build_profile(DEFAULT_GRID, LesionType.OCCLUSIVE, _civ(), 0.2, np.random.default_rng(5))
    conductance = profile.unit_conductance()
    assert float(np.min(conductance)) == 0.0
    assert float(np.count_nonzero(conductance <= 0.0)) > 0
    # Algorithm 1 skips the non-opacified samples in the series sum and reports the
    # occlusion as a length, so a partly occluded segment still has a finite sum.
    vector = analytic_descriptor_vector(profile)
    assert np.all(np.isfinite(vector))
    assert profile.lesion_span_mm() > 0.0


def test_render_mask_covers_the_vessel_and_occludes_the_lesion() -> None:
    for lesion_type in LesionType:
        profile = build_profile(DEFAULT_GRID, lesion_type, _civ(), 0.2, np.random.default_rng(6))
        mask = render_mask(profile)
        assert int(mask.sum()) > 0
        start, stop = profile.lesion_slice_range
        if lesion_type is LesionType.OCCLUSIVE:
            assert int(mask[start:stop].sum()) == 0


def test_render_intensity_responds_to_the_phase() -> None:
    profile = build_profile(DEFAULT_GRID, LesionType.COMPRESSIVE, _civ(), 0.2, np.random.default_rng(7))
    direct = render_intensity(profile, AcquisitionPhase.DIRECT, "vendor_a", np.random.default_rng(8))
    delayed = render_intensity(profile, AcquisitionPhase.DELAYED_INDIRECT, "vendor_a", np.random.default_rng(8))
    mask = render_mask(profile)
    assert float(direct[mask].mean()) > float(delayed[mask].mean())


def test_generate_limb_records_agreement_and_adjudication() -> None:
    rng = np.random.default_rng(9)
    _, record = generate_limb("L1", "P1", Site.A, rng, lesion_type=LesionType.POST_THROMBOTIC)
    assert record.reader_types[0] in LesionType and record.reader_types[1] in LesionType
    assert record.adjudicated == (record.reader_types[0] != record.reader_types[1])
    assert record.reader_consensus is None or record.reader_consensus == record.reader_types[0]
    assert 0.0 <= record.stenosis_ratio <= 1.0
    assert record.descriptor_vector.shape == (21,)
    assert record.axis_ratio > 0.0


def test_generate_cohort_is_nested_and_reproducible() -> None:
    first = generate_cohort(60, seed=17, grid=DEFAULT_GRID)
    second = generate_cohort(60, seed=17, grid=DEFAULT_GRID)
    assert first.n_patients == second.n_patients
    assert [limb.limb_id for limb in first.limbs] == [limb.limb_id for limb in second.limbs]
    assert first.n_limbs > first.n_patients
    assert len({limb.patient_id for limb in first.limbs}) == first.n_patients
    counts = reference_type_counts(first.limbs)
    assert int(counts.sum()) == first.n_limbs
    assert all(count > 0 for count in counts)


def test_draw_lesion_type_respects_the_prevalence() -> None:
    rng = np.random.default_rng(11)
    draws = [draw_lesion_type(rng, (1.0, 0.0, 0.0, 0.0)) for _ in range(20)]
    assert set(draws) == {LesionType.COMPRESSIVE}


def test_cohort_subset_keeps_patient_nesting() -> None:
    cohort = generate_cohort(80, seed=19, grid=DEFAULT_GRID)
    subset = cohort.subset(Site.A)
    assert subset.n_patients <= cohort.n_patients
    assert all(limb.site is Site.A for limb in subset.limbs)
    assert subset.patient_index().shape[0] == subset.n_limbs


def test_patient_split_is_disjoint_and_deterministic() -> None:
    cohort = generate_cohort(120, seed=23, grid=DEFAULT_GRID)
    first = patient_level_split(cohort, tune_fraction=0.25, seed=5)
    second = patient_level_split(cohort, tune_fraction=0.25, seed=5)
    assert first.train_patients == second.train_patients
    assert not set(first.train_patients) & set(first.tune_patients)
    assert not set(first.all_development) & set(first.external_patients)
    train, tune, external = first.masks(cohort)
    assert int(train.sum()) + int(tune.sum()) + int(external.sum()) == cohort.n_limbs


def test_patient_split_rejects_a_degenerate_fraction() -> None:
    cohort = generate_cohort(20, seed=29, grid=DEFAULT_GRID)
    with pytest.raises(ValueError):
        patient_level_split(cohort, tune_fraction=0.0)


def test_crossfit_folds_are_disjoint_and_complete() -> None:
    patients = tuple(f"P{index:03d}" for index in range(23))
    folds = crossfit_folds(patients, 4, seed=3)
    flat = [value for fold in folds for value in fold]
    assert sorted(flat) == sorted(patients)
    assert len(set(flat)) == len(flat)
    with pytest.raises(ValueError):
        crossfit_folds(patients, 1)
    with pytest.raises(ValueError):
        crossfit_folds(patients[:2], 5)


def test_leave_one_site_out_and_accrual_halves() -> None:
    cohort = generate_cohort(90, seed=31, grid=DEFAULT_GRID)
    folds = leave_one_site_out(cohort)
    assert set(folds) == {"Site A", "Site B", "Site C"}
    for name, split in folds.items():
        held = {patient.site for patient in cohort.patients if patient.patient_id in set(split.external_patients)}
        assert held == {Site(name)}
    first, second = accrual_halves(cohort, seed=7)
    assert not set(first) & set(second)
    assert len(first) + len(second) == cohort.n_patients


def test_site_stratified_patients_and_strata() -> None:
    cohort = generate_cohort(60, seed=37, grid=DEFAULT_GRID)
    groups = site_stratified_patients(cohort, 0.5, seed=11)
    assert set(groups) == set(Site)
    labels = {stratum_of(limb) for limb in cohort.limbs}
    assert len(labels) > 1


def test_confusion_counts_start_at_type_one() -> None:
    reference = np.array([1, 2, 3, 4], dtype=np.int64)
    predicted = np.array([1, 3, 3, 4], dtype=np.int64)
    matrix = confusion_counts(predicted, reference)
    assert matrix.shape == (4, 4)
    assert matrix[0, 0] == 1
    assert matrix[1, 2] == 1
    assert int(matrix.sum()) == 4


def _civ():
    from conductance_ctv.data.schema import Segment

    return Segment.CIV
