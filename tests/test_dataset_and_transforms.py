"""Datasets, sampling and transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conductance_ctv.data.dataset import (
    SegmentationDataset,
    TypeDataset,
    build_case,
    build_cases,
    descriptor_matrix,
    descriptor_targets,
    label_vector,
)
from conductance_ctv.data.sampling import (
    APPEARANCE_TUBE_MM,
    CONFLUENCE_CYLINDER_MM,
    confluence_indices,
    confluence_weight_map,
    extract_patch,
    patch_box,
    sample_offsets,
    star_index_from_descriptors,
    tube_mask_around_star,
)
from conductance_ctv.data.transforms import (
    DescriptorStandardiser,
    SegmentationAugment,
    blur_through_plane,
    normalise_robust,
    random_contrast_phase,
    random_flip_axes,
    replace_non_finite,
    unwindow_intensity,
    window_intensity,
)


def test_build_case_produces_consistent_arrays(small_cohort, small_grid) -> None:
    case = build_case(small_cohort.limbs[0], grid=small_grid, seed=3)
    assert case.image.shape == small_grid.shape
    assert case.mask.shape == small_grid.shape
    assert case.descriptor.shape == (21,)
    assert 0 <= case.star_index < case.path.points.shape[0]
    assert case.star_point.shape == (3,)
    assert case.label in (0, 1, 2, 3)
    assert case.phase_code() in (0, 1, 2)


def test_build_cases_is_deterministic_and_cacheable(small_cohort, small_grid, tmp_path) -> None:
    first = build_cases(small_cohort, grid=small_grid, limit=4)
    second = build_cases(small_cohort, grid=small_grid, limit=4)
    assert [case.star_index for case in first] == [case.star_index for case in second]
    cached = build_cases(small_cohort, grid=small_grid, cache_dir=tmp_path, limit=4)
    assert [case.star_index for case in cached] == [case.star_index for case in first]
    assert any(tmp_path.iterdir())


def test_segmentation_dataset_returns_patch_and_weight(small_cases) -> None:
    dataset = SegmentationDataset(small_cases, patch_shape=(16, 32, 32))
    batch = dataset[0]
    assert batch["image"].shape == (1, 16, 32, 32)
    assert batch["mask"].shape == (1, 16, 32, 32)
    assert batch["weight"].shape == (1, 16, 32, 32)
    assert float(batch["weight"].max()) >= 1.0
    assert torch.isfinite(batch["image"]).all()
    assert int(batch["index"].item()) == 0


def test_segmentation_dataset_augmentation_keeps_the_shapes(small_cases) -> None:
    dataset = SegmentationDataset(
        small_cases, patch_shape=(16, 32, 32), augment=SegmentationAugment(noise_sigma=2.0), seed=5
    )
    batch = dataset[1]
    assert batch["image"].shape == (1, 16, 32, 32)
    assert batch["mask"].shape == (1, 16, 32, 32)


def test_type_dataset_standardises_descriptors(small_cases) -> None:
    standardiser = DescriptorStandardiser().fit(descriptor_matrix(small_cases))
    dataset = TypeDataset(small_cases, patch_shape=(16, 32, 32), standardiser=standardiser)
    batch = dataset[0]
    assert batch["descriptor"].shape == (21,)
    assert batch["descriptor_raw"].shape == (21,)
    assert batch["appearance"].shape == (1, 16, 32, 32)
    assert batch["site"].dtype == torch.long
    assert dataset.descriptors().shape == (len(small_cases), 21)
    assert dataset.appearance_patches().shape == (len(small_cases), 16, 32, 32)


def test_descriptor_targets_carry_the_patient_index(small_cases) -> None:
    descriptors, labels, patients = descriptor_targets(small_cases, [case.limb.patient_id for case in small_cases])
    assert descriptors.shape == (len(small_cases), 21)
    assert labels.shape == (len(small_cases),)
    assert np.unique(patients).size < len(small_cases)
    assert label_vector(small_cases).shape == (len(small_cases),)


def test_appearance_tube_radius_and_span(small_cases) -> None:
    case = small_cases[0]
    indices = sample_offsets(case.path.step, 0.5 * APPEARANCE_TUBE_MM)
    assert indices >= 1
    tube = tube_mask_around_star(case.grid, case.path, case.star_index)
    assert tube.dtype == bool
    assert int(tube.sum()) > 0


def test_confluence_cylinder_and_weight_map(small_cases) -> None:
    case = small_cases[0]
    indices = confluence_indices(case.path)
    assert indices.size >= 1
    assert len(indices) == indices[-1] - indices[0] + 1
    weights = confluence_weight_map(case.grid, case.path)
    assert float(weights.min()) == 1.0
    assert float(weights.max()) > 1.0
    assert CONFLUENCE_CYLINDER_MM == 10.0


def test_patch_box_and_extraction(small_cases) -> None:
    case = small_cases[0]
    box = patch_box(case.star_point, case.grid.spacing_zyx, (8, 8, 8), case.grid.shape)
    patch = extract_patch(case.image, box, (8, 8, 8))
    assert patch.shape == (8, 8, 8)
    inside = box.pad_widths((8, 8, 8))
    assert all(left >= 0 and right >= 0 for left, right in inside)


def test_star_index_lookup_is_nearest() -> None:
    from conductance_ctv.hydraulics.centerline import PathSamples

    path = PathSamples(
        points=np.zeros((5, 3), dtype=np.float64),
        tangents=np.tile(np.array([[1.0, 0.0, 0.0]]), (5, 1)),
        arclength=np.arange(5, dtype=np.float64),
    )
    assert star_index_from_descriptors(2.4, path) == 2
    assert star_index_from_descriptors(3.6, path) == 4


def test_window_intensity_round_trips() -> None:
    volume = np.linspace(-200.0, 600.0, 64).reshape(4, 4, 4)
    scaled = window_intensity(volume)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    restored = unwindow_intensity(scaled)
    inside = (volume >= -120.0) & (volume <= 520.0)
    assert np.allclose(restored[inside], volume[inside], atol=1e-6)
    with pytest.raises(ValueError):
        window_intensity(volume, high=-300.0)


def test_normalise_robust_is_bounded() -> None:
    values = np.concatenate([np.zeros(50), np.arange(50, dtype=np.float64)])
    out = normalise_robust(values)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(np.max(np.abs(normalise_robust(np.zeros(10))))) == 0.0


def test_random_transforms_preserve_shape(rng) -> None:
    volume = rng.normal(size=(8, 8, 8))
    assert random_flip_axes(volume, (1, 2), rng).shape == volume.shape
    assert random_contrast_phase(volume, rng).shape == volume.shape
    assert blur_through_plane(volume, 1.0).shape == volume.shape
    assert blur_through_plane(volume, 0.0).shape == volume.shape


def test_segmentation_augment_moves_image_and_mask_together() -> None:
    image = np.zeros((8, 8, 8), dtype=np.float64)
    image[2:4, 2:4, 2:4] = 1.0
    mask = image > 0.5
    augment = SegmentationAugment(flip_axes=(1, 2), noise_sigma=0.0, phase_gain_range=(1.0, 1.0))
    for seed in range(8):
        out_image, out_mask = augment(image, mask, np.random.default_rng(seed))
        assert out_image.shape == image.shape
        assert out_mask.shape == mask.shape
        assert out_mask.dtype == bool
        assert int(out_mask.sum()) == int(mask.sum())
        # The flips are drawn before any intensity transform, so replaying them on
        # the mask alone has to reproduce the mask the augment returned.
        replay = np.random.default_rng(seed)
        expected = np.array(mask)
        for axis in augment.flip_axes:
            if float(replay.random()) < 0.5:
                expected = np.ascontiguousarray(np.flip(expected, axis=axis))
        assert np.array_equal(out_mask, expected)


def test_descriptor_standardiser_round_trip() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(3.0, 2.0, size=(64, 21))
    standardiser = DescriptorStandardiser().fit(features)
    scaled = standardiser.transform(features)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    restored = standardiser.inverse_transform(scaled)
    assert np.allclose(restored, features, atol=1e-9)
    with pytest.raises(ValueError):
        standardiser.transform(np.zeros((4, 5)))
    record = standardiser.record()
    assert len(record["mean"]) == 21


def test_standardiser_on_constant_columns_does_not_divide_by_zero() -> None:
    features = np.ones((16, 3), dtype=np.float64) * 4.0
    standardiser = DescriptorStandardiser().fit(features)
    scaled = standardiser.transform(features)
    assert np.all(np.isfinite(scaled))
    assert np.allclose(scaled, 0.0)


def test_replace_non_finite_handles_the_occluded_encoding() -> None:
    features = np.array([1.0, np.inf, -np.inf, np.nan], dtype=np.float64)
    out = replace_non_finite(features, value=1.0)
    assert np.all(np.isfinite(out))
    assert out[1] == 1.0
