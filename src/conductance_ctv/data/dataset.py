"""Datasets that turn a cohort record into the tensors the two models consume.

Two dataset flavours share one case builder: the segmentation dataset returns an
image patch and its lumen mask under the confluence weight map, and the type
dataset returns the descriptor vector together with the appearance patch centred
on ``s*``. Rendering a case is the expensive step, so it is cached on disk keyed
by the limb identifier and the grid it was rendered for.

Ref: Sec. 6.4 (segmentation inputs), Sec. 6.6 (the two branches of the type model).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from conductance_ctv.data.fixture import (
    DEFAULT_GRID,
    LimbProfile,
    build_profile,
    render_intensity,
    render_mask,
)
from conductance_ctv.data.sampling import confluence_weight_map, star_index_from_descriptors
from conductance_ctv.data.schema import CohortRecord, GridSpec, LimbRecord, Site
from conductance_ctv.data.transforms import DescriptorStandardiser, SegmentationAugment, window_intensity
from conductance_ctv.hydraulics.centerline import PathSamples, build_path
from conductance_ctv.hydraulics.descriptors import DESCRIPTOR_FIELDS, SEGMENT_NAMES
from conductance_ctv.utils.io import read_json, write_json, write_npz
from conductance_ctv.utils.numeric import as_triple_int
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

DEFAULT_PATCH_SHAPE: tuple[int, int, int] = (32, 64, 64)
SITE_ORDER: tuple[Site, ...] = (Site.A, Site.B, Site.C)


@dataclass(frozen=True)
class LimbCase:
    """One rendered limb: intensities, lumen mask, centreline and its record."""

    limb: LimbRecord
    grid: GridSpec
    image: FloatArray
    mask: BoolArray
    path: PathSamples
    star_index: int

    @property
    def descriptor(self) -> FloatArray:
        return np.asarray(self.limb.descriptor_vector, dtype=np.float64)

    @property
    def label(self) -> int:
        return int(self.limb.reference_type) - 1

    @property
    def star_point(self) -> FloatArray:
        point: FloatArray = self.path.points[self.star_index]
        return point

    def phase_code(self) -> int:
        order = ("direct", "early_indirect", "delayed_indirect")
        return order.index(str(self.limb.phase))


def _profile_from_record(limb: LimbRecord, grid: GridSpec, seed: int) -> LimbProfile:
    """Rebuild the generating profile of a recorded limb exactly.

    The axis ratio is taken from the record rather than re-derived from the
    stenosis ratio, so a rendered case reproduces the anatomy its recorded
    descriptors were computed on.
    """
    rng = np.random.default_rng(seed)
    return build_profile(
        grid,
        limb.reference_type,
        limb.lesion_segment,
        float(limb.axis_ratio),
        rng,
        collateral_grade=limb.collateral_grade,
    )


def build_case(limb: LimbRecord, grid: GridSpec = DEFAULT_GRID, seed: int = 0, star_offset: int = 0) -> LimbCase:
    """Render one limb into intensities, a lumen mask and a centreline."""
    profile = _profile_from_record(limb, grid, seed)
    mask = render_mask(profile)
    rng = np.random.default_rng(seed + 1)
    image = render_intensity(profile, limb.phase, limb.vendor, rng)
    path = build_path(mask, grid.spacing_zyx, max(grid.spacing_zyx[0], 1.0))
    star = star_index_from_descriptors(
        float(limb.descriptor_vector[SEGMENT_NAMES.index("CIV") * len(DESCRIPTOR_FIELDS) + 1]),
        path,
    )
    star = int(np.clip(star + star_offset, 0, path.points.shape[0] - 1))
    return LimbCase(limb=limb, grid=grid, image=image, mask=mask, path=path, star_index=star)


def build_cases(
    cohort: CohortRecord,
    grid: GridSpec = DEFAULT_GRID,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[LimbCase]:
    """Render every limb of a cohort, optionally reusing an on-disk cache."""
    limbs = list(cohort.limbs)
    if limit is not None:
        limbs = limbs[:limit]
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    cases: list[LimbCase] = []
    for index, limb in enumerate(limbs):
        key = f"{limb.limb_id}_{grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]}"
        mask_path = cache / f"{key}_mask.npz" if cache else None
        json_path = cache / f"{key}.json" if cache else None
        if mask_path is not None and json_path is not None and mask_path.exists() and json_path.exists():
            payload = np.load(mask_path)
            meta = read_json(json_path)
            mask = payload["mask"].astype(bool)
            path = PathSamples(
                points=np.asarray(payload["points"], dtype=np.float64),
                tangents=np.asarray(payload["tangents"], dtype=np.float64),
                arclength=np.asarray(payload["arclength"], dtype=np.float64),
            )
            cases.append(
                LimbCase(
                    limb=limb,
                    grid=grid,
                    image=np.asarray(payload["image"], dtype=np.float64),
                    mask=mask,
                    path=path,
                    star_index=int(meta["star_index"]),
                )
            )
            continue
        case = build_case(limb, grid=grid, seed=cohort.seed + index)
        if cache is not None and mask_path is not None and json_path is not None:
            write_npz(
                mask_path,
                {
                    "mask": case.mask,
                    "image": case.image,
                    "points": case.path.points,
                    "tangents": case.path.tangents,
                    "arclength": case.path.arclength,
                },
            )
            write_json(json_path, {"star_index": case.star_index, "limb_id": limb.limb_id})
        cases.append(case)
    return cases


def descriptor_targets(cases: Sequence[LimbCase], patient_ids: Sequence[str]) -> tuple[FloatArray, IntArray, IntArray]:
    """Stacked descriptors, labels and patient indices of a case list.

    ``patient_ids`` fixes the order the patient index is taken in, so two limbs
    of one patient receive the same index and the clustered resampling of
    Sec. 6.8 can propagate a draw to both of them.
    """
    lookup = {value: index for index, value in enumerate(dict.fromkeys(patient_ids))}
    descriptors = np.stack([case.descriptor for case in cases], axis=0)
    labels = np.array([case.label for case in cases], dtype=np.int64)
    patients = np.array([lookup[case.limb.patient_id] for case in cases], dtype=np.int64)
    return descriptors, labels, patients


class IndexSet(Dataset[dict[str, torch.Tensor]]):
    """A view over a case list, returning indices only.

    Keeping the case list outside the dataset lets every dataset flavour share
    one rendering pass, which matters because rendering dominates the run time.
    """

    def __init__(self, cases: Sequence[LimbCase], indices: Sequence[int] | None = None) -> None:
        self.cases = list(cases)
        self.indices = list(range(len(self.cases))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def case(self, position: int) -> LimbCase:
        return self.cases[self.indices[position]]


class SegmentationDataset(IndexSet):
    """Image and lumen mask patches with the confluence weighting already applied."""

    def __init__(
        self,
        cases: Sequence[LimbCase],
        indices: Sequence[int] | None = None,
        patch_shape: tuple[int, int, int] = DEFAULT_PATCH_SHAPE,
        augment: SegmentationAugment | None = None,
        seed: int = 0,
        inside_weight: float = 3.0,
    ) -> None:
        super().__init__(cases, indices)
        self.patch_shape = patch_shape
        self.augment = augment
        self.seed = seed
        # One means a uniform voxel cost, which is the ablation of Table 2; any
        # value above one is the confluence weighting of Sec. 6.4.
        self.inside_weight = float(inside_weight)
        self._weights: dict[int, FloatArray] = {}

    def weight_map(self, position: int) -> FloatArray:
        if position not in self._weights:
            case = self.case(position)
            self._weights[position] = confluence_weight_map(case.grid, case.path, inside_weight=self.inside_weight)
        return self._weights[position]

    def patch_origin(self, position: int) -> tuple[int, int, int]:
        case = self.case(position)
        centre = np.asarray(case.star_point, dtype=np.float64) / np.asarray(case.grid.spacing_zyx)
        half = (np.asarray(self.patch_shape, dtype=np.float64) - 1.0) / 2.0
        start = np.rint(centre - half).astype(np.int64)
        volume = np.asarray(case.grid.shape, dtype=np.int64)
        clipped = np.clip(start, 0, np.maximum(volume - np.asarray(self.patch_shape), 0))
        return as_triple_int([int(value) for value in clipped])

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        case = self.case(position)
        start = np.asarray(self.patch_origin(position), dtype=np.int64)
        stop = np.minimum(start + np.asarray(self.patch_shape), np.asarray(case.grid.shape))
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
        image_patch = np.zeros(self.patch_shape, dtype=np.float64)
        mask_patch = np.zeros(self.patch_shape, dtype=bool)
        weight_patch = np.ones(self.patch_shape, dtype=np.float64)
        present = tuple(slice(0, int(b - a)) for a, b in zip(start, stop, strict=True))
        image_patch[present] = case.image[slices]
        mask_patch[present] = case.mask[slices]
        weight_patch[present] = self.weight_map(position)[slices]
        if self.augment is not None:
            rng = np.random.default_rng(self.seed + position)
            image_patch, mask_patch = self.augment(image_patch, mask_patch, rng)
        scaled = window_intensity(image_patch)
        return {
            "image": torch.from_numpy(scaled[None].astype(np.float32)),
            "mask": torch.from_numpy(mask_patch[None].astype(np.float32)),
            "weight": torch.from_numpy(weight_patch[None].astype(np.float32)),
            "index": torch.tensor(position, dtype=torch.long),
        }


class TypeDataset(IndexSet):
    """Descriptor vector plus the appearance patch centred on ``s*``."""

    def __init__(
        self,
        cases: Sequence[LimbCase],
        indices: Sequence[int] | None = None,
        patch_shape: tuple[int, int, int] = DEFAULT_PATCH_SHAPE,
        standardiser: DescriptorStandardiser | None = None,
        augment: SegmentationAugment | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(cases, indices)
        self.patch_shape = patch_shape
        self.standardiser = standardiser if standardiser is not None else DescriptorStandardiser()
        self.augment = augment
        self.seed = seed

    def descriptors(self) -> FloatArray:
        return np.stack([self.case(position).descriptor for position in range(len(self))], axis=0)

    def appearance_patches(self) -> FloatArray:
        """All appearance patches of the view, stacked, for the cached-feature path."""
        half = (np.asarray(self.patch_shape, dtype=np.float64) - 1.0) / 2.0
        out = np.zeros((len(self), *self.patch_shape), dtype=np.float64)
        for position in range(len(self)):
            case = self.case(position)
            centre = np.asarray(case.star_point, dtype=np.float64) / np.asarray(case.grid.spacing_zyx)
            start = np.clip(
                np.rint(centre - half).astype(np.int64),
                0,
                np.maximum(np.asarray(case.grid.shape, dtype=np.int64) - np.asarray(self.patch_shape), 0),
            )
            stop = np.minimum(start + np.asarray(self.patch_shape), np.asarray(case.grid.shape))
            slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
            present = tuple(slice(0, int(b - a)) for a, b in zip(start, stop, strict=True))
            out[position][present] = case.image[slices]
        return out

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        case = self.case(position)
        span = np.asarray(case.grid.spacing_zyx, dtype=np.float64)
        half = (np.asarray(self.patch_shape, dtype=np.float64) - 1.0) / 2.0
        centre = np.asarray(case.star_point, dtype=np.float64) / span
        start = np.rint(centre - half).astype(np.int64)
        start = np.clip(
            start,
            0,
            np.maximum(np.asarray(case.grid.shape, dtype=np.int64) - np.asarray(self.patch_shape), 0),
        )
        stop = np.minimum(start + np.asarray(self.patch_shape), np.asarray(case.grid.shape))
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
        present = tuple(slice(0, int(b - a)) for a, b in zip(start, stop, strict=True))
        patch = np.zeros(self.patch_shape, dtype=np.float64)
        patch[present] = case.image[slices]
        if self.augment is not None:
            rng = np.random.default_rng(self.seed + position)
            patch, _ = self.augment(patch, np.zeros(self.patch_shape, dtype=bool), rng)
        descriptor = self.standardiser.transform(case.descriptor[None])[0]
        return {
            "descriptor": torch.from_numpy(np.nan_to_num(descriptor).astype(np.float32)),
            "descriptor_raw": torch.from_numpy(case.descriptor.astype(np.float32)),
            "appearance": torch.from_numpy(window_intensity(patch)[None].astype(np.float32)),
            "label": torch.tensor(case.label, dtype=torch.long),
            "site": torch.tensor(SITE_ORDER.index(case.limb.site), dtype=torch.long),
            "index": torch.tensor(position, dtype=torch.long),
        }


def descriptor_matrix(cases: Sequence[LimbCase]) -> FloatArray:
    return np.stack([case.descriptor for case in cases], axis=0)


def label_vector(cases: Sequence[LimbCase]) -> np.ndarray:
    return np.array([case.label for case in cases], dtype=np.int64)
