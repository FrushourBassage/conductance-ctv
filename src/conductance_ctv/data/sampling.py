"""Sampling the two volumes of interest around the descriptor location ``s*``.

The type model reads a 20 mm tube of backbone features centred on ``s*`` and the
segmentation loss concentrates its weight in a 10 mm cylinder around the
iliocaval confluence. Both are geometric selections on the centreline, so they
live here rather than in the datasets.

Ref: Sec. 6.6 (20 mm tube at ``s*``), Sec. 6.4 (10 mm confluence cylinder).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.data.schema import GridSpec
from conductance_ctv.hydraulics.centerline import PathSamples
from conductance_ctv.utils.numeric import as_triple_int
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

APPEARANCE_TUBE_MM = 20.0
CONFLUENCE_CYLINDER_MM = 10.0


@dataclass(frozen=True)
class PatchBox:
    """A clipped index box inside a volume, with its padding reported."""

    start: IntArray
    stop: IntArray
    shape: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(int(self.start[0]), int(self.stop[0])),
            slice(int(self.start[1]), int(self.stop[1])),
            slice(int(self.start[2]), int(self.stop[2])),
        )

    def pad_widths(self, target: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
        missing = [max(0, want - have) for want, have in zip(target, self.shape, strict=True)]
        return tuple((value // 2, value - value // 2) for value in missing)


def indices_within_arclength(path: PathSamples, centre_index: int, radius: float) -> IntArray:
    """Path samples whose arc length lies within ``radius`` of a sample."""
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive")
    centre = float(path.arclength[centre_index])
    keep = np.abs(path.arclength - centre) <= radius
    return np.flatnonzero(keep).astype(np.int64)


def appearance_tube_indices(path: PathSamples, star_index: int, radius: float = APPEARANCE_TUBE_MM) -> IntArray:
    """Path samples of the appearance tube around ``s*``.

    Ref: Sec. 6.6 ("a backbone feature vector gathered in a 20 mm tube with a
    focus on s*").
    """
    return indices_within_arclength(path, star_index, 0.5 * radius)


def confluence_indices(path: PathSamples, radius: float = CONFLUENCE_CYLINDER_MM) -> IntArray:
    """Path samples inside the confluence cylinder, where the lumen collapses.

    Ref: Sec. 6.4 and Algorithm 1 line 4 context.
    """
    return indices_within_arclength(path, 0, radius)


def patch_box(
    centre_zyx: FloatArray,
    spacing_zyx: tuple[float, float, float],
    shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
) -> PatchBox:
    """Index box of a fixed-size patch centred on a physical point, clipped to the volume."""
    span = np.asarray(spacing_zyx, dtype=np.float64)
    half = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    centre_voxel = np.asarray(centre_zyx, dtype=np.float64) / span
    start = np.rint(centre_voxel - half).astype(np.int64)
    volume = np.asarray(volume_shape, dtype=np.int64)
    clipped_start = np.clip(start, 0, np.maximum(volume - np.asarray(shape, dtype=np.int64), 0))
    clipped_stop = np.minimum(clipped_start + np.asarray(shape, dtype=np.int64), volume)
    clipped = clipped_stop - clipped_start
    return PatchBox(start=clipped_start, stop=clipped_stop, shape=as_triple_int([int(value) for value in clipped]))


def extract_patch(volume: FloatArray, box: PatchBox, target: tuple[int, int, int], fill: float = 0.0) -> FloatArray:
    """Copy a clipped patch out of a volume, zero-padding to the target shape."""
    patch = np.asarray(volume[box.slices], dtype=np.float64)
    pads = box.pad_widths(target)
    if any(left or right for left, right in pads):
        patch = np.pad(patch, pads, mode="constant", constant_values=fill)
    return patch


def confluence_weight_map(
    grid: GridSpec,
    path: PathSamples,
    cylinder_radius: float = CONFLUENCE_CYLINDER_MM,
    inside_weight: float = 3.0,
) -> FloatArray:
    """Per-voxel loss weights that concentrate on the confluence cylinder.

    The lumen collapses at the confluence, which is also where a uniform voxel
    loss is least informative about the minimum-area region, so the weight there
    is raised. The value of the raise is an engineering default exposed as
    ``loss.confluence_inside_weight``; the manuscript states only that the weight
    is higher inside the cylinder than outside it.

    Ref: Sec. 6.4 ("we selected the confluence-weighted cost instead of the
    uniform voxel cost").
    """
    if cylinder_radius <= 0.0:
        raise ValueError("cylinder radius must be strictly positive")
    if inside_weight < 1.0:
        raise ValueError("the inside weight must be at least one")
    selected = confluence_indices(path, cylinder_radius)
    inside = path.tube_mask((grid.spacing_zyx, grid.shape), cylinder_radius, selected)
    weights = np.ones(grid.shape, dtype=np.float64)
    weights[inside] = float(inside_weight)
    return weights


def tube_mask_around_star(
    grid: GridSpec, path: PathSamples, star_index: int, radius: float = APPEARANCE_TUBE_MM
) -> BoolArray:
    """Voxels of the appearance tube, used to mask the backbone feature gather."""
    selected = appearance_tube_indices(path, star_index, radius)
    return path.tube_mask((grid.spacing_zyx, grid.shape), 0.5 * radius, selected)


def star_index_from_descriptors(descriptor_arclength: float, path: PathSamples) -> int:
    """Nearest path sample to the ``s*`` recorded in a limb descriptor."""
    return int(np.argmin(np.abs(path.arclength - float(descriptor_arclength))))


def sample_offsets(step: float, radius: float) -> int:
    """Number of path samples either side of ``s*`` within a radius."""
    if step <= 0.0:
        raise ValueError("step must be strictly positive")
    return int(np.floor(radius / step))
