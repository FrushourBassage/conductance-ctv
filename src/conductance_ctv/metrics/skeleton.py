"""Centreline extraction used by the centreline Dice metric.

The centreline of a tubular segmentation is recovered as the medial ridge of the
exact Euclidean distance transform: a voxel belongs to the centreline when it is
a local maximum of the distance to the nearest lumen-free voxel along at least
two of the three axes. On a straight trunk this yields a thin line down the
middle of the lumen, which is the object the centreline Dice of Table 1 is
defined against.

The manuscript specifies centreline Dice as a metric and does not fix the
skeletonisation algorithm, so the ridge detector is this release's choice of
centreline extractor. The property the manuscript attributes to the metric, that
it exceeds voxel Dice on a single-trunk vessel, is checked in the test suite.

Ref: Table 1 (clDice column); Sec. 2.3 (centreline overlap for thin vessels).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from conductance_ctv.utils.types import BoolArray, FloatArray


def _squared_edt_axis(field: FloatArray, spacing: float) -> FloatArray:
    """Exact one-dimensional squared distance transform along the last axis.

    Each output entry is the minimum over the source positions of the squared
    separation plus the source value, which is the lower envelope of a family of
    parabolas. It is evaluated directly: the axis length is small compared with
    the number of lines, so looping over source positions and vectorising over
    lines is the cheaper of the two loops.
    """
    lines, length = field.shape
    positions = np.arange(length, dtype=np.float64) * spacing
    result = np.array(field, dtype=np.float64, copy=True)
    for source in range(length):
        separation = (positions - positions[source]) ** 2
        result = np.minimum(result, field[:, source][:, None] + separation[None, :])
    return result


def _seed_cost(occupancy: BoolArray) -> FloatArray:
    """Zero on lumen-free voxels and infinity inside the lumen."""
    return np.where(occupancy, np.inf, 0.0)


def distance_transform(mask: BoolArray, spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> FloatArray:
    """Exact Euclidean distance from every voxel to the nearest lumen-free voxel.

    The transform is separable, so it is three successive one-dimensional
    transforms on the squared distance.
    """
    if mask.ndim != 3:
        raise ValueError("a lumen mask is a three-dimensional array")
    if any(value <= 0.0 for value in spacing_zyx):
        raise ValueError("spacing must be strictly positive")
    squared = _seed_cost(mask)
    for axis in range(3):
        moved = np.moveaxis(squared, axis, -1)
        shape = moved.shape
        moved = moved.reshape(-1, shape[-1])
        moved = _squared_edt_axis(moved, float(spacing_zyx[axis]))
        squared = np.moveaxis(moved.reshape(shape), -1, axis)
    return np.sqrt(np.maximum(squared, 0.0))


def ridge_skeleton(
    mask: BoolArray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axes_required: int = 2,
) -> BoolArray:
    """Medial ridge of a binary lumen mask.

    A voxel belongs to the ridge when the distance field is strictly greater than
    its two neighbours along at least ``axes_required`` axes. A strict comparison
    matters: along the axis of a straight trunk the field is constant, so a
    non-strict test would award that axis to every voxel and flood the ridge.

    The distance field of a digital lumen has plateaus, so a trunk of radius ``r``
    yields a ridge whose radial extent is the plateau width of the field rather
    than a single voxel. That width is reported by ``skeleton_redundancy`` rather
    than being thinned away, because thinning a plateau by index order would collapse
    an entire trunk into one voxel.
    """
    if not bool(mask.any()):
        return np.zeros(mask.shape, dtype=bool)
    if not 1 <= axes_required <= 3:
        raise ValueError("between one and three axes must be required")
    field = distance_transform(mask, spacing_zyx)
    votes = np.zeros(mask.shape, dtype=np.int64)
    for axis in range(3):
        high = np.roll(field, 1, axis=axis)
        low = np.roll(field, -1, axis=axis)
        local = (field > high) & (field > low)
        edge: list[Any] = [slice(None)] * 3
        edge[axis] = 0
        local[tuple(edge)] = False
        edge[axis] = -1
        local[tuple(edge)] = False
        votes += local.astype(np.int64)
    ridge = mask & (votes >= axes_required)
    if bool(ridge.any()):
        ridge_mask: BoolArray = ridge
        return ridge_mask
    fallback = mask & (votes == int(votes.max())) & (votes > 0)
    if bool(fallback.any()):
        fallback_mask: BoolArray = fallback
        return fallback_mask
    last_resort: BoolArray = mask & (field == float(field.max()))
    return last_resort


def skeleton_voxels(skeleton: BoolArray) -> int:
    """Number of voxels of a skeleton, the count the centreline Dice dilates over."""
    return int(np.count_nonzero(skeleton))


def radial_thickness(mask: BoolArray, spacing_zyx: tuple[float, float, float]) -> float:
    """Diameter of the largest disc inscribed in a lumen, in mm.

    Twice the largest distance-transform value inside the lumen. The largest value
    is the inscribed radius; a median would instead report the typical distance of
    a lumen voxel from the wall, which for a square cross-section is half the
    inscribed radius.
    """
    if not bool(mask.any()):
        return 0.0
    field = distance_transform(mask, spacing_zyx)
    return float(2.0 * np.max(field[mask]))
