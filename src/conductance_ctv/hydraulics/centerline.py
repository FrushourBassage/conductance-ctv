"""Centreline extraction and arc-length parametrisation of a venous segment.

The path runs from the iliocaval confluence to the common femoral bifurcation.
It is recovered slice by slice from the lumen mask, smoothed, and resampled at a
uniform arc-length step so that every downstream quantity (the perpendicular
section, the conductance at ``s_k``, the series sum) refers to the same grid.

Ref: Sec. 6.1 (the curve gamma), Sec. 6.5 and Algorithm 1 lines 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.hydraulics.grid import PlaneBasis, orthonormal_frame, resample_arclength
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

SEGMENT_NAMES: tuple[str, str, str] = ("CIV", "EIV", "CFV")

# Anatomical split of the confluence-to-bifurcation path. The manuscript names the
# three segments but does not state where their interfaces sit along the arc, so the
# fractions below are an engineering default exposed as ``path.segment_fractions``.
DEFAULT_SEGMENT_FRACTIONS: tuple[float, float, float, float] = (0.0, 0.40, 0.75, 1.0)


@dataclass(frozen=True)
class PathSamples:
    """A resampled centreline with its tangents and arc-length grid."""

    points: FloatArray
    tangents: FloatArray
    arclength: FloatArray

    def __post_init__(self) -> None:
        count = self.points.shape[0]
        if self.points.shape != (count, 3):
            raise ValueError("points must have shape (N, 3)")
        if self.tangents.shape != (count, 3):
            raise ValueError("tangents must have shape (N, 3)")
        if self.arclength.shape != (count,):
            raise ValueError("arclength must have shape (N,)")

    @property
    def total_length(self) -> float:
        return float(self.arclength[-1]) if self.arclength.size else 0.0

    @property
    def step(self) -> float:
        if self.arclength.size < 2:
            return 0.0
        return float(self.arclength[1] - self.arclength[0])

    def segment_of(self, fractions: tuple[float, float, float, float]) -> IntArray:
        """Assign each sample to CIV, EIV or CFV by its fractional arc position."""
        if len(fractions) != 4:
            raise ValueError("four fractions are required for three segments")
        if fractions[0] != 0.0 or abs(fractions[-1] - 1.0) > 1e-9:
            raise ValueError("segment fractions must span 0 to 1")
        ordered = np.asarray(fractions, dtype=np.float64)
        if np.any(np.diff(ordered) <= 0.0):
            raise ValueError("segment fractions must be strictly increasing")
        position = self.arclength / max(self.total_length, 1e-12)
        index = np.searchsorted(ordered, position, side="right") - 1
        return np.clip(index, 0, 2).astype(np.int64)

    def bounds_for(self, index: int, fractions: tuple[float, float, float, float]) -> tuple[int, int]:
        """Half-open sample range ``[start, stop)`` of segment ``index``."""
        labels = self.segment_of(fractions)
        members = np.flatnonzero(labels == index)
        if members.size == 0:
            return (0, 0)
        return (int(members[0]), int(members[-1] + 1))

    def frame(self, index: int) -> PlaneBasis:
        axis_u, axis_v = orthonormal_frame(self.tangents[index])
        return PlaneBasis(
            origin=self.points[index],
            axis_u=axis_u,
            axis_v=axis_v,
            arclength=float(self.arclength[index]),
        )

    def tube_mask(
        self, grid: tuple[tuple[float, float, float], tuple[int, int, int]], radius: float, indices: IntArray
    ) -> BoolArray:
        """Voxels within ``radius`` of the centreline at the given sample indices.

        Used for the 20 mm appearance tube around ``s*`` and for the 10 mm
        confluence cylinder of the segmentation loss. Each centre only touches
        its own bounding box, so the cost follows the tube and not the volume.
        """
        spacing, shape = grid
        if radius <= 0.0:
            raise ValueError("tube radius must be strictly positive")
        selected = self.points[np.asarray(indices, dtype=np.int64)]
        mask = np.zeros(shape, dtype=bool)
        if selected.size == 0:
            return mask
        span = np.asarray(spacing, dtype=np.float64)
        radius_sq = radius * radius
        for point in selected:
            centre = point / span
            low = np.maximum(np.floor(centre - radius / span).astype(np.int64), 0)
            high = np.minimum(np.ceil(centre + radius / span).astype(np.int64) + 1, np.asarray(shape))
            if np.any(high <= low):
                continue
            z = (np.arange(low[0], high[0], dtype=np.float64)) * span[0]
            y = (np.arange(low[1], high[1], dtype=np.float64)) * span[1]
            x = (np.arange(low[2], high[2], dtype=np.float64)) * span[2]
            gz, gy, gx = np.meshgrid(z, y, x, indexing="ij")
            local = (gz - point[0]) ** 2 + (gy - point[1]) ** 2 + (gx - point[2]) ** 2 <= radius_sq
            mask[low[0] : high[0], low[1] : high[1], low[2] : high[2]] |= local
        return mask


def centroid_path(mask: BoolArray, spacing_zyx: tuple[float, float, float]) -> FloatArray:
    """Slice-wise centroid of a lumen mask along the first array axis.

    Empty slices inherit the nearest populated centroid, and a fully empty mask
    raises rather than returning a fabricated path.
    """
    if mask.ndim != 3:
        raise ValueError("a venous lumen mask is a three-dimensional array")
    slices = int(mask.shape[0])
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    points = np.zeros((slices, 3), dtype=np.float64)
    populated = np.zeros(slices, dtype=bool)
    for index in range(slices):
        plane = mask[index]
        if not bool(plane.any()):
            continue
        rows, cols = np.nonzero(plane)
        points[index, 1] = float(rows.mean()) * spacing[1]
        points[index, 2] = float(cols.mean()) * spacing[2]
        populated[index] = True
    if not bool(populated.any()):
        raise ValueError("the mask contains no lumen along the path axis")
    indices = np.arange(slices, dtype=np.float64)
    for axis in range(3):
        points[:, axis] = np.interp(indices, indices[populated], points[populated, axis])
    points[:, 0] = indices * spacing[0]
    return points


def smooth_path(points: FloatArray, window: int = 5) -> FloatArray:
    """Moving-average smoothing with edge replication and pinned endpoints.

    The first and last samples are restored after the filter. They are the
    iliocaval confluence and the common femoral bifurcation, and a filter that
    moves them would silently shorten the path and shift every arc-length
    descriptor that is indexed against it.
    """
    if window < 1:
        raise ValueError("window must be at least one")
    if window == 1 or points.shape[0] < 3:
        return np.array(points, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)], axis=1)[
        : points.shape[0]
    ]
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def estimate_tangents(points: FloatArray) -> FloatArray:
    """Central-difference tangents with one-sided ends, normalised."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        raise ValueError("at least two path vertices are required")
    tangents = np.gradient(pts, axis=0)
    norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    unit: FloatArray = tangents / norm
    return unit


def build_path(
    mask: BoolArray,
    spacing_zyx: tuple[float, float, float],
    step: float,
    smooth_window: int = 5,
) -> PathSamples:
    """Centroid path, smoothed and resampled at a uniform arc-length step."""
    if step <= 0.0:
        raise ValueError("arc-length step must be strictly positive")
    raw = centroid_path(mask, spacing_zyx)
    smoothed = smooth_path(raw, window=smooth_window)
    points, _, arclength = resample_arclength(smoothed, None, step)
    tangents = estimate_tangents(points)
    if arclength.shape[0] != points.shape[0]:
        raise ValueError("arc-length grid and path vertices disagree")
    return PathSamples(points=points, tangents=tangents, arclength=arclength)


def smoothing_step_ratio(step: float, spacing_zyx: tuple[float, float, float]) -> float:
    """Arc-length step as a multiple of the coarsest voxel edge, for sanity checks."""
    return float(step) / max(float(np.max(np.asarray(spacing_zyx, dtype=np.float64))), 1e-12)
