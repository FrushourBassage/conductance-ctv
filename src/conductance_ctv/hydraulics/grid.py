"""Cross-section geometry on a rectangular grid.

A lumen cross-section is a binary occupancy array with an in-plane spacing.
Everything downstream (Dirichlet solve, area, channel count) works on this
representation, so the mask is the single source of geometric truth.

Ref: Sec. 6.1 (the curve gamma and the perpendicular sections Omega(s)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

_MIN_AXIS_EPS = 1e-9


@dataclass(frozen=True)
class CrossSection:
    """Binary occupancy of the lumen in a plane, with in-plane spacing in mm.

    ``spacing_uv`` is the physical size of one array cell along each array axis,
    so the section area is ``occupancy.sum() * spacing_uv[0] * spacing_uv[1]``
    regardless of grid orientation.
    """

    occupancy: BoolArray
    spacing_uv: tuple[float, float]

    def __post_init__(self) -> None:
        if self.occupancy.ndim != 2:
            raise ValueError("a cross-section is a two-dimensional occupancy array")
        if min(self.spacing_uv) <= 0.0:
            raise ValueError("in-plane spacing must be strictly positive")

    @property
    def cell_area(self) -> float:
        return float(self.spacing_uv[0] * self.spacing_uv[1])

    @property
    def area(self) -> float:
        return float(self.occupancy.sum()) * self.cell_area

    @property
    def interface_edges(self) -> int:
        """Number of occupied/unoccupied cell interfaces along both grid axes."""
        occ = self.occupancy
        count = int(np.count_nonzero(occ[0, :])) + int(np.count_nonzero(occ[-1, :]))
        count += int(np.count_nonzero(occ[:, 0])) + int(np.count_nonzero(occ[:, -1]))
        count += int(np.count_nonzero(occ[:-1, :] & ~occ[1:, :]))
        count += int(np.count_nonzero(occ[1:, :] & ~occ[:-1, :]))
        count += int(np.count_nonzero(occ[:, :-1] & ~occ[:, 1:]))
        count += int(np.count_nonzero(occ[:, 1:] & ~occ[:, :-1]))
        return count

    @property
    def effective_area(self) -> float:
        """Lumen area under the half-cell boundary convention.

        A cell marked as lumen reaches half a cell beyond its centre, so the
        continuous domain the five-point stencil discretises is the occupied set
        grown by half a cell in every direction. Its area is the occupancy area
        plus half the occupied/unoccupied interface length per unit spacing,

            A = (N + L / 2) * h^2,

        with ``L`` the interface-edge count. This is the area the conductance of
        the same discretisation corresponds to, and it is what keeps the
        Polya-Szego ratio of a section free of a first-order grid bias.
        """
        return (float(self.occupancy.sum()) + 0.5 * self.interface_edges) * self.cell_area

    @property
    def is_empty(self) -> bool:
        return not bool(self.occupancy.any())

    @property
    def width(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def height(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        rows = np.flatnonzero(self.occupancy.any(axis=1))
        cols = np.flatnonzero(self.occupancy.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return (0, 0, 0, 0)
        return (int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1))

    def cropped(self, pad: int = 1) -> CrossSection:
        """Trim to the bounding box, keeping ``pad`` cells of Dirichlet margin."""
        if self.is_empty:
            return self
        r0, r1, c0, c1 = self.bounding_box
        r0 = max(0, r0 - pad)
        c0 = max(0, c0 - pad)
        r1 = min(self.height, r1 + pad)
        c1 = min(self.width, c1 + pad)
        return CrossSection(self.occupancy[r0:r1, c0:c1], self.spacing_uv)

    def label_map(self) -> tuple[IntArray, int]:
        """Label the 4-connected components; returns (labels, count)."""
        return _connected_components(self.occupancy)

    def component_sizes(self) -> list[int]:
        _, counts = np.unique(self.label_map()[0][self.occupancy], return_counts=True)
        return [int(value) for value in counts]


@dataclass(frozen=True)
class PlaneBasis:
    """An oriented orthonormal frame sampled from a venous centreline."""

    origin: FloatArray
    axis_u: FloatArray
    axis_v: FloatArray
    arclength: float = 0.0
    segment_index: int = 0
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("origin", "axis_u", "axis_v"):
            vector = np.asarray(getattr(self, name), dtype=np.float64)
            if vector.shape != (3,):
                raise ValueError(f"{name} must be a three-vector")
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be finite")
        normal = np.cross(self.axis_u, self.axis_v)
        if abs(float(np.linalg.norm(normal))) < _MIN_AXIS_EPS:
            raise ValueError("axis_u and axis_v must not be parallel")

    @property
    def normal(self) -> FloatArray:
        return np.cross(self.axis_u, self.axis_v)


def orthonormal_frame(tangent: FloatArray, reference: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
    """Build two unit vectors spanning the plane perpendicular to ``tangent``."""
    t = np.asarray(tangent, dtype=np.float64)
    norm = float(np.linalg.norm(t))
    if norm < _MIN_AXIS_EPS:
        raise ValueError("tangent must be non-degenerate")
    t = t / norm
    if reference is None:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if abs(float(np.dot(ref, t))) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(ref, t)
    u = u / float(np.linalg.norm(u))
    v = np.cross(t, u)
    return u, v / float(np.linalg.norm(v))


def section_from_volume(
    mask: BoolArray,
    spacing_zyx: tuple[float, float, float],
    basis: PlaneBasis,
    out_shape: tuple[int, int],
    out_spacing: float,
) -> CrossSection:
    """Resample an oblique section out of a volume by nearest-neighbour lookup.

    ``mask`` is indexed ``[z, y, x]``. Out-of-volume samples count as lumen-free,
    which matches the Dirichlet convention for the flow problem.
    """
    nz, ny, nx = mask.shape
    out_h, out_w = out_shape
    du, dv = out_spacing, out_spacing
    iu = (np.arange(out_h, dtype=np.float64) - (out_h - 1) / 2.0) * du
    iv = (np.arange(out_w, dtype=np.float64) - (out_w - 1) / 2.0) * dv
    grid_u, grid_v = np.meshgrid(iu, iv, indexing="ij")
    points = (
        basis.origin[None, None, :]
        + grid_u[:, :, None] * basis.axis_u[None, None, :]
        + grid_v[:, :, None] * basis.axis_v[None, None, :]
    )
    dz, dy, dx = spacing_zyx
    zi = np.rint(points[:, :, 0] / dz).astype(np.int64)
    yi = np.rint(points[:, :, 1] / dy).astype(np.int64)
    xi = np.rint(points[:, :, 2] / dx).astype(np.int64)
    inside = (zi >= 0) & (zi < nz) & (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
    occupancy = np.zeros(out_shape, dtype=bool)
    occupancy[inside] = mask[zi[inside], yi[inside], xi[inside]]
    return CrossSection(occupancy, (du, dv))


def upsample_section(section: CrossSection, factor: int) -> CrossSection:
    """Replicate a section ``factor`` times along both in-plane axes.

    The mask is refined rather than interpolated, so the lumen boundary stays at
    the positions the segmentation put it and the five-point stencil sees a
    proportionally finer grid. This is the upsampling of Sec. 6.5: the descriptor
    error is then limited by the mask boundary rather than by the resampling grid.
    """
    if factor < 1:
        raise ValueError("upsampling factor must be at least one")
    if factor == 1:
        return section
    occupancy = np.repeat(np.repeat(section.occupancy, factor, axis=0), factor, axis=1)
    du, dv = section.spacing_uv
    return CrossSection(occupancy, (du / factor, dv / factor))


def downsample_section(section: CrossSection, factor: int) -> CrossSection:
    """Inverse of :func:`upsample_section` for masks that are exact replicas."""
    if factor < 1:
        raise ValueError("downsampling factor must be at least one")
    if factor == 1:
        return section
    height, width = section.occupancy.shape
    if height % factor or width % factor:
        raise ValueError("section shape is not divisible by the downsampling factor")
    padded = section.occupancy.reshape(height // factor, factor, width // factor, factor)
    occupancy = np.asarray(padded.any(axis=(1, 3)), dtype=bool)
    du, dv = section.spacing_uv
    return CrossSection(occupancy, (du * factor, dv * factor))


def resample_arclength(
    points: FloatArray, values: FloatArray | None, step: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Resample a polyline (and per-vertex values) onto a uniform arclength grid.

    Returns ``(resampled_points, resampled_values, arclength_grid)``. ``values``
    may be ``None``, in which case the second element is an empty array.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[0] < 2:
        raise ValueError("at least two path vertices are required")
    if step <= 0.0:
        raise ValueError("grid step must be strictly positive")
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(s[-1])
    if total <= _MIN_AXIS_EPS:
        raise ValueError("path has zero arclength")
    grid = np.arange(0.0, total + 0.5 * step, step)
    grid = grid[grid <= total + _MIN_AXIS_EPS]
    out_pts = np.stack([np.interp(grid, s, pts[:, axis]) for axis in range(3)], axis=1)
    if values is None:
        out_vals = np.zeros((0,), dtype=np.float64)
    else:
        vals = np.asarray(values, dtype=np.float64)
        if vals.shape[0] != pts.shape[0]:
            raise ValueError("values and points must agree on the vertex count")
        out_vals = np.interp(grid, s, vals)
    return out_pts, out_vals, grid


def _connected_components(occupancy: BoolArray) -> tuple[IntArray, int]:
    """Two-pass 4-connected labelling on the 2D occupancy grid.

    The label array is built over the bounding box of the occupancy, so the
    union-find pass runs over the lumen rather than over the padded output grid.
    """
    rows = np.flatnonzero(occupancy.any(axis=1))
    cols = np.flatnonzero(occupancy.any(axis=0))
    labels = np.zeros(occupancy.shape, dtype=np.int64)
    if rows.size == 0 or cols.size == 0:
        return labels, 0
    r0, r1 = int(rows[0]), int(rows[-1] + 1)
    c0, c1 = int(cols[0]), int(cols[-1] + 1)
    window = occupancy[r0:r1, c0:c1]
    height, width = window.shape
    local = np.zeros((height, width), dtype=np.int64)
    parent = [0]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(lhs: int, rhs: int) -> None:
        left, right = find(lhs), find(rhs)
        if left != right:
            parent[max(left, right)] = min(left, right)

    next_label = 1
    for row in range(height):
        for col in range(width):
            if not window[row, col]:
                continue
            up = local[row - 1, col] if row > 0 else 0
            left = local[row, col - 1] if col > 0 else 0
            if up == 0 and left == 0:
                local[row, col] = next_label
                parent.append(next_label)
                next_label += 1
            elif up == 0:
                local[row, col] = left
            elif left == 0:
                local[row, col] = up
            else:
                local[row, col] = min(up, left)
                union(int(up), int(left))
    if next_label == 1:
        return labels, 0
    root_of = np.array([find(node) for node in range(next_label)], dtype=np.int64)
    roots = np.unique(root_of[1:])
    compact = np.zeros(next_label, dtype=np.int64)
    compact[roots] = np.arange(1, roots.size + 1, dtype=np.int64)
    labels[r0:r1, c0:c1] = compact[root_of[local]]
    return labels, int(roots.size)


def section_diameter(section: CrossSection) -> float:
    """Maximum caliper diameter of the occupancy set, in mm.

    Only boundary cells can realise the maximum, so the comparison runs over the
    boundary of the occupancy rather than over its interior.
    """
    if section.is_empty:
        return 0.0
    occupancy = section.occupancy
    eroded = np.ones_like(occupancy)
    eroded[1:, :] &= occupancy[:-1, :]
    eroded[:-1, :] &= occupancy[1:, :]
    eroded[:, 1:] &= occupancy[:, :-1]
    eroded[:, :-1] &= occupancy[:, 1:]
    boundary = occupancy & ~eroded
    rows, cols = np.nonzero(boundary)
    if rows.size < 2:
        return 0.0
    du, dv = section.spacing_uv
    pts = np.stack([rows * du, cols * dv], axis=1)
    best = 0.0
    for start in range(pts.shape[0] - 1):
        delta = pts[start + 1 :] - pts[start]
        best = max(best, float(np.max(np.hypot(delta[:, 0], delta[:, 1]))))
    return best


def equivalent_diameter(area: float) -> float:
    """Diameter of the disc with the same area."""
    if area <= 0.0:
        return 0.0
    return 2.0 * math.sqrt(area / math.pi)
