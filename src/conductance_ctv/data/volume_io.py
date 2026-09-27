"""Reading and writing volumetric imaging data.

A minimal NIfTI-1 reader and writer keeps the release independent of a specific
imaging library while still stating the exact on-disk contract a site has to
meet: a header whose voxel sizes and affine are read rather than assumed, and an
intensity volume whose values are the ones the scanner wrote. Everything the
pipeline needs beyond that (orientation, windowing) is done on the array.

Ref: Sec. 6.2 (the imaging records) and Sec. 6.4 (the segmentation inputs).
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conductance_ctv.data.schema import GridSpec
from conductance_ctv.utils.io import write_bytes
from conductance_ctv.utils.numeric import as_triple, as_triple_int
from conductance_ctv.utils.types import BoolArray, FloatArray

NIFTI_SINGLE = b"n+1\x00"
NIFTI_PAIR = b"ni1\x00"
NIFTI_HEADER_BYTES = 348
NIFTI_DATATYPES: dict[int, str] = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


class VolumeIOError(RuntimeError):
    """Raised when a volume cannot be read as a plain NIfTI-1 single file."""


@dataclass(frozen=True)
class Volume:
    """An intensity volume with the grid it was sampled on."""

    data: FloatArray
    grid: GridSpec
    affine: FloatArray

    def __post_init__(self) -> None:
        if self.data.shape != self.grid.shape:
            raise ValueError("data shape and grid shape disagree")

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.data.shape))

    def window(self, low: float, high: float) -> FloatArray:
        """Clamp to a window and rescale to ``[0, 1]``."""
        if high <= low:
            raise ValueError("the window must have positive width")
        return np.clip((self.data - low) / (high - low), 0.0, 1.0)


def read_nifti(path: str | Path) -> Volume:
    """Read a NIfTI-1 single-file volume, gzipped or not."""
    target = Path(path)
    if not target.exists():
        raise VolumeIOError(f"missing volume: {target.name}")
    raw = gzip.decompress(target.read_bytes()) if target.suffix == ".gz" else target.read_bytes()
    if len(raw) < NIFTI_HEADER_BYTES:
        raise VolumeIOError("file is shorter than a NIfTI header")
    if raw[344:348] not in (NIFTI_SINGLE, NIFTI_PAIR):
        raise VolumeIOError("only NIfTI-1 single-file volumes are supported")
    little = struct.unpack("<i", raw[0:4])[0] == NIFTI_HEADER_BYTES
    endian = "<" if little else ">"
    datatype = struct.unpack(f"{endian}h", raw[70:72])[0]
    bitpix = struct.unpack(f"{endian}h", raw[72:74])[0]
    if datatype not in NIFTI_DATATYPES:
        raise VolumeIOError(f"unsupported NIfTI datatype code {datatype}")
    dims = struct.unpack(f"{endian}8h", raw[40:56])
    shape = as_triple_int(dims[1:4])
    if any(value < 1 for value in shape):
        raise VolumeIOError("the header declares an empty volume")
    spacing = as_triple(struct.unpack(f"{endian}3f", raw[80:92]))
    offset = int(struct.unpack(f"{endian}f", raw[108:112])[0])
    if offset < NIFTI_HEADER_BYTES:
        offset = NIFTI_HEADER_BYTES + 4
    rows = np.asarray(struct.unpack(f"{endian}12f", raw[280:328]), dtype=np.float64).reshape(3, 4)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = rows
    expected = int(np.prod(shape))
    dtype = np.dtype(NIFTI_DATATYPES[datatype])
    if int(bitpix) != dtype.itemsize * 8:
        raise VolumeIOError("the header bit depth disagrees with its datatype")
    payload = raw[offset : offset + expected * dtype.itemsize]
    if len(payload) != expected * dtype.itemsize:
        raise VolumeIOError("the file is shorter than its header declares")
    data = np.frombuffer(payload, dtype=dtype).reshape(shape).astype(np.float64)
    return Volume(data=data, grid=GridSpec(shape=shape, spacing_zyx=spacing), affine=affine)


def write_nifti(path: str | Path, volume: Volume, datatype: int = 16) -> Path:
    """Write a NIfTI-1 single-file volume (uncompressed) with sform/qform set."""
    if datatype not in NIFTI_DATATYPES:
        raise VolumeIOError(f"unsupported output datatype code {datatype}")
    dtype = np.dtype(NIFTI_DATATYPES[datatype])
    header = bytearray(NIFTI_HEADER_BYTES)
    struct.pack_into("<i", header, 0, NIFTI_HEADER_BYTES)
    struct.pack_into("<8h", header, 40, 3, *volume.data.shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, dtype.itemsize * 8)
    struct.pack_into("<3f", header, 80, *volume.grid.spacing_zyx)
    struct.pack_into("<f", header, 108, NIFTI_HEADER_BYTES + 4)
    header[123] = ord("r")
    header[344:348] = NIFTI_SINGLE
    rows = np.zeros(12, dtype=np.float32)
    rows[:3] = np.asarray(volume.grid.spacing_zyx, dtype=np.float32)
    rows[4:7] = np.asarray(volume.grid.spacing_zyx, dtype=np.float32)
    rows[8:11] = np.asarray(volume.grid.spacing_zyx, dtype=np.float32)
    struct.pack_into("<12f", header, 280, *rows.tolist())
    payload = np.asarray(volume.data, dtype=dtype).tobytes(order="C")
    return write_bytes(Path(path), bytes(header) + b"\x00\x00\x00\x00" + payload)


def resolution_report(volumes: list[Volume]) -> dict[str, object]:
    """Summarise the grids of a set of volumes, for the site-level description."""
    if not volumes:
        return {"count": 0}
    spacings = np.stack([np.asarray(volume.grid.spacing_zyx, dtype=np.float64) for volume in volumes])
    shapes = np.stack([np.asarray(volume.grid.shape, dtype=np.float64) for volume in volumes])
    return {
        "count": len(volumes),
        "spacing_min": spacings.min(axis=0).tolist(),
        "spacing_max": spacings.max(axis=0).tolist(),
        "spacing_median": np.median(spacings, axis=0).tolist(),
        "shape_median": np.median(shapes, axis=0).tolist(),
        "cubic_from_shapes": bool(np.all(np.abs(spacings[:, 0] - spacings[:, 1]) > 0.15)),
    }


def resample_to_isotropic(volume: Volume, target_spacing: float) -> Volume:
    """Nearest-neighbour resample of a volume onto an isotropic grid.

    Nearest neighbour is used rather than interpolation because the through-plane
    step of a venous-phase acquisition is much coarser than the in-plane step and
    an interpolated mask would invent a lumen boundary the segmentation never
    reported.

    Ref: Sec. 6.4 (the anisotropic decoder exists for the same reason).
    """
    if target_spacing <= 0.0:
        raise ValueError("target spacing must be strictly positive")
    shape = tuple(
        max(2, int(round(size * step / target_spacing)))
        for size, step in zip(volume.grid.shape, volume.grid.spacing_zyx, strict=True)
    )
    resampled_shape = as_triple_int(shape)
    source_index = [
        np.clip(np.rint(np.arange(size) * target_spacing / step).astype(np.int64), 0, old - 1)
        for size, step, old in zip(resampled_shape, volume.grid.spacing_zyx, volume.grid.shape, strict=True)
    ]
    out = volume.data[np.ix_(source_index[0], source_index[1], source_index[2])]
    return Volume(
        data=out,
        grid=GridSpec(shape=resampled_shape, spacing_zyx=(target_spacing, target_spacing, target_spacing)),
        affine=np.eye(4),
    )


def mask_from_intensity(volume: Volume, threshold: float) -> BoolArray:
    """Threshold helper for a contrast-enhanced venous phase volume."""
    return volume.data >= threshold
