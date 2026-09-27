"""Segmentation metrics: Dice, the 95th-percentile Hausdorff distance and clDice.

Per-segment values are reported separately for the common iliac, external iliac
and common femoral vein because a pooled figure hides the anatomical gradient: a
model that is accurate at the confluence and inaccurate at the femoral
bifurcation and one that is uniformly mediocre have the same pooled Dice.

Ref: Table 1 and its caption; Sec. 2.2 (the anatomical gradient of segmentation
accuracy); Sec. 3.3 (the same gradient on the external region).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.metrics.skeleton import distance_transform, ridge_skeleton
from conductance_ctv.utils.numeric import safe_scalar
from conductance_ctv.utils.types import BoolArray, FloatArray

HD95_PERCENTILE = 95.0


def dice_score(prediction: BoolArray, reference: BoolArray) -> float:
    """Voxel Dice of two binary masks."""
    if prediction.shape != reference.shape:
        raise ValueError("the two masks must share a shape")
    intersection = float(np.count_nonzero(prediction & reference))
    total = float(np.count_nonzero(prediction)) + float(np.count_nonzero(reference))
    return safe_scalar(2.0 * intersection, total, 1.0)


def jaccard_score(prediction: BoolArray, reference: BoolArray) -> float:
    intersection = float(np.count_nonzero(prediction & reference))
    union = float(np.count_nonzero(prediction | reference))
    return safe_scalar(intersection, union, 1.0)


def distance_to_surface(mask: BoolArray, spacing_zyx: tuple[float, float, float]) -> FloatArray:
    """Distance from every voxel to the nearest surface voxel of ``mask``.

    The surface of a mask is the set of occupied voxels with an unoccupied
    six-neighbour. Taking the distance transform of its complement gives, at any
    voxel, the distance to the nearest member of that set, and zero on the
    surface itself; taking the transform of the mask instead would give the
    distance to the nearest lumen-free voxel, which is one even for a coincident
    surface and would bias every agreement metric.
    """
    if not bool(mask.any()):
        return np.zeros(mask.shape, dtype=np.float64)
    return distance_transform(~_surface(mask), spacing_zyx)


def surface_distances(
    prediction: BoolArray, reference: BoolArray, spacing_zyx: tuple[float, float, float]
) -> FloatArray:
    """Symmetric surface distances between two masks, in mm."""
    if prediction.shape != reference.shape:
        raise ValueError("the two masks must share a shape")
    if not bool(prediction.any()) or not bool(reference.any()):
        return np.zeros(0, dtype=np.float64)
    prediction_surface = _surface(prediction)
    reference_surface = _surface(reference)
    forward = distance_to_surface(reference, spacing_zyx)[prediction_surface]
    backward = distance_to_surface(prediction, spacing_zyx)[reference_surface]
    return np.concatenate([forward, backward])


def _surface(mask: BoolArray) -> BoolArray:
    interior = np.array(mask, dtype=bool)
    for axis in range(3):
        interior &= np.roll(mask, 1, axis=axis)
        interior &= np.roll(mask, -1, axis=axis)
    shell: BoolArray = mask & ~interior
    return shell


def hausdorff_distance(
    prediction: BoolArray,
    reference: BoolArray,
    spacing_zyx: tuple[float, float, float],
    percentile: float = HD95_PERCENTILE,
) -> float:
    """Percentile Hausdorff distance in millimetres; ``nan`` when a mask is empty."""
    values = surface_distances(prediction, reference, spacing_zyx)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def surface_dice(
    prediction: BoolArray, reference: BoolArray, spacing_zyx: tuple[float, float, float], tolerance: float
) -> float:
    """Fraction of surface voxels within a tolerance of the other surface."""
    if not bool(prediction.any()) or not bool(reference.any()):
        return float("nan")
    forward = distance_to_surface(reference, spacing_zyx)[_surface(prediction)]
    backward = distance_to_surface(prediction, spacing_zyx)[_surface(reference)]
    values = np.concatenate([forward, backward])
    return float(np.mean(values <= tolerance))


def centreline_dice(
    prediction: BoolArray,
    reference: BoolArray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tolerance: int = 1,
) -> float:
    """Centreline Dice: overlap of the two centrelines after dilation.

    ``2 * |S_p and V_r| / (|S_p| + |S_r|)`` with ``S`` the centrelines dilated by
    ``tolerance`` voxels and ``V`` the masks; the dilation lets a centreline
    count when it lies within the other mask's tube.

    Ref: Table 1 clDice column, and the caption's remark that clDice exceeds Dice
    for single-trunk vessels.
    """
    if not bool(prediction.any()) or not bool(reference.any()):
        return float("nan")
    skeleton_prediction = ridge_skeleton(prediction, spacing_zyx)
    skeleton_reference = ridge_skeleton(reference, spacing_zyx)
    dilated_prediction = _dilate(skeleton_prediction, tolerance) & reference
    dilated_reference = _dilate(skeleton_reference, tolerance) & prediction
    numerator = 2.0 * float(np.count_nonzero(dilated_prediction & dilated_reference))
    denominator = float(np.count_nonzero(dilated_prediction)) + float(np.count_nonzero(dilated_reference))
    return safe_scalar(numerator, denominator, 1.0)


def _dilate(mask: BoolArray, radius: int) -> BoolArray:
    if radius <= 0:
        return mask
    out = np.array(mask, dtype=bool)
    for _ in range(radius):
        grown = np.array(out, dtype=bool)
        for axis in range(3):
            grown |= np.roll(out, 1, axis=axis)
            grown |= np.roll(out, -1, axis=axis)
        out = grown
    return out


@dataclass(frozen=True)
class SegmentScore:
    """One metric of one anatomical segment, with the unit it was computed on."""

    segment: str
    metric: str
    value: float
    samples: int
    unit: str = "segment"


@dataclass(frozen=True)
class SegmentationReport:
    """Per-segment Dice, HD95 and clDice of one volume."""

    dice: dict[str, float]
    hd95: dict[str, float]
    cldice: dict[str, float]
    predicted_voxels: int
    reference_voxels: int

    def macro_dice(self) -> float:
        return float(np.mean(list(self.dice.values()))) if self.dice else float("nan")

    def records(self) -> list[dict[str, float | str]]:
        out: list[dict[str, float | str]] = []
        for segment, value in self.dice.items():
            out.append({"segment": segment, "metric": "dice", "value": value})
        for segment, value in self.hd95.items():
            out.append({"segment": segment, "metric": "hd95_mm", "value": value})
        for segment, value in self.cldice.items():
            out.append({"segment": segment, "metric": "cldice", "value": value})
        return out


def score_volume(
    prediction: BoolArray,
    reference: BoolArray,
    spacing_zyx: tuple[float, float, float],
    segment_labels: np.ndarray | None = None,
    segment_names: tuple[str, ...] = ("CIV", "EIV", "CFV"),
) -> SegmentationReport:
    """Score one predicted mask against its reference, per anatomical segment.

    ``segment_labels`` is a per-voxel label array with the same shape as the
    masks; without it the whole volume is scored as a single segment.
    """
    if segment_labels is None:
        return SegmentationReport(
            dice={"all": dice_score(prediction, reference)},
            hd95={"all": hausdorff_distance(prediction, reference, spacing_zyx)},
            cldice={"all": centreline_dice(prediction, reference, spacing_zyx)},
            predicted_voxels=int(np.count_nonzero(prediction)),
            reference_voxels=int(np.count_nonzero(reference)),
        )
    dice: dict[str, float] = {}
    hd95: dict[str, float] = {}
    cldice: dict[str, float] = {}
    for index, name in enumerate(segment_names):
        band = segment_labels == index
        dice[name] = dice_score(prediction & band, reference & band)
        hd95[name] = hausdorff_distance(prediction & band, reference & band, spacing_zyx)
        cldice[name] = centreline_dice(prediction & band, reference & band, spacing_zyx)
    return SegmentationReport(
        dice=dice,
        hd95=hd95,
        cldice=cldice,
        predicted_voxels=int(np.count_nonzero(prediction)),
        reference_voxels=int(np.count_nonzero(reference)),
    )


def aggregate_segment_scores(reports: list[SegmentationReport], segment: str, metric: str) -> FloatArray:
    """Collect one metric of one segment across volumes, dropping undefined entries."""
    values: list[float] = []
    for report in reports:
        table = {"dice": report.dice, "hd95_mm": report.hd95, "cldice": report.cldice}.get(metric)
        if table is None:
            raise KeyError(f"unknown segmentation metric {metric!r}")
        value = table.get(segment)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def anatomical_gradient(report: SegmentationReport) -> dict[str, float]:
    """Change in Dice per segment along the confluence-to-bifurcation direction.

    The manuscript reports that pooled segmentation metrics hide the gradient
    from the inferior vena cava to the common iliac vein; this exposes it
    explicitly for the three segments the model covers.

    Ref: Sec. 2.2 and Sec. 3.3.
    """
    order = [name for name in ("CIV", "EIV", "CFV") if name in report.dice]
    if len(order) < 2:
        return {}
    out: dict[str, float] = {}
    for first, second in zip(order[:-1], order[1:], strict=True):
        out[f"{first}_to_{second}"] = float(report.dice[second] - report.dice[first])
    out["span"] = float(report.dice[order[-1]] - report.dice[order[0]])
    return out
