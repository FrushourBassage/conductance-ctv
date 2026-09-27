"""Comparator models for the four-type table.

Four families are declared, matching the comparator rows of Table 3: the
single-diameter rule, a hand-computed radiomic feature set with three off-the-shelf
learners, an end-to-end convolutional classifier, and the transferred, frozen and
prompted variants of the backbone.

None of these models sees a hydraulic descriptor: that is the point of the
comparison. The radiomic feature extractor is written out rather than delegated,
so the feature definitions are auditable and the extractor carries no hidden
dependency.

Ref: Table 3; Sec. 2.2 (what the published systems do); Sec. 6.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import torch
from torch import nn

from conductance_ctv.data.fixture import DEFAULT_GRID, build_profile, render_mask
from conductance_ctv.data.schema import LimbRecord
from conductance_ctv.models.encoders import ViT3D
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

GLCM_OFFSETS: Final[tuple[tuple[int, int, int], ...]] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)
GLCM_LEVELS: Final[int] = 16

RADIOMIC_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "volume_mm3",
    "surface_mm2",
    "sphericity",
    "max_diameter_mm",
    "elongation",
    "flatness",
    "intensity_mean",
    "intensity_std",
    "intensity_skew",
    "intensity_kurtosis",
    "intensity_p10",
    "intensity_p90",
    "intensity_energy",
    "intensity_entropy",
    "glcm_contrast",
    "glcm_correlation",
    "glcm_energy",
    "glcm_homogeneity",
    "lesion_length_mm",
    "stenosis_fraction",
    "voxel_anisotropy",
)


def _voxel_covariance_axes(mask: BoolArray, spacing: tuple[float, float, float]) -> FloatArray:
    """Principal axis lengths of the mask, in mm, largest first."""
    coords = np.stack(np.nonzero(mask), axis=1).astype(np.float64)
    if coords.shape[0] < 4:
        return np.zeros(3, dtype=np.float64)
    scaled = coords * np.asarray(spacing, dtype=np.float64)
    centred = scaled - scaled.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(coords.shape[0] - 1, 1)
    values = np.linalg.eigvalsh(covariance)
    values = np.sort(np.clip(values, 0.0, None))[::-1]
    axes: FloatArray = np.sqrt(values) * math.sqrt(12.0)
    return axes


def _surface_area(mask: BoolArray, spacing: tuple[float, float, float]) -> float:
    """Interface area between occupied and unoccupied cells, in mm^2."""
    dz, dy, dx = spacing
    area = 0.0
    area += float(np.count_nonzero(mask[0, :, :] | mask[-1, :, :])) * dy * dx
    area += float(np.count_nonzero(mask[:, 0, :] | mask[:, -1, :])) * dz * dx
    area += float(np.count_nonzero(mask[:, :, 0] | mask[:, :, -1])) * dz * dy
    area += float(np.count_nonzero(mask[:-1, :, :] ^ mask[1:, :, :])) * dy * dx
    area += float(np.count_nonzero(mask[:, :-1, :] ^ mask[:, 1:, :])) * dz * dx
    area += float(np.count_nonzero(mask[:, :, :-1] ^ mask[:, :, 1:])) * dz * dy
    return area


def _first_order(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return np.zeros(8, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    centred = values - mean
    skew = float(np.mean(centred**3) / std**3) if std > 1e-9 else 0.0
    kurtosis = float(np.mean(centred**4) / std**4) if std > 1e-9 else 0.0
    histogram, _ = np.histogram(values, bins=32, density=False)
    probabilities = histogram.astype(np.float64) / max(histogram.sum(), 1)
    nonzero = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    energy = float(np.sum(probabilities**2))
    return np.array(
        [
            mean,
            std,
            skew,
            kurtosis,
            float(np.percentile(values, 10)),
            float(np.percentile(values, 90)),
            energy,
            entropy,
        ],
        dtype=np.float64,
    )


def _max_caliper(mask: BoolArray, spacing: tuple[float, float, float], max_points: int = 400) -> float:
    """Largest distance between two boundary voxels, in mm.

    Only boundary voxels can realise the maximum, and the point set is strided
    down to a bounded size so the comparison stays linear in a limb rather than
    quadratic in its voxel count.
    """
    if not bool(mask.any()):
        return 0.0
    eroded = np.array(mask, dtype=bool)
    for axis in range(3):
        eroded &= np.roll(mask, 1, axis=axis)
        eroded &= np.roll(mask, -1, axis=axis)
    boundary = mask & ~eroded
    coords = np.stack(np.nonzero(boundary if bool(boundary.any()) else mask), axis=1).astype(np.float64)
    points = coords * np.asarray(spacing, dtype=np.float64)
    if points.shape[0] > max_points:
        stride = int(math.ceil(points.shape[0] / max_points))
        points = points[::stride]
    deltas = points[:, None, :] - points[None, :, :]
    return float(np.max(np.sqrt(np.sum(deltas**2, axis=2))))


def _glcm(values: FloatArray, mask: BoolArray) -> FloatArray:
    """Grey-level co-occurrence statistics over six three-dimensional offsets."""
    if values.size == 0 or not bool(mask.any()):
        return np.zeros(4, dtype=np.float64)
    finite = values[mask]
    low = float(finite.min())
    high = float(finite.max())
    if high - low < 1e-9:
        return np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    quantised = np.zeros(values.shape, dtype=np.int64)
    quantised[mask] = np.clip(
        ((values[mask] - low) / (high - low) * (GLCM_LEVELS - 1)).astype(np.int64), 0, GLCM_LEVELS - 1
    )
    levels = np.arange(GLCM_LEVELS, dtype=np.float64)
    difference = levels[:, None] - levels[None, :]
    contrast = 0.0
    correlation = 0.0
    energy = 0.0
    homogeneity = 0.0
    counted = 0
    for offset in GLCM_OFFSETS:
        shifted = np.roll(quantised, shift=offset, axis=(0, 1, 2))
        valid = mask & np.roll(mask, shift=offset, axis=(0, 1, 2))
        pairs = np.stack([quantised[valid], shifted[valid]], axis=1)
        if pairs.shape[0] == 0:
            continue
        matrix = np.zeros((GLCM_LEVELS, GLCM_LEVELS), dtype=np.float64)
        np.add.at(matrix, (pairs[:, 0], pairs[:, 1]), 1.0)
        matrix = matrix + matrix.T
        matrix = matrix / matrix.sum()
        mu_row = float(np.sum(levels[:, None] * matrix))
        mu_col = float(np.sum(levels[None, :] * matrix))
        var_row = float(np.sum((levels[:, None] - mu_row) ** 2 * matrix))
        var_col = float(np.sum((levels[None, :] - mu_col) ** 2 * matrix))
        contrast += float(np.sum(matrix * difference**2))
        if var_row > 1e-9 and var_col > 1e-9:
            covariance = float(np.sum(matrix * (levels[:, None] - mu_row) * (levels[None, :] - mu_col)))
            correlation += covariance / math.sqrt(var_row * var_col)
        homogeneity += float(np.sum(matrix / (1.0 + difference**2)))
        energy += float(np.sum(matrix**2))
        counted += 1
    if counted == 0:
        return np.zeros(4, dtype=np.float64)
    scale = 1.0 / float(counted)
    return np.array([contrast * scale, correlation * scale, energy * scale, homogeneity * scale], dtype=np.float64)


def radiomic_features(
    mask: BoolArray,
    image: FloatArray,
    spacing_zyx: tuple[float, float, float],
    lesion_length_mm: float = 0.0,
    stenosis_fraction: float = 0.0,
) -> FloatArray:
    """Hand-computed shape, first-order and co-occurrence features of one limb."""
    if mask.shape != image.shape:
        raise ValueError("mask and image must share a shape")
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    n_voxels = int(np.count_nonzero(mask))
    volume = n_voxels * voxel_volume
    surface = _surface_area(mask, spacing_zyx)
    sphericity = (math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface if surface > 1e-9 else 0.0
    axes = _voxel_covariance_axes(mask, spacing_zyx)
    max_diameter = _max_caliper(mask, spacing_zyx)
    elongation = float(axes[1] / axes[0]) if axes[0] > 1e-9 else 0.0
    flatness = float(axes[2] / axes[0]) if axes[0] > 1e-9 else 0.0
    first_order = _first_order(image[mask] if n_voxels else np.zeros(0, dtype=np.float64))
    glcm = _glcm(image, mask)
    anisotropy = float(np.max(spacing_zyx) / min(spacing_zyx))
    return np.concatenate(
        [
            np.array([volume, surface, sphericity, max_diameter, elongation, flatness], dtype=np.float64),
            first_order,
            glcm,
            np.array([lesion_length_mm, stenosis_fraction, anisotropy], dtype=np.float64),
        ]
    )


@dataclass(frozen=True)
class RadiomicDataset:
    """Feature matrix, type labels and the columns the fairness constraint uses."""

    features: FloatArray
    labels: FloatArray
    patient_index: np.ndarray
    feature_names: tuple[str, ...] = RADIOMIC_FEATURE_NAMES


def diameter_rule_scores(records: list[LimbRecord], occlusion_cut: float = 0.95) -> FloatArray:
    """Per-type scores of the single-diameter rule.

    The rule measures one diameter-derived number, so the four one-versus-rest
    scores have to come from that number and from the two binary observations a
    diameter reading supports, namely a spur and a closed lumen. Severity is the
    diameter-derived stenosis fraction; the score for the compressive type is its
    complement, because a diameter rule assigns severity to a compressive lesion
    only in proportion to how much it narrowed the lumen.

    Ref: Table 3 first row; Sec. 1 and Sec. 4.6 (the systematic bias of a
    diameter-calibrated rule).
    """
    rows = np.zeros((len(records), 4), dtype=np.float64)
    for index, record in enumerate(records):
        severity = float(np.clip(record.stenosis_ratio, 0.0, 1.0))
        rows[index, 0] = 1.0 - severity
        rows[index, 1] = severity
        rows[index, 2] = severity if record.intraluminal_spur else 0.5 * severity
        rows[index, 3] = 1.0 if severity >= occlusion_cut else severity * 0.5
    return rows


class ProbabilisticClassifier(Protocol):
    """The part of the scikit-learn estimator interface these comparators use."""

    classes_: np.ndarray

    def fit(self, features: np.ndarray, labels: np.ndarray) -> object: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


class RadiomicsComparator:
    """One of the three radiomic learners, on shared features.

    The learner is selected by name so the three rows of Table 3 differ only in
    the classifier and never in the feature set. Feature standardisation is
    fitted on the training partition and reused for every later partition.

    Ref: Table 3 rows two to four.
    """

    def __init__(self, learner: str = "logistic", seed: int = 0, **kwargs: float | int) -> None:
        self.learner = learner
        self.seed = seed
        self.kwargs = kwargs
        self.model: ProbabilisticClassifier | None = None
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def _build(self) -> ProbabilisticClassifier:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        if self.learner in ("lr", "logistic"):
            logistic: ProbabilisticClassifier = LogisticRegression(
                max_iter=2000, C=float(self.kwargs.get("C", 1.0)), random_state=self.seed
            )
            return logistic
        if self.learner == "gbm":
            boosted: ProbabilisticClassifier = HistGradientBoostingClassifier(
                max_iter=int(self.kwargs.get("max_iter", 300)), random_state=self.seed
            )
            return boosted
        if self.learner == "svm":
            margin: ProbabilisticClassifier = SVC(
                probability=True, C=float(self.kwargs.get("C", 2.0)), random_state=self.seed
            )
            return margin
        raise KeyError(f"unknown radiomic learner {self.learner!r}")

    def fit(self, features: FloatArray, labels: FloatArray) -> RadiomicsComparator:
        values = np.nan_to_num(np.asarray(features, dtype=np.float64))
        self.mean_ = values.mean(axis=0)
        self.scale_ = np.where(values.std(axis=0) < 1e-9, 1.0, values.std(axis=0))
        model = self._build()
        model.fit((values - self.mean_) / self.scale_, np.asarray(labels, dtype=np.int64))
        self.model = model
        return self

    def decision_scores(self, features: FloatArray) -> FloatArray:
        if self.model is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("the comparator must be fitted before it is scored")
        values = np.nan_to_num(np.asarray(features, dtype=np.float64))
        standardised = (values - self.mean_) / self.scale_
        scores = np.asarray(self.model.predict_proba(standardised), dtype=np.float64)
        order = [int(label) for label in self.model.classes_]
        out = np.zeros((values.shape[0], 4), dtype=np.float64)
        for column, label in enumerate(order):
            out[:, label] = scores[:, column]
        return out

    def predict(self, features: FloatArray) -> IntArray:
        assignment: IntArray = np.argmax(self.decision_scores(features), axis=1)
        return assignment


class DeepComparator(nn.Module):
    """Encoder, global pooling and a linear four-type head, without descriptors.

    Ref: Table 3 end-to-end, frozen, fine-tuned, transferred and prompted rows.
    """

    def __init__(self, encoder: nn.Module, feature_channels: int, n_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_channels, n_classes)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        if isinstance(encoder, ViT3D):
            pooled: torch.Tensor = encoder.pooled(inputs)
            return pooled
        encoded: torch.Tensor = encoder(inputs)
        reduced: torch.Tensor = torch.amax(encoded.flatten(2), dim=2)
        return reduced

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.dropout(self.features(inputs)))
        return logits


def fixture_radiomic_dataset(
    records: list[LimbRecord], grid: tuple[int, int, int] = (48, 40, 40), seed: int = 0
) -> RadiomicDataset:
    """Radiomic features of a set of records, rendered from their own profiles.

    The cohort fixture shares its schema with the clinical cohort and its
    anatomy is prescribed in closed form, so the comparator rows are executable
    without the clinical volumes.
    """
    from conductance_ctv.data.fixture import render_intensity
    from conductance_ctv.data.schema import GridSpec

    spec_grid = GridSpec(shape=grid, spacing_zyx=DEFAULT_GRID.spacing_zyx)
    features: list[FloatArray] = []
    labels: list[float] = []
    patients: list[int] = []
    lookup: dict[str, int] = {}
    for index, record in enumerate(records):
        rng = np.random.default_rng(seed + index)
        profile = build_profile(
            spec_grid,
            record.reference_type,
            record.lesion_segment,
            float(record.axis_ratio),
            rng,
            collateral_grade=record.collateral_grade,
        )
        mask = render_mask(profile)
        image = render_intensity(profile, record.phase, record.vendor, np.random.default_rng(seed + 5000 + index))
        features.append(
            radiomic_features(
                mask,
                image,
                spec_grid.spacing_zyx,
                lesion_length_mm=float(record.lesion_length),
                stenosis_fraction=float(record.stenosis_ratio),
            )
        )
        labels.append(float(int(record.reference_type) - 1))
        if record.patient_id not in lookup:
            lookup[record.patient_id] = len(lookup)
        patients.append(lookup[record.patient_id])
    return RadiomicDataset(
        features=np.stack(features, axis=0),
        labels=np.asarray(labels, dtype=np.float64),
        patient_index=np.asarray(patients, dtype=np.int64),
    )
