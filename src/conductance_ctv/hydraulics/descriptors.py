"""The hydraulic descriptor vector of a limb, per segment.

Algorithm 1 is implemented here: the centreline is extracted and resampled, a
perpendicular section is sliced at every arc-length sample, Eq. (1) is solved on
each section, and the per-segment descriptor vector is assembled from the
conductance curve. The seven fields are the minimal luminal area, the arc length
of the minimum-conductance sample, the shape factor and channel count there, the
series resistance, the dimensionless resistance ratio and the
conductance-weighted length.

Ref: Sec. 6.5, Algorithm 1; the descriptor vector is the input of the descriptor
branch in Sec. 6.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from conductance_ctv.hydraulics.centerline import (
    DEFAULT_SEGMENT_FRACTIONS,
    PathSamples,
    build_path,
)
from conductance_ctv.hydraulics.centerline import (
    SEGMENT_NAMES as SEGMENT_NAMES,
)
from conductance_ctv.hydraulics.conductance import SectionHydraulics, analyse_section
from conductance_ctv.hydraulics.grid import CrossSection, section_from_volume, upsample_section
from conductance_ctv.hydraulics.series import (
    SpanError,
    conductance_weighted_length,
    mean_conductance,
    minimal_area_index,
    minimal_conductance_index,
    occlusion_length,
    reference_resistance,
    reference_span,
    resistance_ratio,
    series_resistance,
)
from conductance_ctv.utils.numeric import as_triple
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

DESCRIPTOR_FIELDS: tuple[str, ...] = (
    "minimum_luminal_area",
    "minimum_conductance_arclength",
    "shape_factor",
    "channel_count",
    "series_resistance",
    "resistance_ratio",
    "conductance_weighted_length",
)

DEFAULT_FIELD_OF_VIEW_MM = 40.0


@dataclass(frozen=True)
class SegmentDescriptor:
    """The descriptor vector of one anatomical segment, plus its diagnostics."""

    name: str
    index: int
    minimum_luminal_area: float
    minimum_conductance_arclength: float
    shape_factor: float
    channel_count: float
    series_resistance: float
    resistance_ratio: float
    conductance_weighted_length: float
    occlusion_length: float
    patent: bool
    mean_area: float
    minimum_area_arclength: float
    resolution_limited: bool
    samples: int
    reference_segment: str = ""

    def vector(self) -> FloatArray:
        return np.array([float(getattr(self, name)) for name in DESCRIPTOR_FIELDS], dtype=np.float64)

    def record(self) -> dict[str, float | str | bool]:
        payload: dict[str, float | str | bool] = {
            "segment": self.name,
            "patent": self.patent,
            "occlusion_length": self.occlusion_length,
            "mean_area": self.mean_area,
            "minimum_area_arclength": self.minimum_area_arclength,
            "resolution_limited": self.resolution_limited,
            "samples": float(self.samples),
            "reference_segment": self.reference_segment,
        }
        for name in DESCRIPTOR_FIELDS:
            payload[name] = float(getattr(self, name))
        return payload


@dataclass(frozen=True)
class HydraulicSignature:
    """Full hydraulic description of a limb: three segments over one path."""

    segments: tuple[SegmentDescriptor, ...]
    arclength: FloatArray
    area: FloatArray
    conductance: FloatArray
    shape_factor: FloatArray
    channels: FloatArray
    segment_of_sample: IntArray
    reference_mode: str = "max"
    diagnostics: dict[str, float] = field(default_factory=dict)

    def vector(self) -> FloatArray:
        """Concatenated descriptor vector, ``7`` values per segment."""
        if not self.segments:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate([segment.vector() for segment in self.segments])

    def descriptor_names(self) -> tuple[str, ...]:
        return tuple(f"{segment.name}_{name}" for segment in self.segments for name in DESCRIPTOR_FIELDS)

    def segment(self, index: int) -> SegmentDescriptor:
        return self.segments[index]

    def record(self) -> dict[str, object]:
        payload: dict[str, object] = {"reference_mode": self.reference_mode}
        for segment in self.segments:
            for key, value in segment.record().items():
                payload[f"{segment.name}.{key}"] = value
        payload.update(self.diagnostics)
        return payload


def section_shape_for_field_of_view(field_of_view_mm: float, spacing_mm: float) -> tuple[int, int]:
    if field_of_view_mm <= 0.0 or spacing_mm <= 0.0:
        raise ValueError("field of view and spacing must be strictly positive")
    cells = int(math.ceil(field_of_view_mm / spacing_mm))
    return (max(3, cells | 1), max(3, cells | 1))


def sample_sections(
    mask: BoolArray,
    spacing_zyx: tuple[float, float, float],
    path: PathSamples,
    spacing_mm: float,
    field_of_view_mm: float = DEFAULT_FIELD_OF_VIEW_MM,
) -> list[CrossSection]:
    """Slice a perpendicular cross-section at every arc-length sample."""
    out_shape = section_shape_for_field_of_view(field_of_view_mm, spacing_mm)
    sections: list[CrossSection] = []
    for index in range(path.points.shape[0]):
        basis = path.frame(index)
        sections.append(section_from_volume(mask, spacing_zyx, basis, out_shape, spacing_mm))
    return sections


def analyse_samples(sections: list[CrossSection], upsample_factor: int = 1) -> list[SectionHydraulics]:
    """Solve Eq. (1) on every sampled section."""
    return [analyse_section(upsample_section(section, upsample_factor)) for section in sections]


def describe_sections(
    results: list[SectionHydraulics],
    arclength: FloatArray,
    segment_of_sample: IntArray,
    reference_mode: str = "max",
) -> HydraulicSignature:
    """Assemble the per-segment descriptor vector from a solved conductance curve."""
    if len(results) != arclength.shape[0] or len(results) != segment_of_sample.shape[0]:
        raise ValueError("results, arclength and segment labels must have equal length")
    ds = float(arclength[1] - arclength[0]) if arclength.shape[0] > 1 else 1.0
    area = np.array([result.area for result in results], dtype=np.float64)
    conductance = np.array([result.conductance for result in results], dtype=np.float64)
    kappa = np.array([result.shape_factor for result in results], dtype=np.float64)
    channels = np.array([float(result.channels) for result in results], dtype=np.float64)
    resolution_limited = np.array([result.resolution_limited for result in results], dtype=bool)

    spans: dict[str, IntArray] = {}
    for index, name in enumerate(SEGMENT_NAMES):
        members = np.flatnonzero(segment_of_sample == index)
        if members.size:
            spans[name] = members.astype(np.int64)

    ordered_names = [name for name in SEGMENT_NAMES if name in spans]
    descriptors: list[SegmentDescriptor] = []
    for index, name in enumerate(SEGMENT_NAMES):
        if name not in spans:
            continue
        members = spans[name]
        span_conductance = conductance[members]
        span_area = area[members]
        span_kappa = kappa[members]
        local_star = minimal_conductance_index(span_conductance)
        star = int(members[local_star])
        resistance = series_resistance(span_conductance, ds)
        closed_length = occlusion_length(span_conductance, ds)

        candidates = {
            other: conductance[spans[other]]
            for other in ordered_names
            if other != name and mean_conductance(conductance[spans[other]]) > 0.0
        }
        reference_name = ""
        ratio = math.inf
        weighted = 0.0
        if candidates:
            upstream_order = tuple(other for other in ordered_names[: ordered_names.index(name)]) + tuple(
                other for other in ordered_names[ordered_names.index(name) + 1 :]
            )
            try:
                reference_name = reference_span(
                    candidates,
                    span_length=float(members.size) * ds,
                    ds=ds,
                    mode=reference_mode,
                    order=upstream_order,
                )
            except SpanError:
                reference_name = ""
        if reference_name:
            reference_values = conductance[spans[reference_name]]
            try:
                ref = reference_resistance(reference_values, float(members.size) * ds, ds)
                ratio = resistance_ratio(resistance, ref)
                weighted = conductance_weighted_length(span_conductance, ds, mean_conductance(reference_values))
            except SpanError:
                ratio = math.inf

        descriptors.append(
            SegmentDescriptor(
                name=name,
                index=index,
                minimum_luminal_area=float(span_area[local_star]),
                minimum_conductance_arclength=float(arclength[star]),
                shape_factor=float(span_kappa[local_star]),
                channel_count=float(channels[star]),
                series_resistance=float(resistance),
                resistance_ratio=float(ratio),
                conductance_weighted_length=float(weighted),
                occlusion_length=float(closed_length),
                patent=bool(np.any(span_conductance > 0.0)),
                mean_area=float(np.mean(span_area)),
                minimum_area_arclength=float(arclength[members[minimal_area_index(span_area)]]),
                resolution_limited=bool(resolution_limited[star]),
                samples=int(members.size),
                reference_segment=reference_name,
            )
        )

    diagnostics = {
        "occluded_samples": float(np.count_nonzero(conductance <= 0.0)),
        "unconverged_samples": float(np.count_nonzero([not result.converged for result in results])),
        "max_section_residual": float(max((result.residual for result in results), default=0.0)),
        "resolution_limited_samples": float(np.count_nonzero(resolution_limited)),
    }
    return HydraulicSignature(
        segments=tuple(descriptors),
        arclength=np.asarray(arclength, dtype=np.float64),
        area=area,
        conductance=conductance,
        shape_factor=kappa,
        channels=channels,
        segment_of_sample=np.asarray(segment_of_sample, dtype=np.int64),
        reference_mode=reference_mode,
        diagnostics=diagnostics,
    )


def signature_from_mask(
    mask: BoolArray,
    spacing_zyx: tuple[float, float, float],
    step: float | None = None,
    upsample_factor: int = 1,
    field_of_view_mm: float = DEFAULT_FIELD_OF_VIEW_MM,
    segment_fractions: tuple[float, float, float, float] = DEFAULT_SEGMENT_FRACTIONS,
    reference_mode: str = "max",
    smooth_window: int = 5,
) -> HydraulicSignature:
    """End-to-end Algorithm 1 on one limb volume."""
    spacing = as_triple(spacing_zyx)
    resolved_step = float(step) if step is not None else max(min(spacing), 1.0)
    path = build_path(mask, spacing, resolved_step, smooth_window=smooth_window)
    labels = path.segment_of(segment_fractions)
    spacing_mm = min(spacing[:2]) / float(upsample_factor)
    sections = sample_sections(mask, spacing, path, spacing_mm, field_of_view_mm)
    results = analyse_samples(sections, upsample_factor=1)
    return describe_sections(results, path.arclength, labels, reference_mode=reference_mode)


def descriptor_matrix(signatures: list[HydraulicSignature]) -> FloatArray:
    """Stack limb descriptor vectors into a design matrix."""
    if not signatures:
        return np.zeros((0, len(SEGMENT_NAMES) * len(DESCRIPTOR_FIELDS)), dtype=np.float64)
    return np.stack([signature.vector() for signature in signatures], axis=0)


def descriptor_feature_names() -> tuple[str, ...]:
    return tuple(f"{segment}_{name}" for segment in SEGMENT_NAMES for name in DESCRIPTOR_FIELDS)


def dimensionless_columns() -> tuple[int, ...]:
    """Column indices of the dimensionless families, used by the transport study.

    The resistance ratio and the shape factor carry no length scale, so their
    agreement with manual measurement is expected to survive the move to another
    site while the absolute-area family degrades.

    Ref: Sec. 3.3 and Table 4.
    """
    names = descriptor_feature_names()
    keep = {"resistance_ratio", "shape_factor"}
    return tuple(index for index, name in enumerate(names) if name.split("_", 1)[1] in keep)


def absolute_columns() -> tuple[int, ...]:
    """Column indices of the length-scaled descriptor families."""
    names = descriptor_feature_names()
    keep = {"minimum_luminal_area", "conductance_weighted_length"}
    return tuple(index for index, name in enumerate(names) if name.split("_", 1)[1] in keep)
