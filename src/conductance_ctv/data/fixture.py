"""Schema-compatible cohort generator.

The clinical cohort behind the manuscript is not shareable, so the mechanism has
to be executable on a fixture that has the same schema and, crucially, a known
ground truth. Every cross-section here is prescribed in closed form, which means
the area, the conductance, the Polya-Szego ratio and the channel count of the
generated anatomy are known analytically and the finite-difference solve of
Eq. (1) can be scored against them.

The four generating shapes follow the four morphological types of Sec. 6.1:

* a circular lumen (patent reference and the lesion-free spans),
* an elliptical slit whose semi-axis ratio is drawn from the 0.15-0.30 range
  quoted for compression in Sec. 1 (types I and II),
* a septated lumen of ``k`` equal channels preserving the total area, which is
  the post-thrombotic configuration that raises resistance at constant area
  (type III),
* an absent lumen, ``G = 0`` (type IV).

Intensity is written as well, because the appearance branch of Sec. 6.6
consumes voxels rather than geometry and the transport study of Sec. 3.3 turns
on scanner and contrast differences that leave the geometry untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from conductance_ctv.data.schema import (
    AcquisitionPhase,
    CohortRecord,
    GridSpec,
    LesionType,
    LimbRecord,
    PatientRecord,
    SectionKind,
    SectionSpec,
    Segment,
    Site,
)
from conductance_ctv.hydraulics.centerline import DEFAULT_SEGMENT_FRACTIONS
from conductance_ctv.hydraulics.conductance import (
    disc_channel_conductance,
    ellipse_conductance,
    ellipse_shape_factor,
    shape_factor,
)
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

DEFAULT_GRID = GridSpec(shape=(112, 96, 96), spacing_zyx=(1.0, 0.6, 0.6))

CONFLUENCE_RADIUS_MM: Final[float] = 7.2
BIFURCATION_RADIUS_MM: Final[float] = 4.6
WOBBLE_AMPLITUDE_MM: Final[float] = 0.25
WOBBLE_WAVELENGTH_MM: Final[float] = 46.0

LESION_LENGTH_MM: Final[float] = 24.0

# The compression site sits close to the confluence, where the right common iliac
# artery crosses. The value is the position of the lesion centre inside its
# segment, as a fraction of the segment length.
LESION_CENTRE_FRACTION: Final[float] = 0.35
SEPTUM_CHANNELS: Final[tuple[int, ...]] = (2, 3, 4)
COLLATERAL_RADIUS_MM: Final[float] = 1.5

TISSUE_HU: Final[float] = 42.0
WALL_HU: Final[float] = 96.0
CONTRAST_HU: Final[dict[AcquisitionPhase, float]] = {
    AcquisitionPhase.DIRECT: 385.0,
    AcquisitionPhase.EARLY_INDIRECT: 268.0,
    AcquisitionPhase.DELAYED_INDIRECT: 196.0,
}
VENDOR_GAIN: Final[dict[str, float]] = {
    "vendor_a": 1.0,
    "vendor_b": 0.93,
    "vendor_c": 1.06,
    "vendor_d": 0.87,
}

TYPE_PREVALENCE: Final[tuple[float, float, float, float]] = (0.45, 0.30, 0.18, 0.07)

VENDORS: Final[tuple[str, ...]] = ("vendor_a", "vendor_b", "vendor_c", "vendor_d")
SITE_PHASE: Final[dict[Site, AcquisitionPhase]] = {
    Site.A: AcquisitionPhase.DIRECT,
    Site.B: AcquisitionPhase.DELAYED_INDIRECT,
    Site.C: AcquisitionPhase.EARLY_INDIRECT,
}
SITE_VENDORS: Final[dict[Site, tuple[str, ...]]] = {
    Site.A: ("vendor_a", "vendor_b"),
    Site.B: ("vendor_a", "vendor_c"),
    Site.C: ("vendor_b", "vendor_d"),
}


@dataclass(frozen=True)
class LimbProfile:
    """Closed-form cross-section prescription at every grid slice of a limb."""

    grid: GridSpec
    kind: tuple[SectionKind, ...]
    semi_major: FloatArray
    semi_minor: FloatArray
    channels: IntArray
    collateral_grade: IntArray
    centre_u: FloatArray
    centre_v: FloatArray
    centreline: FloatArray
    lesion_slice_range: tuple[int, int]

    @property
    def n_slices(self) -> int:
        return len(self.kind)

    def unit_conductance(self) -> FloatArray:
        """``G`` of every slice from the closed forms, with no grid involved."""
        out = np.zeros(self.n_slices, dtype=np.float64)
        for index, kind in enumerate(self.kind):
            if kind is SectionKind.DISC:
                out[index] = math.pi * self.semi_major[index] ** 4 / 8.0
            elif kind is SectionKind.ELLIPSE:
                out[index] = ellipse_conductance(self.semi_major[index], self.semi_minor[index])
            elif kind is SectionKind.SEPTUM:
                area = math.pi * self.semi_major[index] ** 2 * float(self.channels[index])
                out[index] = disc_channel_conductance(area, int(self.channels[index]))
            else:
                out[index] = 0.0
        return out

    def unit_area(self) -> FloatArray:
        """``A`` of every slice from the closed forms."""
        out = np.zeros(self.n_slices, dtype=np.float64)
        for index, kind in enumerate(self.kind):
            if kind is SectionKind.DISC:
                out[index] = math.pi * self.semi_major[index] ** 2
            elif kind is SectionKind.ELLIPSE:
                out[index] = math.pi * self.semi_major[index] * self.semi_minor[index]
            elif kind is SectionKind.SEPTUM:
                out[index] = math.pi * self.semi_major[index] ** 2 * float(self.channels[index])
            else:
                out[index] = 0.0
        for index in range(self.n_slices):
            if self.collateral_grade[index] > 0:
                out[index] += float(self.collateral_grade[index]) * math.pi * COLLATERAL_RADIUS_MM**2
        return out

    def unit_shape_factor(self) -> FloatArray:
        area = self.unit_area()
        conductance = self.unit_conductance()
        kappa = np.full(self.n_slices, np.nan, dtype=np.float64)
        for index, kind in enumerate(self.kind):
            if kind is SectionKind.ELLIPSE:
                kappa[index] = ellipse_shape_factor(self.semi_major[index], self.semi_minor[index])
            elif kind is SectionKind.DISC:
                kappa[index] = 1.0
            elif kind is SectionKind.SEPTUM:
                kappa[index] = shape_factor(area[index], conductance[index])
            else:
                kappa[index] = float("nan")
        return kappa

    def unit_channels(self) -> IntArray:
        out = np.zeros(self.n_slices, dtype=np.int64)
        for index, kind in enumerate(self.kind):
            if kind is SectionKind.OCCLUDED:
                out[index] = 0
            elif kind is SectionKind.SEPTUM:
                out[index] = int(self.channels[index])
            else:
                out[index] = 1
            if self.collateral_grade[index] > 0:
                out[index] += int(self.collateral_grade[index])
        return out

    def lesion_span_mm(self) -> float:
        start, stop = self.lesion_slice_range
        return float(stop - start) * float(self.grid.spacing_zyx[0])

    def segment_of_slice(self) -> IntArray:
        position = np.arange(self.n_slices, dtype=np.float64) / max(self.n_slices - 1, 1)
        bounds = np.asarray(DEFAULT_SEGMENT_FRACTIONS, dtype=np.float64)
        index = np.searchsorted(bounds, position, side="right") - 1
        return np.clip(index, 0, 2).astype(np.int64)


def baseline_radius(z_mm: FloatArray, total_length: float) -> FloatArray:
    """Confluence-to-bifurcation calibre with a smooth taper and a mild wobble."""
    fraction = np.clip(z_mm / max(total_length, 1e-6), 0.0, 1.0)
    taper = CONFLUENCE_RADIUS_MM + (BIFURCATION_RADIUS_MM - CONFLUENCE_RADIUS_MM) * fraction
    wobble = WOBBLE_AMPLITUDE_MM * np.sin(2.0 * math.pi * z_mm / WOBBLE_WAVELENGTH_MM)
    return taper + wobble


def build_profile(
    grid: GridSpec,
    lesion_type: LesionType,
    lesion_segment: Segment,
    axis_ratio: float,
    rng: np.random.Generator,
    collateral_grade: int = 0,
) -> LimbProfile:
    """Prescribe the closed-form cross-section of every slice of a limb."""
    nz = grid.shape[0]
    dz = grid.spacing_zyx[0]
    z_mm = np.arange(nz, dtype=np.float64) * dz
    total_length = float(nz - 1) * dz
    radius = baseline_radius(z_mm, total_length)

    segment_index = np.clip(
        np.searchsorted(
            np.asarray(DEFAULT_SEGMENT_FRACTIONS, dtype=np.float64),
            z_mm / max(total_length, 1e-6),
            side="right",
        )
        - 1,
        0,
        2,
    ).astype(np.int64)
    members = np.flatnonzero(segment_index == int(lesion_segment))
    if members.size == 0:
        raise ValueError("the requested lesion segment has no slices")
    centre_slice = int(round(LESION_CENTRE_FRACTION * (members.size - 1))) + int(members[0])
    half_span = max(1, int(round(0.5 * LESION_LENGTH_MM / dz)))
    start = max(int(members[0]), centre_slice - half_span)
    stop = min(int(members[-1]) + 1, centre_slice + half_span + 1)

    kind: list[SectionKind] = [SectionKind.DISC] * nz
    semi_major = np.array(radius, dtype=np.float64)
    semi_minor = np.array(radius, dtype=np.float64)
    channels = np.ones(nz, dtype=np.int64)
    grades = np.zeros(nz, dtype=np.int64)

    if lesion_type is LesionType.COMPRESSIVE_WITH_COLLATERALS and collateral_grade <= 0:
        collateral_grade = int(rng.integers(1, 4))
    if lesion_type is LesionType.COMPRESSIVE and collateral_grade > 0:
        collateral_grade = 0

    for index in range(start, stop):
        reference = float(radius[index])
        if lesion_type in (LesionType.COMPRESSIVE, LesionType.COMPRESSIVE_WITH_COLLATERALS):
            kind[index] = SectionKind.ELLIPSE
            semi_major[index] = reference
            semi_minor[index] = reference * float(axis_ratio)
        elif lesion_type is LesionType.POST_THROMBOTIC:
            kind[index] = SectionKind.SEPTUM
            channels[index] = int(SEPTUM_CHANNELS[index % len(SEPTUM_CHANNELS)])
            area = math.pi * reference * reference
            semi_major[index] = math.sqrt(area / (math.pi * int(channels[index])))
            semi_minor[index] = semi_major[index]
        else:
            kind[index] = SectionKind.OCCLUDED
            semi_major[index] = 0.0
            semi_minor[index] = 0.0
            channels[index] = 0
        grades[index] = collateral_grade

    centre_u = (0.6 * np.sin(2.0 * math.pi * z_mm / 120.0)).astype(np.float64)
    centre_v = (0.3 * np.cos(2.0 * math.pi * z_mm / 90.0)).astype(np.float64)
    centreline = np.stack([z_mm, centre_u, centre_v], axis=1)
    return LimbProfile(
        grid=grid,
        kind=tuple(kind),
        semi_major=semi_major,
        semi_minor=semi_minor,
        channels=channels,
        collateral_grade=grades,
        centre_u=centre_u,
        centre_v=centre_v,
        centreline=centreline,
        lesion_slice_range=(start, stop),
    )


def rasterise_section(spec: SectionSpec, spacing_uv: tuple[float, float], shape: tuple[int, int]) -> BoolArray:
    """Rasterise one prescribed cross-section onto an in-plane grid."""
    if spec.kind is SectionKind.OCCLUDED:
        return np.zeros(shape, dtype=bool)
    du, dv = spacing_uv
    rows = (np.arange(shape[0], dtype=np.float64) - (shape[0] - 1) / 2.0) * du
    cols = (np.arange(shape[1], dtype=np.float64) - (shape[1] - 1) / 2.0) * dv
    grid_u, grid_v = np.meshgrid(rows, cols, indexing="ij")
    u = grid_u - spec.centre_uv[0]
    v = grid_v - spec.centre_uv[1]
    if spec.kind is SectionKind.DISC:
        occupancy = (u * u + v * v) <= spec.semi_major**2
    elif spec.kind is SectionKind.ELLIPSE:
        occupancy = (u / spec.semi_major) ** 2 + (v / spec.semi_minor) ** 2 <= 1.0
    else:
        occupancy = np.zeros(shape, dtype=bool)
        pitch = 2.0 * spec.semi_major + 1.0
        span = (spec.channels - 1) * pitch
        for channel in range(spec.channels):
            offset = -0.5 * span + channel * pitch
            occupancy |= u * u + (v - offset) ** 2 <= spec.semi_major**2
    if spec.collateral_grade > 0:
        for grade in range(spec.collateral_grade):
            angle = 2.0 * math.pi * grade / spec.collateral_grade
            cu = 1.9 * spec.semi_major * math.cos(angle)
            cv = 1.9 * spec.semi_major * math.sin(angle)
            occupancy |= (u - cu) ** 2 + (v - cv) ** 2 <= COLLATERAL_RADIUS_MM**2
    return occupancy


def profile_to_spec(profile: LimbProfile, index: int) -> SectionSpec:
    return SectionSpec(
        kind=profile.kind[index],
        semi_major=float(profile.semi_major[index]),
        semi_minor=float(profile.semi_minor[index]),
        channels=int(profile.channels[index]),
        collateral_grade=int(profile.collateral_grade[index]),
        centre_uv=(float(profile.centre_u[index]), float(profile.centre_v[index])),
    )


def render_mask(profile: LimbProfile) -> BoolArray:
    """Rasterise a limb profile into a voxel mask indexed ``[z, y, x]``."""
    nz, ny, nx = profile.grid.shape
    du, dv = profile.grid.spacing_zyx[1], profile.grid.spacing_zyx[2]
    mask = np.zeros((nz, ny, nx), dtype=bool)
    for index in range(nz):
        spec = profile_to_spec(profile, index)
        spec = SectionSpec(
            kind=spec.kind,
            semi_major=spec.semi_major,
            semi_minor=spec.semi_minor,
            channels=spec.channels,
            collateral_grade=spec.collateral_grade,
            centre_uv=(0.0, 0.0),
        )
        plane = rasterise_section(spec, (du, dv), (ny, nx))
        mask[index] = _shift_plane(plane, profile.centre_u[index] / du, profile.centre_v[index] / dv)
    return mask


def _shift_plane(plane: BoolArray, shift_u: float, shift_v: float) -> BoolArray:
    """Integer-shift an in-plane mask by an array offset, filling with lumen-free."""
    ny, nx = plane.shape
    rows = np.arange(ny, dtype=np.float64) - shift_u
    cols = np.arange(nx, dtype=np.float64) - shift_v
    source_rows = np.clip(np.rint(rows).astype(np.int64), 0, ny - 1)
    source_cols = np.clip(np.rint(cols).astype(np.int64), 0, nx - 1)
    shifted = plane[np.ix_(source_rows, source_cols)]
    inside_rows = (np.rint(rows) >= 0) & (np.rint(rows) < ny)
    inside_cols = (np.rint(cols) >= 0) & (np.rint(cols) < nx)
    valid = inside_rows[:, None] & inside_cols[None, :]
    shifted_valid: BoolArray = shifted & valid
    return shifted_valid


def render_intensity(
    profile: LimbProfile, phase: AcquisitionPhase, vendor: str, rng: np.random.Generator
) -> FloatArray:
    """Write a contrast-enhanced volume consistent with the mask and phase."""
    mask = render_mask(profile)
    lumen_hu = float(CONTRAST_HU[phase]) * float(VENDOR_GAIN.get(vendor, 1.0))
    volume = np.full(profile.grid.shape, TISSUE_HU, dtype=np.float64)
    dilated = _dilate3(mask)
    wall = dilated & ~mask
    volume[wall] = WALL_HU
    attenuation = np.zeros(profile.grid.shape, dtype=np.float64)
    attenuation[mask] = lumen_hu
    for index, kind in enumerate(profile.kind):
        if kind is SectionKind.OCCLUDED:
            attenuation[index] = np.where(dilated[index], TISSUE_HU + 18.0, TISSUE_HU)
        elif kind is SectionKind.SEPTUM:
            plane = mask[index]
            septa = plane & ~_erode2(plane)
            attenuation[index][septa] = TISSUE_HU + 55.0
    volume = np.where(mask | wall, attenuation, volume)
    return volume + rng.normal(0.0, 6.0, size=volume.shape)


def _dilate3(mask: BoolArray) -> BoolArray:
    out = np.array(mask, dtype=bool)
    for axis in range(3):
        out |= np.roll(mask, 1, axis=axis)
        out |= np.roll(mask, -1, axis=axis)
    return out


def _erode2(plane: BoolArray) -> BoolArray:
    out = np.array(plane, dtype=bool)
    out[1:, :] &= plane[:-1, :]
    out[:-1, :] &= plane[1:, :]
    out[:, 1:] &= plane[:, :-1]
    out[:, :-1] &= plane[:, 1:]
    return out


def analytic_descriptor_vector(profile: LimbProfile) -> FloatArray:
    """Descriptor vector of a limb from the closed forms, with no solver involved.

    Matches the seven fields of :data:`conductance_ctv.hydraulics.descriptors.DESCRIPTOR_FIELDS`
    per segment, so a solver-based signature can be differenced against it.
    """
    area = profile.unit_area()
    conductance = profile.unit_conductance()
    kappa = profile.unit_shape_factor()
    channels = profile.unit_channels()
    labels = profile.segment_of_slice()
    ds = float(profile.grid.spacing_zyx[0])
    z_mm = np.arange(profile.n_slices, dtype=np.float64) * ds

    spans: dict[int, IntArray] = {}
    for index in range(3):
        members = np.flatnonzero(labels == index)
        if members.size:
            spans[index] = members.astype(np.int64)
    means = {index: float(np.mean(conductance[members])) for index, members in spans.items()}

    vector: list[float] = []
    for index in range(3):
        if index not in spans:
            vector.extend([0.0] * 7)
            continue
        members = spans[index]
        span_g = conductance[members]
        span_a = area[members]
        span_k = kappa[members]
        span_n = channels[members]
        patent = span_g > 0.0
        if bool(patent.any()):
            star = int(np.argmin(np.where(patent, span_g, np.inf)))
        else:
            star = int(np.argmin(span_a))
        resistance = float(np.sum(ds / span_g[patent])) if bool(patent.any()) else math.inf
        rivals = [means[other] for other in means if other != index]
        reference = max(rivals) if rivals else 0.0
        if math.isfinite(resistance) and reference > 0.0:
            ratio = resistance * reference / (float(members.size) * ds)
            weighted = float(np.sum(span_g / reference) * ds)
        else:
            ratio = math.inf
            weighted = 0.0
        vector.extend(
            [
                float(span_a[star]),
                float(z_mm[members[star]]),
                float(span_k[star]),
                float(span_n[star]),
                float(resistance),
                float(ratio),
                float(weighted),
            ]
        )
    return np.array(vector, dtype=np.float64)


def draw_lesion_type(rng: np.random.Generator, prevalence: tuple[float, float, float, float]) -> LesionType:
    probabilities = np.asarray(prevalence, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return LesionType(int(rng.choice(np.arange(1, 5), p=probabilities)))


def generate_limb(
    limb_id: str,
    patient_id: str,
    site: Site,
    rng: np.random.Generator,
    grid: GridSpec = DEFAULT_GRID,
    lesion_type: LesionType | None = None,
    lesion_segment: Segment = Segment.CIV,
    axis_ratio: float | None = None,
    collateral_grade: int = 0,
    prevalence: tuple[float, float, float, float] = TYPE_PREVALENCE,
) -> tuple[LimbProfile, LimbRecord]:
    """Generate one limb profile and the record that goes with it."""
    resolved_type = lesion_type if lesion_type is not None else draw_lesion_type(rng, prevalence)
    ratio = float(axis_ratio) if axis_ratio is not None else float(rng.uniform(0.15, 0.30))
    profile = build_profile(grid, resolved_type, lesion_segment, ratio, rng, collateral_grade)

    area = profile.unit_area()
    conductance = profile.unit_conductance()
    start, stop = profile.lesion_slice_range
    reference_area = float(np.median(area[: max(start, 1)])) if start > 0 else float(np.median(area))
    lesion_area = float(np.min(area[start:stop])) if stop > start else reference_area
    stenosis = 1.0 - math.sqrt(max(lesion_area, 0.0) / max(reference_area, 1e-9))
    minimal_area = float(np.min(area))
    lesion_length = float(np.count_nonzero(conductance[start:stop] <= 0.0)) * float(grid.spacing_zyx[0])
    if lesion_length == 0.0:
        lesion_length = profile.lesion_span_mm()

    vendor = SITE_VENDORS[site][int(rng.integers(0, len(SITE_VENDORS[site])))]
    phase = SITE_PHASE[site]
    grade = int(profile.collateral_grade[start])

    first = _reader_call(resolved_type, rng)
    second = _reader_call(resolved_type, rng)
    record = LimbRecord(
        limb_id=limb_id,
        patient_id=patient_id,
        site=site,
        phase=phase,
        vendor=vendor,
        reference_type=resolved_type,
        reader_types=(first, second),
        adjudicated=first != second,
        lesion_segment=lesion_segment,
        collateral_grade=grade,
        axis_ratio=float(ratio),
        stenosis_ratio=float(max(0.0, min(1.0, stenosis))),
        minimal_luminal_area=minimal_area,
        lesion_length=lesion_length,
        intraluminal_spur=bool(resolved_type is LesionType.POST_THROMBOTIC),
        descriptor_vector=analytic_descriptor_vector(profile),
    )
    return profile, record


def _reader_call(reference: LesionType, rng: np.random.Generator) -> LesionType:
    """A single reader call: the consensus with a limited, type-adjacent error rate.

    Measured with the two-reader agreement of Sec. 6.3 in mind: disagreements
    concentrate on the type II/type III boundary, which is the pair of types the
    manuscript reports the larger error count for.
    """
    roll = float(rng.random())
    if reference is LesionType.COMPRESSIVE_WITH_COLLATERALS:
        if roll < 0.075:
            return LesionType.POST_THROMBOTIC
        if roll < 0.095:
            return LesionType.COMPRESSIVE
    elif reference is LesionType.POST_THROMBOTIC:
        if roll < 0.070:
            return LesionType.COMPRESSIVE_WITH_COLLATERALS
        if roll < 0.090:
            return LesionType.OCCLUSIVE
    elif reference is LesionType.COMPRESSIVE:
        if roll < 0.045:
            return LesionType.COMPRESSIVE_WITH_COLLATERALS
    else:
        if roll < 0.040:
            return LesionType.POST_THROMBOTIC
    return reference


def generate_cohort(
    n_patients: int,
    seed: int,
    site_mix: tuple[float, float, float] = (0.42, 0.36, 0.22),
    prevalence: tuple[float, float, float, float] = TYPE_PREVALENCE,
    grid: GridSpec = DEFAULT_GRID,
    two_limb_fraction: float = 0.68,
    association_model: dict[str, float] | None = None,
) -> CohortRecord:
    """Generate a three-site cohort with patient-nested limbs.

    Both limbs of a patient are generated whenever a patient is two-limbed, and
    they share the patient-level presentation variables, which is what makes the
    clustered resampling of Sec. 6.8 materially different from an unclustered one.
    """
    if n_patients < 1:
        raise ValueError("a cohort needs at least one patient")
    rng = np.random.default_rng(seed)
    sites = (Site.A, Site.B, Site.C)
    mix = np.asarray(site_mix, dtype=np.float64)
    mix = mix / mix.sum()
    model = dict(association_model) if association_model else dict(DEFAULT_ASSOCIATION_MODEL)

    patients: list[PatientRecord] = []
    for index in range(n_patients):
        site = sites[int(rng.choice(np.arange(3), p=mix))]
        patient_id = f"P{index:05d}"
        n_limbs = 2 if float(rng.random()) < two_limb_fraction else 1
        lesions: list[tuple[LimbProfile, LimbRecord]] = []
        for side in range(n_limbs):
            limb_id = f"{patient_id}-L{side + 1}"
            lesions.append(
                generate_limb(
                    limb_id,
                    patient_id,
                    site,
                    rng,
                    grid=grid,
                    prevalence=prevalence,
                )
            )
        patient = _assemble_patient(patient_id, site, lesions, model, rng)
        patients.append(patient)

    return CohortRecord(
        patients=tuple(patients),
        seed=seed,
        type_prevalence=prevalence,
        association_model=model,
        reader_noise={"ii_iii_boundary": 0.075},
    )


DEFAULT_ASSOCIATION_MODEL: Final[dict[str, float]] = {
    "intercept": -1.10,
    "type_iii_iv": 1.17,
    "stenosis_per_10pp": 0.55,
    "resistance_ratio_per_sd": 0.65,
    "shape_factor_per_sd": 0.48,
    "age_per_decade": 0.14,
}


def _assemble_patient(
    patient_id: str,
    site: Site,
    lesions: list[tuple[LimbProfile, LimbRecord]],
    model: dict[str, float],
    rng: np.random.Generator,
) -> PatientRecord:
    age = float(np.clip(rng.normal(58.0, 13.0), 19.0, 92.0))
    sex = "female" if float(rng.random()) < 0.54 else "male"

    severe = sum(
        1 for _, record in lesions if record.reference_type in (LesionType.POST_THROMBOTIC, LesionType.OCCLUSIVE)
    )
    mean_stenosis = float(np.mean([record.stenosis_ratio for _, record in lesions]))
    linear = (
        model["intercept"]
        + model["type_iii_iv"] * severe
        + model["stenosis_per_10pp"] * (mean_stenosis * 100.0 / 10.0)
        + model["age_per_decade"] * ((age - 58.0) / 10.0)
    )
    probability = 1.0 / (1.0 + math.exp(-linear))
    dvt_history = bool(rng.random() < probability)
    ceap_class = int(np.clip(round(rng.normal(3.4 + 0.9 * severe, 1.0)), 0, 6))
    claudication = bool(rng.random() < 0.5 * probability + 0.08)
    laterality = "left" if float(rng.random()) < 0.5 + 0.12 * severe else "right"
    composite = float(np.clip(2.4 + 1.1 * severe + 1.6 * mean_stenosis + rng.normal(0.0, 0.8), 0.0, 12.0))
    return PatientRecord(
        patient_id=patient_id,
        site=site,
        age=age,
        sex=sex,
        dvt_history=dvt_history,
        ceap_class=ceap_class,
        venous_claudication=claudication,
        symptom_laterality=laterality,
        composite_score=composite,
        limbs=tuple(record for _, record in lesions),
    )
