"""Cohort schema: analysis units, the four lesion types and the reader records.

The manuscript keeps three nested analysis units apart, and every reported
quantity is tied to one of them: the segment (common iliac, external iliac,
common femoral vein on each limb), the limb, and the patient. The schema follows
that nesting so a metric cannot silently be computed at the wrong level.

Ref: Sec. 6.1 (the four types), Sec. 6.2 (analysis units), Sec. 6.3 (the
two-reader consensus with third-reader adjudication).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np

from conductance_ctv.utils.types import FloatArray, IntArray

SEGMENT_LABELS: tuple[str, ...] = ("CIV", "EIV", "CFV")
TYPE_LABELS: tuple[str, ...] = ("I", "II", "III", "IV")
LESION_TYPE_COUNT = 4


class LesionType(IntEnum):
    """The morphological type of Eq. (3) in Sec. 6.1."""

    COMPRESSIVE = 1
    COMPRESSIVE_WITH_COLLATERALS = 2
    POST_THROMBOTIC = 3
    OCCLUSIVE = 4


class Segment(IntEnum):
    CIV = 0
    EIV = 1
    CFV = 2


class AcquisitionPhase(StrEnum):
    DIRECT = "direct"
    EARLY_INDIRECT = "early_indirect"
    DELAYED_INDIRECT = "delayed_indirect"


class Site(StrEnum):
    A = "Site A"
    B = "Site B"
    C = "Site C"


class Region(StrEnum):
    REGION_I = "Region I"
    REGION_II = "Region II"


SITE_REGION: dict[Site, Region] = {
    Site.A: Region.REGION_I,
    Site.B: Region.REGION_I,
    Site.C: Region.REGION_II,
}

DEVELOPMENT_SITES: tuple[Site, ...] = (Site.A, Site.B)
EXTERNAL_SITE = Site.C

LESION_TYPE_DESCRIPTION: dict[LesionType, str] = {
    LesionType.COMPRESSIVE: "obstructive compression without thrombus",
    LesionType.COMPRESSIVE_WITH_COLLATERALS: "obstructive compression with collateral development",
    LesionType.POST_THROMBOTIC: "post-thrombotic stenosis from previous thrombosis",
    LesionType.OCCLUSIVE: "complete obstruction",
}


class SectionKind(StrEnum):
    DISC = "disc"
    ELLIPSE = "ellipse"
    SEPTUM = "septum"
    OCCLUDED = "occluded"


@dataclass(frozen=True)
class GridSpec:
    """Voxel grid of a computed tomography venography volume."""

    shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value < 3 for value in self.shape):
            raise ValueError("a volume needs three axes of at least three voxels")
        if any(value <= 0.0 for value in self.spacing_zyx):
            raise ValueError("voxel spacing must be strictly positive")

    @property
    def extent_zyx(self) -> tuple[float, float, float]:
        return (
            float(self.shape[0]) * float(self.spacing_zyx[0]),
            float(self.shape[1]) * float(self.spacing_zyx[1]),
            float(self.shape[2]) * float(self.spacing_zyx[2]),
        )

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(np.asarray(self.spacing_zyx, dtype=np.float64)))


@dataclass(frozen=True)
class SectionSpec:
    """The closed-form cross-section prescribed at one arc-length sample."""

    kind: SectionKind
    semi_major: float = 0.0
    semi_minor: float = 0.0
    channels: int = 1
    collateral_grade: int = 0
    centre_uv: tuple[float, float] = (0.0, 0.0)

    @property
    def collateralised(self) -> bool:
        return self.collateral_grade > 0


@dataclass(frozen=True)
class LimbRecord:
    """One limb: its imaging metadata, its reference type and its descriptors."""

    limb_id: str
    patient_id: str
    site: Site
    phase: AcquisitionPhase
    vendor: str
    reference_type: LesionType
    reader_types: tuple[LesionType, LesionType]
    adjudicated: bool
    lesion_segment: Segment
    collateral_grade: int
    axis_ratio: float
    stenosis_ratio: float
    minimal_luminal_area: float
    lesion_length: float
    intraluminal_spur: bool
    descriptor_vector: FloatArray
    n_segments: int = 3

    @property
    def region(self) -> Region:
        return SITE_REGION[self.site]

    @property
    def reader_consensus(self) -> LesionType | None:
        first, second = self.reader_types
        return first if first == second else None

    def record(self) -> dict[str, object]:
        return {
            "limb_id": self.limb_id,
            "patient_id": self.patient_id,
            "site": str(self.site),
            "region": str(self.region),
            "phase": str(self.phase),
            "vendor": self.vendor,
            "reference_type": int(self.reference_type),
            "reader_types": [int(value) for value in self.reader_types],
            "adjudicated": self.adjudicated,
            "lesion_segment": int(self.lesion_segment),
            "collateral_grade": self.collateral_grade,
            "axis_ratio": self.axis_ratio,
            "stenosis_ratio": self.stenosis_ratio,
            "minimal_luminal_area": self.minimal_luminal_area,
            "lesion_length": self.lesion_length,
            "intraluminal_spur": self.intraluminal_spur,
        }


@dataclass(frozen=True)
class PatientRecord:
    """One patient: demographics, presentation variables and their limbs."""

    patient_id: str
    site: Site
    age: float
    sex: str
    dvt_history: bool
    ceap_class: int
    venous_claudication: bool
    symptom_laterality: str
    composite_score: float
    limbs: tuple[LimbRecord, ...]

    @property
    def region(self) -> Region:
        return SITE_REGION[self.site]

    @property
    def n_limbs(self) -> int:
        return len(self.limbs)

    def presentation(self) -> dict[str, float]:
        """Presentation endpoints of Table 7, as numeric columns."""
        return {
            "dvt_history": float(self.dvt_history),
            "ceap_c4_c6": float(self.ceap_class >= 4),
            "venous_claudication": float(self.venous_claudication),
            "symptom_laterality": float(self.symptom_laterality == "left"),
            "composite_score": float(self.composite_score),
        }


@dataclass(frozen=True)
class CohortRecord:
    """A generated cohort with the provenance of its generating parameters."""

    patients: tuple[PatientRecord, ...]
    seed: int
    type_prevalence: tuple[float, float, float, float]
    association_model: dict[str, float] = field(default_factory=dict)
    reader_noise: dict[str, float] = field(default_factory=dict)

    @property
    def limbs(self) -> tuple[LimbRecord, ...]:
        return tuple(limb for patient in self.patients for limb in patient.limbs)

    @property
    def n_patients(self) -> int:
        return len(self.patients)

    @property
    def n_limbs(self) -> int:
        return sum(patient.n_limbs for patient in self.patients)

    def limb_matrix(self) -> FloatArray:
        return np.stack([limb.descriptor_vector for limb in self.limbs], axis=0)

    def type_labels(self) -> IntArray:
        return np.array([int(limb.reference_type) for limb in self.limbs], dtype=np.int64)

    def patient_index(self) -> IntArray:
        lookup = {patient.patient_id: index for index, patient in enumerate(self.patients)}
        return np.array([lookup[limb.patient_id] for limb in self.limbs], dtype=np.int64)

    def site_labels(self) -> tuple[Site, ...]:
        return tuple(limb.site for limb in self.limbs)

    def subset(self, site: Site) -> CohortRecord:
        """Cohort restricted to one site, preserving patient nesting."""
        patients = []
        for patient in self.patients:
            limbs = tuple(limb for limb in patient.limbs if limb.site == site)
            if limbs:
                patients.append(
                    PatientRecord(
                        patient_id=patient.patient_id,
                        site=patient.site,
                        age=patient.age,
                        sex=patient.sex,
                        dvt_history=patient.dvt_history,
                        ceap_class=patient.ceap_class,
                        venous_claudication=patient.venous_claudication,
                        symptom_laterality=patient.symptom_laterality,
                        composite_score=patient.composite_score,
                        limbs=limbs,
                    )
                )
        return CohortRecord(
            patients=tuple(patients),
            seed=self.seed,
            type_prevalence=self.type_prevalence,
            association_model=dict(self.association_model),
            reader_noise=dict(self.reader_noise),
        )

    def records(self) -> list[dict[str, object]]:
        return [limb.record() for limb in self.limbs]


def confusion_counts(predicted: IntArray, reference: IntArray, n_classes: int = LESION_TYPE_COUNT) -> IntArray:
    """Row-reference, column-prediction confusion counts starting at type I."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(reference, predicted, strict=True):
        matrix[int(truth) - 1, int(guess) - 1] += 1
    return matrix
