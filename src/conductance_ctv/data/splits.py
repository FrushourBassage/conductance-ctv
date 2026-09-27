"""Patient-level splits, site transport and cross-fitting folds.

The nesting of the analysis units drives every split here. Limbs of one patient
are correlated, so a random limb-level split leaks the patient into both sides
of the evaluation; the manuscript therefore fixes a patient-level split and
cross-fits at the patient level during segmentation training, and never
substitutes a same-source random split for the external evaluation.

Ref: Sec. 6.4 (patient-level cross-fitting), Sec. 6.2 and Algorithm 3 lines 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.data.schema import (
    DEVELOPMENT_SITES,
    EXTERNAL_SITE,
    CohortRecord,
    LesionType,
    LimbRecord,
    Site,
)
from conductance_ctv.utils.types import IntArray


@dataclass(frozen=True)
class PatientSplit:
    """A train/tune/external partition of a cohort, indexed by patient."""

    train_patients: tuple[str, ...]
    tune_patients: tuple[str, ...]
    external_patients: tuple[str, ...]
    seed: int
    tune_fraction: float

    @property
    def all_development(self) -> tuple[str, ...]:
        return self.train_patients + self.tune_patients

    def is_train(self, patient_id: str) -> bool:
        return patient_id in self.train_patients

    def masks(self, cohort: CohortRecord) -> tuple[IntArray, IntArray, IntArray]:
        """Boolean index masks over ``cohort.limbs`` for the three partitions."""
        train = set(self.train_patients)
        tune = set(self.tune_patients)
        external = set(self.external_patients)
        train_mask = np.array([limb.patient_id in train for limb in cohort.limbs], dtype=bool)
        tune_mask = np.array([limb.patient_id in tune for limb in cohort.limbs], dtype=bool)
        external_mask = np.array([limb.patient_id in external for limb in cohort.limbs], dtype=bool)
        return train_mask, tune_mask, external_mask

    def record(self) -> dict[str, object]:
        return {
            "train_patients": len(self.train_patients),
            "tune_patients": len(self.tune_patients),
            "external_patients": len(self.external_patients),
            "seed": self.seed,
            "tune_fraction": self.tune_fraction,
        }


def patient_level_split(cohort: CohortRecord, tune_fraction: float = 0.2, seed: int = 20260926) -> PatientSplit:
    """Split development patients into a training and a tuning partition.

    Patients from the external site are held out wholesale; the split is drawn
    over patients and not over limbs, so both limbs of a patient land together.
    """
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("the tuning fraction must lie strictly inside (0, 1)")
    rng = np.random.default_rng(seed)
    development = [p.patient_id for p in cohort.patients if p.site in DEVELOPMENT_SITES]
    external = [p.patient_id for p in cohort.patients if p.site == EXTERNAL_SITE]
    order = rng.permutation(len(development))
    n_tune = max(1, int(round(tune_fraction * len(development))))
    tune = tuple(sorted(development[index] for index in order[:n_tune]))
    train = tuple(sorted(development[index] for index in order[n_tune:]))
    return PatientSplit(
        train_patients=train,
        tune_patients=tune,
        external_patients=tuple(sorted(external)),
        seed=seed,
        tune_fraction=tune_fraction,
    )


def crossfit_folds(patient_ids: tuple[str, ...], n_folds: int, seed: int = 20260926) -> tuple[tuple[str, ...], ...]:
    """Disjoint patient folds for the cross-fitting of Sec. 6.4.

    Every development patient appears in exactly one validation fold, so a limb
    is scored only by a segmenter that never saw its patient.
    """
    if n_folds < 2:
        raise ValueError("cross-fitting needs at least two folds")
    if len(patient_ids) < n_folds:
        raise ValueError("fewer patients than folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patient_ids))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for position, index in enumerate(order):
        folds[position % n_folds].append(patient_ids[int(index)])
    return tuple(tuple(sorted(fold)) for fold in folds)


def leave_one_site_out(
    cohort: CohortRecord, order: tuple[Site, ...] = (Site.A, Site.B, Site.C)
) -> dict[str, PatientSplit]:
    """One split per site, each holding that site out as the evaluation set.

    The manuscript reports leave-one-site-out values to separate a regional from
    a site-specific effect.

    Ref: Sec. 3.3 and Supplementary Table S15.
    """
    out: dict[str, PatientSplit] = {}
    for held_out in order:
        development = tuple(sorted(p.patient_id for p in cohort.patients if p.site != held_out))
        held = tuple(sorted(p.patient_id for p in cohort.patients if p.site == held_out))
        out[str(held_out)] = PatientSplit(
            train_patients=development,
            tune_patients=(),
            external_patients=held,
            seed=cohort.seed,
            tune_fraction=0.0,
        )
    return out


def site_stratified_patients(cohort: CohortRecord, fraction: float, seed: int) -> dict[Site, tuple[str, ...]]:
    """Split each site's patients into two groups at a fixed fraction."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("the fraction must lie strictly inside (0, 1)")
    rng = np.random.default_rng(seed)
    out: dict[Site, tuple[str, ...]] = {}
    for site in Site:
        ids = sorted(p.patient_id for p in cohort.patients if p.site == site)
        order = rng.permutation(len(ids))
        cut = max(1, int(round(fraction * len(ids))))
        out[site] = tuple(sorted(ids[int(index)] for index in order[:cut]))
    return out


def accrual_halves(cohort: CohortRecord, seed: int = 20260926) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the accrual into two halves for the temporal replication of Sec. 3.3.

    A generated cohort has no accrual calendar, so the halves are drawn per site
    at the patient level; this keeps the temporal check executable while the
    calendar-based version of it needs the site records that are not shared.
    """
    first: list[str] = []
    second: list[str] = []
    for offset, site in enumerate(Site):
        ids = sorted(p.patient_id for p in cohort.patients if p.site == site)
        rng = np.random.default_rng(seed + 1013 * offset)
        order = rng.permutation(len(ids))
        cut = len(ids) // 2
        first.extend(ids[int(index)] for index in order[:cut])
        second.extend(ids[int(index)] for index in order[cut:])
    return tuple(sorted(first)), tuple(sorted(second))


def reference_type_counts(limbs: tuple[LimbRecord, ...] | list[LimbRecord]) -> IntArray:
    counts = np.zeros(4, dtype=np.int64)
    for limb in limbs:
        counts[int(limb.reference_type) - 1] += 1
    return counts


def stratum_of(limb: LimbRecord) -> str:
    """Fine stratum label used for the fairness-constrained variant.

    Ref: Table 3 (fairness-debiased row) and Supplementary Table S18.
    """
    band = "young" if limb.site is Site.A else "older"
    severity = "severe" if limb.reference_type in (LesionType.POST_THROMBOTIC, LesionType.OCCLUSIVE) else "mild"
    return f"{limb.site}|{band}|{severity}"
