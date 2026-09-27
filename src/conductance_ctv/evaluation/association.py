"""The association analysis of the presentation endpoints.

Two nested models are compared per endpoint: a degree-only model carrying the
stenosis ratio and the conventional covariates, and the same model augmented with
the four-type call and the hydraulic descriptors. The comparison is a
likelihood-ratio test with as many degrees of freedom as there are added
parameters, and it is reported together with the change in the area under the
curve, the continuous net reclassification improvement and the decision-curve net
benefit at the pre-specified threshold.

Ref: Table 7 and its caption; Sec. 6.8 (association framework).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conductance_ctv.data.schema import CohortRecord, LesionType
from conductance_ctv.hydraulics.descriptors import DESCRIPTOR_FIELDS, SEGMENT_NAMES
from conductance_ctv.metrics.classification import auroc
from conductance_ctv.metrics.inference_stats import (
    adjusted_odds_ratio,
    continuous_nri,
    design_matrix,
    likelihood_ratio_test,
    log_likelihood_of_design,
    net_benefit_gain,
)
from conductance_ctv.utils.types import FloatArray, IntArray

PRESENTATION_ENDPOINTS: tuple[str, ...] = (
    "dvt_history",
    "ceap_c4_c6",
    "venous_claudication",
    "symptom_laterality",
    "composite_score_ge_median",
)

CONVENTIONAL_COVARIATES: tuple[str, ...] = ("age_per_decade", "sex_female")


def _descriptor_column(limb_vector: FloatArray, segment: str, field_name: str) -> float:
    index = SEGMENT_NAMES.index(segment) * len(DESCRIPTOR_FIELDS) + DESCRIPTOR_FIELDS.index(field_name)
    return float(limb_vector[index])


@dataclass
class AssociationTable:
    """Rows of the association analysis, ready for the table builder."""

    degree_only_auc: dict[str, float] = field(default_factory=dict)
    extended_auc: dict[str, float] = field(default_factory=dict)
    likelihood_ratio: dict[str, dict[str, float]] = field(default_factory=dict)
    odds_ratios: dict[str, dict[str, float]] = field(default_factory=dict)
    net_renclassification: dict[str, float] = field(default_factory=dict)
    net_benefit: float = float("nan")
    threshold: float = 0.5
    units: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "degree_only_auc": self.degree_only_auc,
            "extended_auc": self.extended_auc,
            "likelihood_ratio": self.likelihood_ratio,
            "odds_ratios": self.odds_ratios,
            "nri": self.net_renclassification,
            "net_benefit": self.net_benefit,
            "threshold": self.threshold,
            "units": self.units,
            "notes": self.notes,
        }


def build_columns(cohort: CohortRecord, rows: IntArray | None = None) -> dict[str, FloatArray]:
    """Per-limb columns of the association design, at the limb unit.

    Every column is at the limb unit because the type call and the descriptors
    are limb-level; the presentation variables are patient-level and are repeated
    on each limb, which is why the resampling has to stay at the patient.
    """
    limbs = list(cohort.limbs)
    lookup = {patient.patient_id: patient for patient in cohort.patients}
    selection = np.arange(len(limbs), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    columns: dict[str, list[float]] = {name: [] for name in _column_names()}
    for index in selection:
        limb = limbs[int(index)]
        patient = lookup[limb.patient_id]
        entries = _limb_entries(limb, patient)
        for name, value in entries.items():
            columns[name].append(float(value))
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def _column_names() -> tuple[str, ...]:
    return (
        "stenosis_ratio",
        "age_per_decade",
        "sex_female",
        "type_iii_iv",
        "resistance_ratio_civ",
        "shape_factor_civ",
        "minimum_luminal_area_civ",
        "composite_score",
        "ceap_class",
        "types",
    )


def _limb_entries(limb: Any, patient: Any) -> dict[str, float]:
    vector = np.asarray(limb.descriptor_vector, dtype=np.float64)
    return {
        "stenosis_ratio": float(limb.stenosis_ratio),
        "age_per_decade": (float(patient.age) - 58.0) / 10.0,
        "sex_female": 1.0 if patient.sex == "female" else 0.0,
        "type_iii_iv": 1.0 if limb.reference_type in (LesionType.POST_THROMBOTIC, LesionType.OCCLUSIVE) else 0.0,
        "resistance_ratio_civ": _descriptor_column(vector, "CIV", "resistance_ratio"),
        "shape_factor_civ": _descriptor_column(vector, "CIV", "shape_factor"),
        "minimum_luminal_area_civ": _descriptor_column(vector, "CIV", "minimum_luminal_area"),
        "composite_score": float(patient.composite_score),
        "ceap_class": float(patient.ceap_class),
        # Retained only so the outcome builder can threshold it; it is never a feature.
        "types": float(int(limb.reference_type)),
    }


def endpoint_labels(cohort: CohortRecord, rows: IntArray | None = None) -> dict[str, IntArray]:
    """Binary presentation endpoints, one row per limb."""
    limbs = list(cohort.limbs)
    lookup = {patient.patient_id: patient for patient in cohort.patients}
    selection = np.arange(len(limbs), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    composites = np.array([lookup[limb.patient_id].composite_score for limb in limbs], dtype=np.float64)
    median = float(np.median(composites[selection])) if selection.size else 0.0
    out: dict[str, list[int]] = {name: [] for name in PRESENTATION_ENDPOINTS}
    for index in selection:
        limb = limbs[int(index)]
        patient = lookup[limb.patient_id]
        out["dvt_history"].append(int(patient.dvt_history))
        out["ceap_c4_c6"].append(int(patient.ceap_class >= 4))
        out["venous_claudication"].append(int(patient.venous_claudication))
        out["symptom_laterality"].append(int(patient.symptom_laterality == "left"))
        out["composite_score_ge_median"].append(int(patient.composite_score >= median))
    return {name: np.asarray(values, dtype=np.int64) for name, values in out.items()}


def scaled_columns(columns: dict[str, FloatArray], rows: IntArray | None = None) -> dict[str, FloatArray]:
    """Standardise the continuous descriptor columns on the given rows."""
    selection = np.arange(len(next(iter(columns.values())))) if rows is None else np.asarray(rows, dtype=np.int64)
    out: dict[str, FloatArray] = {}
    for name, values in columns.items():
        out[name] = values
    for name in ("resistance_ratio_civ", "shape_factor_civ", "minimum_luminal_area_civ"):
        values = columns[name][selection]
        scale = float(np.nanstd(values))
        if scale < 1e-9:
            scale = 1.0
        out[name] = (columns[name] - float(np.nanmean(values))) / scale
    return out


def run_association(
    cohort: CohortRecord,
    rows: IntArray | None = None,
    threshold: float = 0.5,
    degree_columns: tuple[str, ...] = ("stenosis_ratio", "age_per_decade", "sex_female"),
    extended_columns: tuple[str, ...] = (
        "stenosis_ratio",
        "age_per_decade",
        "sex_female",
        "type_iii_iv",
        "resistance_ratio_civ",
        "shape_factor_civ",
    ),
) -> AssociationTable:
    """Fit the nested pair per endpoint and collect every reported quantity."""
    raw = build_columns(cohort, rows)
    columns = scaled_columns(raw, rows)
    labels = endpoint_labels(cohort, rows)
    degree = design_matrix(columns, degree_columns)
    extended = design_matrix(columns, extended_columns)
    result = AssociationTable(threshold=threshold)
    result.units = {
        "limbs": int(degree.shape[0]),
        "patients": int(np.unique(cohort.patient_index()[rows] if rows is not None else cohort.patient_index()).size),
        "added_parameters": int(len(extended_columns) - len(degree_columns)),
    }
    for endpoint, target in labels.items():
        if int(np.unique(target).size) < 2:
            result.notes.append(f"{endpoint}: only one class present, the comparison is undefined")
            continue
        base_log_likelihood = log_likelihood_of_design(degree, target)
        extended_log_likelihood = log_likelihood_of_design(extended, target)
        test = likelihood_ratio_test(
            extended_log_likelihood, base_log_likelihood, len(extended_columns) - len(degree_columns)
        )
        result.likelihood_ratio[endpoint] = test
        result.degree_only_auc[endpoint] = _fitted_auc(degree, target)
        result.extended_auc[endpoint] = _fitted_auc(extended, target)
        base_probability = _fitted_probability(degree, target)
        extended_probability = _fitted_probability(extended, target)
        result.net_renclassification[endpoint] = continuous_nri(extended_probability, base_probability, target)["nri"]
    result.odds_ratios = _odds_ratio_block(extended, labels, extended_columns)
    result.net_benefit = _net_benefit_block(extended, labels, threshold)
    result.notes.append(
        "the extended model has to add discrimination beyond the stenosis ratio for the type to carry presentation"
        " information"
    )
    return result


def _fitted_weights(design: FloatArray, labels: IntArray, ridge: float = 1e-6) -> FloatArray:
    features = np.asarray(design, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(100):
        linear = np.clip(features @ weights, -60.0, 60.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        gradient = features.T @ (targets - probability) - ridge * weights
        variance = probability * (1.0 - probability)
        hessian = features.T @ (features * variance[:, None]) + ridge * np.eye(features.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights = weights + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return weights


def _fitted_probability(design: FloatArray, labels: IntArray) -> FloatArray:
    weights = _fitted_weights(design, labels)
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(design) @ weights, -60.0, 60.0)))


def _fitted_auc(design: FloatArray, labels: IntArray) -> float:
    return auroc(_fitted_probability(design, labels), np.asarray(labels, dtype=np.int64))


def _odds_ratio_block(
    design: FloatArray, labels: dict[str, IntArray], columns: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Adjusted odds ratios of the descriptor columns for each endpoint."""
    out: dict[str, dict[str, float]] = {}
    for endpoint, target in labels.items():
        if int(np.unique(target).size) < 2:
            continue
        for name in ("type_iii_iv", "stenosis_ratio", "resistance_ratio_civ", "shape_factor_civ"):
            if name not in columns:
                continue
            column = list(columns).index(name) + 1
            out[f"{endpoint}:{name}"] = adjusted_odds_ratio(design, target, column)
    return out


def _net_benefit_block(design: FloatArray, labels: dict[str, IntArray], threshold: float) -> float:
    endpoint = "dvt_history"
    target = labels.get(endpoint)
    if target is None or int(np.unique(target).size) < 2:
        return float("nan")
    probability = _fitted_probability(design, target)
    calls = (probability >= threshold).astype(np.int64)
    return float(net_benefit_gain(calls, np.zeros_like(calls), target, threshold))


def association_by_patient_bootstrap(
    cohort: CohortRecord, resamples: int = 200, seed: int = 0, threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """Percentile intervals of the association quantities under patient resampling."""
    from conductance_ctv.metrics.bootstrap import clustered_resample_indices, patient_clusters

    clusters = patient_clusters(cohort.patient_index())
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(resamples):
        rows = clustered_resample_indices(clusters, rng)
        table = run_association(cohort, rows=rows, threshold=threshold)
        for endpoint, value in table.net_renclassification.items():
            collected.setdefault(f"nri:{endpoint}", []).append(float(value))
        for endpoint, entry in table.likelihood_ratio.items():
            collected.setdefault(f"lr_p:{endpoint}", []).append(float(entry["p_value"]))
    out: dict[str, dict[str, float]] = {}
    for name, values in collected.items():
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            continue
        out[name] = {
            "estimate": float(np.mean(array)),
            "lower": float(np.quantile(array, 0.025)),
            "upper": float(np.quantile(array, 0.975)),
        }
    return out
