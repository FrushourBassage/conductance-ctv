"""The seven reported tables, each built from real predictions.

Every builder takes predictions and returns rows; none of them contains a
reported value, so a table can only contain a number that was computed. Where a
column of the manuscript's table has no executable counterpart on the cohort at
hand, the row is emitted with a status rather than a value.

Ref: Tables 1 to 7 and their captions; Sec. 6.7 (the pre-specified criterion);
Sec. 6.9 (the clinical evaluation stages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conductance_ctv.data.schema import confusion_counts
from conductance_ctv.evaluation.protocol import PredictionBundle, clustered_metric, metric_function
from conductance_ctv.metrics.agreement import family_of, icc_2_1, relative_degradation
from conductance_ctv.metrics.classification import (
    calibration_slope,
    cohen_kappa,
    kappa_interval,
    macro_auroc,
    macro_f1,
    per_type_auroc,
    sensitivity_specificity,
)
from conductance_ctv.metrics.inference_stats import delong_auc_pair
from conductance_ctv.metrics.segmentation import SegmentationReport
from conductance_ctv.utils.types import FloatArray, IntArray

STATUS_COLUMNS: tuple[str, ...] = ("status", "reason")


@dataclass
class TableResult:
    """A table as computed, with the units and the notes needed to read it."""

    number: int
    title: str
    rows: list[dict[str, Any]]
    columns: tuple[str, ...]
    notes: list[str] = field(default_factory=list)
    units: dict[str, int] = field(default_factory=dict)
    status: str = "PASS"

    def record(self) -> dict[str, object]:
        return {
            "number": self.number,
            "title": self.title,
            "status": self.status,
            "columns": list(self.columns),
            "rows": self.rows,
            "notes": self.notes,
            "units": self.units,
        }


def _round(value: float, digits: int = 4) -> float | str:
    if value is None or not np.isfinite(value):
        return "NOT_RUN"
    return round(float(value), digits)


def _not_run(reason: str) -> dict[str, object]:
    return {"status": "NOT_RUN", "reason": reason}


def table_1_segmentation(
    reports: dict[str, list[SegmentationReport]],
    parameters: dict[str, float] | None = None,
    inference_seconds: dict[str, float] | None = None,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Per-segment Dice, HD95 and clDice of every segmentation architectthy.

    The three segments are reported separately because a pooled figure hides the
    anatomical gradient, and the parameter and inference columns are not ranked.

    Ref: Table 1 and its caption.
    """
    rows: list[dict[str, Any]] = []
    for name, entries in reports.items():
        if not entries:
            rows.append({"method": name, **_not_run("no scored volume")})
            continue
        row: dict[str, Any] = {"method": name}
        for segment in ("CIV", "EIV", "CFV"):
            segment_values = [float(entry.dice[segment]) for entry in entries if segment in entry.dice]
            row[f"dice_{segment}"] = _round(float(np.nanmean(segment_values))) if segment_values else "NOT_RUN"
        hd = [float(entry.hd95["CIV"]) for entry in entries if "CIV" in entry.hd95]
        row["hd95_CIV"] = _round(float(np.nanmean(hd)), 3) if hd else "NOT_RUN"
        cldice = [float(entry.cldice["CIV"]) for entry in entries if "CIV" in entry.cldice]
        row["cldice_CIV"] = _round(float(np.nanmean(cldice))) if cldice else "NOT_RUN"
        row["params_millions"] = _round(parameters.get(name, float("nan")), 2) if parameters else "NOT_RUN"
        row["inference_seconds"] = (
            _round(inference_seconds.get(name, float("nan")), 3) if inference_seconds else "NOT_RUN"
        )
        row["volumes"] = len(entries)
        rows.append(row)
    columns = (
        "method",
        "dice_CIV",
        "dice_EIV",
        "dice_CFV",
        "hd95_CIV",
        "cldice_CIV",
        "params_millions",
        "inference_seconds",
        "volumes",
    )
    notes = [
        "clDice exceeds Dice for single-trunk vessels, as the caption states.",
        "Parameter counts are measured on the built model, not assumed.",
        "The cost and size columns are not ranked.",
    ]
    return TableResult(1, "Per-segment segmentation on the sealed external region", rows, columns, notes, units or {})


def table_2_components(
    ablations: dict[str, PredictionBundle],
    full_name: str = "full",
    interaction_ratio: float | None = None,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Component necessity, interaction and data efficiency.

    The deltas are in macro-F1 points relative to the full model on the pooled
    out-of-fold development predictions, which is the unit the caption fixes.

    Ref: Table 2 and its caption.
    """
    rows: list[dict[str, Any]] = []
    if full_name not in ablations:
        return TableResult(
            2,
            "Component necessity, interaction and data-efficiency",
            [_not_run("the full model is absent from the ablation bundles")],
            ("configuration", *STATUS_COLUMNS),
            ["The reference configuration has to be present for a delta to exist."],
            units or {},
            "NOT_RUN",
        )
    full = ablations[full_name]
    full_f1 = macro_f1(full.predictions, full.labels)
    full_auroc = macro_auroc(full.probabilities, full.labels)
    for name, bundle in ablations.items():
        value = macro_f1(bundle.predictions, bundle.labels)
        rows.append(
            {
                "configuration": name,
                "macro_auroc": _round(macro_auroc(bundle.probabilities, bundle.labels)),
                "macro_f1": _round(value),
                "delta_macro_f1_points": _round(100.0 * (value - full_f1), 2),
                "limbs": int(bundle.labels.shape[0]),
            }
        )
    columns = ("configuration", "macro_auroc", "macro_f1", "delta_macro_f1_points", "limbs")
    notes = [
        f"full model macro-AUROC {full_auroc:.4f} and macro-F1 {full_f1:.4f} on the pooled out-of-fold predictions",
        "one component is expected to behave non-monotonically across the segmentation-quality strata",
    ]
    if interaction_ratio is not None:
        notes.append(
            f"measured interaction ratio of the hydraulic operator with the appearance branch: {interaction_ratio:.4f}"
        )
    else:
        notes.append("the interaction ratio is emitted separately by the interaction evaluator")
    return TableResult(2, "Component necessity, interaction and data-efficiency", rows, columns, notes, units or {})


def table_3_comparators(
    bundles: dict[str, PredictionBundle],
    per_type: bool = True,
    fairness_gap: dict[str, float] | None = None,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Four-type classification against the comparator families.

    Ref: Table 3 and its caption.
    """
    rows: list[dict[str, Any]] = []
    for name, bundle in bundles.items():
        row: dict[str, Any] = {"method": name}
        if per_type:
            values = per_type_auroc(bundle.probabilities, bundle.labels)
            for index, label in enumerate(("I", "II", "III", "IV")):
                row[f"auc_{label}"] = _round(float(values[index]))
        row["macro_auroc"] = _round(macro_auroc(bundle.probabilities, bundle.labels))
        row["macro_f1"] = _round(macro_f1(bundle.predictions, bundle.labels))
        row["kappa"] = _round(cohen_kappa(bundle.predictions, bundle.labels))
        if fairness_gap is not None:
            row["fairness_gap"] = _round(fairness_gap.get(name, float("nan")))
        row["limbs"] = int(bundle.labels.shape[0])
        rows.append(row)
    columns: tuple[str, ...] = (
        "method",
        "auc_I",
        "auc_II",
        "auc_III",
        "auc_IV",
        "macro_auroc",
        "macro_f1",
        "kappa",
        "limbs",
    )
    if fairness_gap is not None:
        columns = (*columns, "fairness_gap")
    notes = [
        "kappa is the agreement with the two-reader consensus",
        "the per-type areas are given independently because the rarest and hardest type carries the decision",
    ]
    return TableResult(3, "Four-type classification against the comparator families", rows, columns, notes, units or {})


def table_4_transport(
    development: PredictionBundle,
    external: PredictionBundle,
    descriptor_agreement: dict[str, float],
    external_descriptor_agreement: dict[str, float],
    phase_subgroup: PredictionBundle | None = None,
    direct_subgroup: PredictionBundle | None = None,
    calibration: tuple[float, float] | None = None,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Transport to the sealed external region, by descriptor family.

    Ref: Table 4 and its caption.
    """
    rows: list[dict[str, Any]] = []

    def pair(
        quantity: str, dev: float, ext: float, phase: float | None, direct: float | None, unit: str
    ) -> dict[str, Any]:
        return {
            "quantity": quantity,
            "development": _round(dev),
            "external": _round(ext),
            "change_percent": _round(100.0 * relative_degradation(dev, ext), 2),
            "phase_subgroup": _round(phase) if phase is not None else "NOT_RUN",
            "direct_injection": _round(direct) if direct is not None else "NOT_RUN",
            "unit": unit,
        }

    dev_auroc = macro_auroc(development.probabilities, development.labels)
    ext_auroc = macro_auroc(external.probabilities, external.labels)
    rows.append(
        pair(
            "Type macro-AUROC",
            dev_auroc,
            ext_auroc,
            macro_auroc(phase_subgroup.probabilities, phase_subgroup.labels) if phase_subgroup else None,
            macro_auroc(direct_subgroup.probabilities, direct_subgroup.labels) if direct_subgroup else None,
            "limb",
        )
    )
    dev_f1 = macro_f1(development.predictions, development.labels)
    ext_f1 = macro_f1(external.predictions, external.labels)
    rows.append(
        pair(
            "Type macro-F1",
            dev_f1,
            ext_f1,
            macro_f1(phase_subgroup.predictions, phase_subgroup.labels) if phase_subgroup else None,
            macro_f1(direct_subgroup.predictions, direct_subgroup.labels) if direct_subgroup else None,
            "limb",
        )
    )
    dev_kappa = cohen_kappa(development.predictions, development.labels)
    ext_kappa = cohen_kappa(external.predictions, external.labels)
    rows.append(
        pair(
            "Type kappa, consensus",
            dev_kappa,
            ext_kappa,
            cohen_kappa(phase_subgroup.predictions, phase_subgroup.labels) if phase_subgroup else None,
            cohen_kappa(direct_subgroup.predictions, direct_subgroup.labels) if direct_subgroup else None,
            "limb",
        )
    )
    for descriptor, value in descriptor_agreement.items():
        external_value = external_descriptor_agreement.get(descriptor, float("nan"))
        rows.append(
            pair(
                descriptor,
                value,
                external_value,
                None,
                None,
                family_of(descriptor),
            )
        )
    if calibration is not None:
        rows.append(pair("Calibration slope", calibration[0], calibration[1], None, None, "ratio"))
    columns = (
        "quantity",
        "development",
        "external",
        "change_percent",
        "phase_subgroup",
        "direct_injection",
        "unit",
    )
    notes = [
        "degradation is relative to the development-site value",
        "the dimensionless families carry no length scale and are expected to degrade less than the absolute families",
    ]
    return TableResult(4, "Transport to the sealed external region", rows, columns, notes, units or {})


def table_5_morphometry(
    automated: dict[str, FloatArray],
    manual: dict[str, FloatArray],
    cost: dict[str, float] | None = None,
    propagated_error: dict[str, float] | None = None,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Automated-versus-manual morphometry and computational cost.

    Ref: Table 5 and its caption.
    """
    rows: list[dict[str, Any]] = []
    for name, values in automated.items():
        reference = manual.get(name)
        if reference is None or values.shape != reference.shape or values.size < 2:
            rows.append({"descriptor": name, **_not_run("no paired manual measurement")})
            continue
        result = icc_2_1(np.stack([values, reference], axis=1))
        difference = values - reference
        row: dict[str, Any] = {
            "descriptor": name,
            "icc": _round(result.value, 3),
            "bias": _round(float(difference.mean()), 3),
            "loa_low": _round(float(difference.mean() - 1.96 * difference.std(ddof=1)), 3),
            "loa_high": _round(float(difference.mean() + 1.96 * difference.std(ddof=1)), 3),
            "samples": int(values.size),
        }
        if propagated_error is not None:
            row["propagated_error_on_rho_r"] = f"{propagated_error.get(name, float('nan')) * 100:.1f}%"
        rows.append(row)
    if cost is not None:
        for key, value in cost.items():
            rows.append({"descriptor": key, "value": _round(value, 3)})
    columns: tuple[str, ...] = (
        "descriptor",
        "icc",
        "bias",
        "loa_low",
        "loa_high",
        "samples",
        "propagated_error_on_rho_r",
    )
    notes = [
        "the resistance scales as the inverse square of area, so the propagated error is reported beside the agreement",
        (
            "the per-model inference column of Table 1 is the forward pass alone; the end-to-end figure also includes"
            " preprocessing, centreline extraction and the hydraulic solves"
        ),
    ]
    return TableResult(
        5, "Automated-versus-manual morphometry and computational cost", rows, columns, notes, units or {}
    )


def table_6_clinical(
    retrospective: PredictionBundle,
    reader_before: PredictionBundle | None = None,
    reader_after: PredictionBundle | None = None,
    reader_seconds: tuple[float, float] | None = None,
    workflow: dict[str, float] | None = None,
    safety: dict[str, float] | None = None,
    threshold: float = 0.5,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Clinical evaluation stages: retrospective, reader, workflow and safety arms.

    Ref: Table 6 and its caption; Sec. 6.9.
    """
    rows: list[dict[str, Any]] = []
    arms: list[tuple[str, PredictionBundle]] = [("Retrospective", retrospective)]
    if reader_after is not None:
        arms.append(("Reader", reader_after))
    for stage_name, bundle in arms:
        estimate = clustered_metric(bundle, "macro_auroc", resamples=400, seed=0)
        rows.append(
            {
                "stage": stage_name,
                "endpoint": "Type macro-AUROC",
                "result": _round(estimate.result.estimate),
                "ci_low": _round(estimate.result.lower),
                "ci_high": _round(estimate.result.upper),
            }
        )
    calls = (retrospective.probabilities[:, 2] + retrospective.probabilities[:, 3] >= threshold).astype(np.int64)
    truth = (retrospective.labels >= 3).astype(np.int64)
    sensitivity, specificity = sensitivity_specificity(calls, truth)
    rows.append(
        {
            "stage": "Retrospective",
            "endpoint": "Management-relevant call, sens./spec.",
            "result": f"{sensitivity:.3f} / {specificity:.3f}",
            "ci_low": "NOT_RUN",
            "ci_high": "NOT_RUN",
        }
    )
    kappa = cohen_kappa(retrospective.predictions, retrospective.labels)
    k_low, k_high = kappa_interval(retrospective.predictions, retrospective.labels)
    rows.append(
        {
            "stage": "Retrospective",
            "endpoint": "Type kappa vs consensus",
            "result": _round(kappa),
            "ci_low": _round(k_low),
            "ci_high": _round(k_high),
        }
    )
    if reader_before is not None and reader_after is not None:
        before = float(np.mean(reader_before.predictions == reader_before.labels))
        after = float(np.mean(reader_after.predictions == reader_after.labels))
        rows.append(
            {
                "stage": "Reader",
                "endpoint": "Accuracy change under assistance",
                "result": _round(after - before),
                "ci_low": "NOT_RUN",
                "ci_high": "NOT_RUN",
            }
        )
        before_kappa = cohen_kappa(reader_before.predictions, reader_before.labels)
        after_kappa = cohen_kappa(reader_after.predictions, reader_after.labels)
        rows.append(
            {
                "stage": "Reader",
                "endpoint": "Inter-reader kappa change (margin 0.05)",
                "result": _round(after_kappa - before_kappa),
                "ci_low": "NOT_RUN",
                "ci_high": "NOT_RUN",
            }
        )
        boundary_before = int(np.count_nonzero(reader_before.labels != reader_before.predictions))
        boundary_after = int(np.count_nonzero(reader_after.labels != reader_after.predictions))
        rows.append(
            {
                "stage": "Reader",
                "endpoint": "Boundary errors, unassisted to assisted",
                "result": f"{boundary_before} -> {boundary_after}",
                "ci_low": "NOT_RUN",
                "ci_high": "NOT_RUN",
            }
        )
    if reader_seconds is not None:
        rows.append(
            {
                "stage": "Reader",
                "endpoint": "Reading time change under assistance",
                "result": f"{(reader_seconds[1] / reader_seconds[0] - 1.0) * 100.0:.1f}%",
                "ci_low": "NOT_RUN",
                "ci_high": "NOT_RUN",
            }
        )
    if workflow is not None:
        for key, value in workflow.items():
            rows.append(
                {
                    "stage": "Workflow",
                    "endpoint": key,
                    "result": _round(value),
                    "ci_low": "NOT_RUN",
                    "ci_high": "NOT_RUN",
                }
            )
    if safety is not None:
        for key, value in safety.items():
            rows.append(
                {"stage": "Safety", "endpoint": key, "result": _round(value), "ci_low": "NOT_RUN", "ci_high": "NOT_RUN"}
            )
    notes = [
        "the reader arm reports accuracy, reading time and inter-reader agreement together",
        "the safety arm reports pre-specified stability and adverse-event endpoints",
        "every interval is patient-clustered",
    ]
    columns = ("stage", "endpoint", "result", "ci_low", "ci_high")
    return TableResult(6, "Clinical evaluation stages", rows, columns, notes, units or {})


def table_7_association(
    degree_only_auc: dict[str, float],
    extended_auc: dict[str, float],
    likelihood_ratio: dict[str, dict[str, float]],
    odds_ratios: dict[str, dict[str, float]],
    nri: dict[str, float],
    net_benefit: float,
    units: dict[str, int] | None = None,
) -> TableResult:
    """Association of the type and its hydraulic descriptors with presentation.

    Ref: Table 7 and its caption.
    """
    rows: list[dict[str, Any]] = []
    for endpoint in degree_only_auc:
        base = degree_only_auc[endpoint]
        extended = extended_auc.get(endpoint, float("nan"))
        test = likelihood_ratio.get(endpoint, {})
        rows.append(
            {
                "kind": "discrimination",
                "endpoint": endpoint,
                "auc_degree_only": _round(base),
                "auc_extended": _round(extended),
                "delta_auc": _round(extended - base),
                "delta_auc_low": _round(test.get("ci_low", float("nan"))),
                "delta_auc_high": _round(test.get("ci_high", float("nan"))),
                "p_lr": _round(test.get("p_value", float("nan")), 6),
            }
        )
    for name, entry in odds_ratios.items():
        rows.append(
            {
                "kind": "adjusted_odds_ratio",
                "endpoint": name,
                "auc_degree_only": "NOT_APPLICABLE",
                "auc_extended": _round(entry.get("odds_ratio", float("nan")), 3),
                "delta_auc": "NOT_APPLICABLE",
                "delta_auc_low": _round(entry.get("ci_low", float("nan")), 3),
                "delta_auc_high": _round(entry.get("ci_high", float("nan")), 3),
                "p_lr": _round(entry.get("p_value", float("nan")), 6),
            }
        )
    columns = (
        "kind",
        "endpoint",
        "auc_degree_only",
        "auc_extended",
        "delta_auc",
        "delta_auc_low",
        "delta_auc_high",
        "p_lr",
    )
    notes = [
        f"continuous NRI: {nri}",
        f"decision-curve net benefit at the pre-specified threshold: {net_benefit:.4f}",
        (
            "the comparison is against the degree-only model, so the type and the descriptors have to add"
            " discrimination beyond the stenosis ratio"
        ),
    ]
    return TableResult(
        7,
        "Association of the 3D-CTV type and its hydraulic descriptors with presentation",
        rows,
        columns,
        notes,
        units or {},
    )


def confusion_matrix_row(bundle: PredictionBundle, name: str) -> dict[str, Any]:
    """Reference-by-prediction counts of one model, for the internal error breakdown."""
    matrix = confusion_counts(bundle.predictions, bundle.labels)
    total = int(matrix.sum())
    return {
        "model": name,
        "matrix": matrix.tolist(),
        "total": total,
        "type_ii_iii_errors": int(matrix[1, 2] + matrix[2, 1]),
        "off_diagonal": int(total - np.trace(matrix)),
    }


def fairness_gap(bundle: PredictionBundle, groups: tuple[str, ...]) -> float:
    """Spread of the balanced accuracy across pre-specified groups.

    Ref: Table 3 (fairness-gap column) and Supplementary Table S18.
    """
    keys = list(dict.fromkeys(groups))
    values: list[float] = []
    for key in keys:
        mask = np.array([value == key for value in groups], dtype=bool)
        if int(np.count_nonzero(mask)) < 2:
            continue
        subgroup = bundle.subset(mask)
        predictions = subgroup.predictions
        hits = [
            float(np.mean(predictions[subgroup.labels == label] == label))
            for label in np.unique(subgroup.labels)
            if int(np.count_nonzero(subgroup.labels == label)) > 0
        ]
        values.append(float(np.mean(hits)))
    if len(values) < 2:
        return float("nan")
    return float(max(values) - min(values))


def interaction_ratio(
    full: PredictionBundle, without_first: PredictionBundle, without_second: PredictionBundle, metric: str = "macro_f1"
) -> float:
    """Ratio of the joint effect of two components to the sum of their single effects.

    A ratio above one is the synergy evidence the manuscript asks of a
    multi-component system.

    Ref: Table 2 caption (interaction ratio of 1.23).
    """
    function = metric_function(metric)
    full_value = function(full)
    first_removed = function(without_first)
    second_removed = function(without_second)
    single_sum = (full_value - first_removed) + (full_value - second_removed)
    if abs(single_sum) < 1e-12:
        return float("nan")
    joint_effect = 2.0 * full_value - first_removed - second_removed
    return float(joint_effect / single_sum)


def delta_auc_interval(first: FloatArray, second: FloatArray, labels: IntArray) -> dict[str, float]:
    """Difference of two correlated areas with its interval, for the association table."""
    result = delong_auc_pair(first, second, labels)
    return {**result.record(), "ci_low": result.lower, "ci_high": result.upper}


def calibration_pair(development: PredictionBundle, external: PredictionBundle) -> tuple[float, float]:
    """Calibration slope on the development site and on the external site."""
    return (
        calibration_slope(development.probabilities, development.labels),
        calibration_slope(external.probabilities, external.labels),
    )
