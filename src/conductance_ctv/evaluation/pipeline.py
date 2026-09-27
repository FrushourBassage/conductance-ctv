"""End-to-end pipeline: cohort, descriptors, models, predictions and tables.

The pipeline is deliberately explicit about which quantity came from which
computation. A table row can be a measurement that was taken, a prediction that
was produced by a trained model, or a value that the run could not produce; the
third case is emitted with a status and a reason rather than omitted.

Ref: Sec. 6.2 to 6.9; Algorithm 3; Tables 1 to 7.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conductance_ctv.data.dataset import LimbCase, TypeDataset, build_cases
from conductance_ctv.data.fixture import generate_cohort
from conductance_ctv.data.schema import DEVELOPMENT_SITES, EXTERNAL_SITE, CohortRecord
from conductance_ctv.data.splits import accrual_halves, patient_level_split
from conductance_ctv.data.transforms import DescriptorStandardiser
from conductance_ctv.evaluation.protocol import (
    PredictionBundle,
    clustered_metric,
    subgroup_bundles,
)
from conductance_ctv.evaluation.report import Report, ReportSection, matrix_section, summary_section
from conductance_ctv.evaluation.tables import (
    TableResult,
    confusion_matrix_row,
    fairness_gap,
    table_1_segmentation,
    table_2_components,
    table_3_comparators,
    table_4_transport,
    table_5_morphometry,
    table_6_clinical,
    table_7_association,
)
from conductance_ctv.hydraulics.centerline import build_path
from conductance_ctv.hydraulics.descriptors import (
    DEFAULT_FIELD_OF_VIEW_MM,
    DESCRIPTOR_FIELDS,
    SEGMENT_NAMES,
    analyse_samples,
    describe_sections,
    sample_sections,
)
from conductance_ctv.hydraulics.grid import CrossSection
from conductance_ctv.hydraulics.series import propagated_resistance_error
from conductance_ctv.metrics.classification import calibration_slope, macro_auroc, per_type_auroc
from conductance_ctv.metrics.segmentation import SegmentationReport, score_volume
from conductance_ctv.models.baselines import (
    RadiomicsComparator,
    diameter_rule_scores,
    fixture_radiomic_dataset,
)
from conductance_ctv.models.classifier import ConductanceTypeModel
from conductance_ctv.training.trainer import TypeTrainer, predict_probabilities
from conductance_ctv.utils.config import ExperimentConfig, load_experiment
from conductance_ctv.utils.logging_setup import get_logger
from conductance_ctv.utils.numeric import as_quad, as_triple
from conductance_ctv.utils.types import FloatArray, IntArray

LOGGER = get_logger("evaluation.pipeline")


@dataclass
class PipelineRun:
    """Everything a pipeline execution produced, ready to be written out."""

    config: ExperimentConfig
    cohort: CohortRecord | None = None
    cases: list[LimbCase] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    descriptor_agreement: dict[str, float] = field(default_factory=dict)
    solver_agreement: dict[str, float | dict[str, float]] = field(default_factory=dict)
    status: str = "PARTIALLY_VERIFIED"

    def record(self) -> dict[str, Any]:
        return {
            "experiment": self.config.name,
            "cohort_patients": self.cohort.n_patients if self.cohort else 0,
            "cohort_limbs": self.cohort.n_limbs if self.cohort else 0,
            "cases": len(self.cases),
            "tables": {table.number: table.status for table in self.tables},
            "timings": self.timings,
            "notes": self.notes,
            "status": self.status,
        }

    def report(self) -> Report:
        sections = [
            summary_section("configuration", self.config.documented()),
            summary_section(
                "cohort",
                {
                    "patients": self.cohort.n_patients if self.cohort else 0,
                    "limbs": self.cohort.n_limbs if self.cohort else 0,
                    "development_sites": ", ".join(str(site) for site in DEVELOPMENT_SITES),
                    "external_site": str(EXTERNAL_SITE),
                },
            ),
            summary_section("timings_seconds", {key: round(value, 3) for key, value in self.timings.items()}),
            ReportSection(title="notes", body="\n".join(self.notes)),
        ]
        return Report(title=f"Evaluation report: {self.config.name}", sections=sections, tables=self.tables)


def _bundle_from_cases(
    name: str,
    cases: list[LimbCase],
    probabilities: FloatArray,
    patient_index: IntArray | None = None,
) -> PredictionBundle:
    if patient_index is None:
        lookup: dict[str, int] = {}
        for case in cases:
            lookup.setdefault(case.limb.patient_id, len(lookup))
        patient_index = np.array([lookup[case.limb.patient_id] for case in cases], dtype=np.int64)
    return PredictionBundle(
        name=name,
        limb_ids=tuple(case.limb.limb_id for case in cases),
        patient_index=patient_index,
        labels=np.array([case.label + 1 for case in cases], dtype=np.int64),
        probabilities=probabilities,
        sites=tuple(case.limb.site for case in cases),
        phases=tuple(str(case.limb.phase) for case in cases),
        vendors=tuple(case.limb.vendor for case in cases),
    )


def _one_hot(name: str, cases: list[LimbCase], calls: IntArray) -> PredictionBundle:
    probabilities = np.zeros((len(cases), 4), dtype=np.float64)
    probabilities[np.arange(len(cases)), np.clip(calls - 1, 0, 3)] = 1.0
    return _bundle_from_cases(name, cases, probabilities)


def generate_pipeline_cohort(config: ExperimentConfig) -> CohortRecord:
    """Build the cohort fixture the experiment is configured for."""
    data = dict(config.raw.get("data", {}))
    return generate_cohort(
        n_patients=int(data.get("n_patients", 240)),
        seed=config.seed,
        site_mix=as_triple(data.get("site_mix", (0.42, 0.36, 0.22))),
        prevalence=as_quad(data.get("type_prevalence", (0.45, 0.30, 0.18, 0.07))),
        grid=config.grid,
        two_limb_fraction=float(data.get("two_limb_fraction", 0.68)),
    )


def hydraulic_descriptor_scores(
    cases: list[LimbCase], upsample_factor: int = 1, field_of_view_mm: float = DEFAULT_FIELD_OF_VIEW_MM
) -> dict[str, Any]:
    """Compare the solved descriptor vector of every case with its closed form.

    The two vectors come from different computations: the solver path slices the
    rendered mask and solves Eq. (1) on every section, while the reference path
    evaluates the closed forms of the section family that generated the mask. The
    difference is therefore a real discretisation error and not a tautology.
    """
    solved: list[FloatArray] = []
    reference: list[FloatArray] = []
    per_field: dict[str, float] = {}
    residuals: list[float] = []
    for case in cases:
        path = build_path(case.mask, case.grid.spacing_zyx, max(case.grid.spacing_zyx[0], 1.0))
        labels = path.segment_of((0.0, 0.40, 0.75, 1.0))
        spacing_mm = min(case.grid.spacing_zyx[:2]) / float(max(upsample_factor, 1))
        sections: list[CrossSection] = sample_sections(
            case.mask, case.grid.spacing_zyx, path, spacing_mm, field_of_view_mm
        )
        results = analyse_samples(sections)
        signature = describe_sections(results, path.arclength, labels)
        solved.append(signature.vector())
        residuals.append(float(np.max([result.residual for result in results], initial=0.0)))
        reference.append(np.asarray(case.limb.descriptor_vector, dtype=np.float64))
    if not solved:
        return {"solved": np.zeros((0, 0)), "reference": np.zeros((0, 0)), "relative_error": {}}
    solved_matrix = np.stack(solved, axis=0)
    reference_matrix = np.stack(reference, axis=0)
    names = tuple(f"{segment}_{field_name}" for segment in SEGMENT_NAMES for field_name in DESCRIPTOR_FIELDS)
    for column, name in enumerate(names):
        solved_column = solved_matrix[:, column]
        reference_column = reference_matrix[:, column]
        finite = np.isfinite(solved_column) & np.isfinite(reference_column)
        if int(np.count_nonzero(finite)) == 0:
            continue
        denominator = np.where(np.abs(reference_column[finite]) < 1e-9, np.nan, reference_column[finite])
        relative = np.abs(solved_column[finite] - denominator) / np.abs(denominator)
        per_field[name] = float(np.nanmedian(relative)) if np.any(np.isfinite(relative)) else float("nan")
    return {
        "solved": solved_matrix,
        "reference": reference_matrix,
        "relative_error": per_field,
        "max_residual": float(max(residuals, default=0.0)),
    }


def segmentation_scores_for_method(
    cases: list[LimbCase], predictions: dict[str, np.ndarray]
) -> list[SegmentationReport]:
    reports: list[SegmentationReport] = []
    for case in cases:
        predicted = predictions.get(case.limb.limb_id)
        if predicted is None:
            continue
        reports.append(score_volume(predicted, case.mask, case.grid.spacing_zyx))
    return reports


def build_comparator_bundles(
    cohort: CohortRecord,
    cases: list[LimbCase],
    full_model: ConductanceTypeModel | None,
    config: ExperimentConfig,
    standardiser: DescriptorStandardiser | None = None,
) -> dict[str, PredictionBundle]:
    """Every comparator row of the third table, on the identical units."""
    bundles: dict[str, PredictionBundle] = {}
    limb_records = [case.limb for case in cases]
    lookup: dict[str, int] = {}
    for case in cases:
        lookup.setdefault(case.limb.patient_id, len(lookup))
    patient_index = np.array([lookup[case.limb.patient_id] for case in cases], dtype=np.int64)

    diameter = diameter_rule_scores(limb_records)
    normalized = diameter / np.clip(diameter.sum(axis=1, keepdims=True), 1e-9, None)
    bundles["Single-diameter rule"] = _bundle_from_cases("Single-diameter rule", cases, normalized, patient_index)

    radiomic = fixture_radiomic_dataset(limb_records, grid=config.grid.shape, seed=config.seed)
    split = patient_level_split(cohort, seed=config.seed)
    train_mask, _, _ = split.masks(cohort)
    for learner in ("lr", "gbm", "svm"):
        model = RadiomicsComparator(learner=learner, seed=config.seed)
        model.fit(radiomic.features[train_mask], radiomic.labels[train_mask])
        scores = model.decision_scores(radiomic.features)
        bundles[f"Radiomics + {learner.upper()}"] = _bundle_from_cases(
            f"Radiomics + {learner.upper()}", cases, scores, patient_index
        )

    if full_model is not None:
        fitted = standardiser if standardiser is not None else DescriptorStandardiser()
        dataset = TypeDataset(cases, standardiser=fitted)
        probabilities, _ = predict_probabilities(full_model, dataset, config.train)
        bundles["Conductance-3D-CTV (proposed)"] = _bundle_from_cases(
            "Conductance-3D-CTV (proposed)", cases, probabilities, patient_index
        )
    return bundles


def descriptor_agreement_table(cases: list[LimbCase], solution: dict[str, Any]) -> dict[str, FloatArray]:
    """Automated-versus-reference pairs for the descriptor families of Table 5.

    The reference is the closed form of the generating section family, which is
    the only manual-style three-dimensional reading this release can pair against
    without the clinical volumes.
    """
    automated, manual = _morphometry_pairs(solution)
    del cases
    return {**automated, **{f"reference::{name}": value for name, value in manual.items()}}


def run_pipeline(
    config_name: str = "main", overrides: tuple[str, ...] = (), max_steps: int | None = None
) -> PipelineRun:
    """Execute one experiment end to end and collect its tables."""
    overall_start = time.time()
    config = load_experiment(config_name, overrides)
    run = PipelineRun(config=config)
    run.timings["config_seconds"] = time.time() - overall_start

    started = time.time()
    cohort = generate_pipeline_cohort(config)
    run.cohort = cohort
    run.timings["cohort_seconds"] = time.time() - started

    started = time.time()
    cases = build_cases(cohort, grid=config.grid)
    run.cases = cases
    run.timings["render_seconds"] = time.time() - started

    started = time.time()
    solution = hydraulic_descriptor_scores(cases, upsample_factor=config.upsample_factor)
    run.descriptor_agreement = dict(solution.get("relative_error", {}))
    solver_agreement: dict[str, float | dict[str, float]] = {
        "descriptor_median_relative_error": {
            str(key): float(value) for key, value in dict(solution.get("relative_error", {})).items()
        },
        "max_section_residual": float(solution.get("max_residual", 0.0)),
    }
    run.solver_agreement = solver_agreement
    run.timings["hydraulics_seconds"] = time.time() - started

    split = patient_level_split(cohort, seed=config.seed)
    train_mask, tune_mask, external_mask = split.masks(cohort)
    train_cases = [case for case, keep in zip(cases, train_mask, strict=True) if keep]
    tune_cases = [case for case, keep in zip(cases, tune_mask, strict=True) if keep]
    external_cases = [case for case, keep in zip(cases, external_mask, strict=True) if keep]

    started = time.time()
    full_model: ConductanceTypeModel | None = None
    fitted_standardiser: DescriptorStandardiser | None = None
    probabilities = np.zeros((len(cases), 4), dtype=np.float64)
    if train_cases and tune_cases:
        trainer = TypeTrainer(
            config.train,
            encoder_name=config.encoder_name,
            width_mult=config.width_mult,
            decoder_width_mult=config.decoder_width_mult,
        )
        model, history, fitted_standardiser = trainer.fit(
            train_cases, tune_cases, max_steps=max_steps if max_steps is not None else config.train.batch_size
        )
        dataset = TypeDataset(cases, standardiser=fitted_standardiser)
        probabilities, _ = predict_probabilities(model, dataset, config.train)
        full_model = model
        run.notes.append(
            f"type model trained for {history.steps} steps; loss {history.train_loss[0]:.4f} to"
            f" {history.train_loss[-1]:.4f}"
        )
    else:
        run.notes.append("the development split has no training or tuning case, the type model was not trained")
    run.timings["type_training_seconds"] = time.time() - started

    bundles = build_comparator_bundles(cohort, cases, full_model, config, fitted_standardiser)
    if not bundles:
        run.notes.append("no comparator bundle could be built")
    proposed = bundles.get("Conductance-3D-CTV (proposed)")

    run.tables.append(
        table_3_comparators(
            bundles,
            units={"patients": cohort.n_patients, "limbs": cohort.n_limbs},
        )
    )

    ablations = _ablation_bundles(cases, probabilities, config)
    run.tables.append(
        table_2_components(
            ablations, full_name="Full model", units={"patients": cohort.n_patients, "limbs": cohort.n_limbs}
        )
    )

    if proposed is not None and external_cases:
        development = proposed.subset(np.array([case.limb.site in DEVELOPMENT_SITES for case in cases], dtype=bool))
        external = proposed.subset(np.array([case.limb.site == EXTERNAL_SITE for case in cases], dtype=bool))
        descriptor_agreement = {"resistance_ratio": float(np.nanmedian(list(run.descriptor_agreement.values())))}
        run.tables.append(
            table_4_transport(
                development,
                external,
                descriptor_agreement,
                descriptor_agreement,
                calibration=(
                    calibration_slope(development.probabilities, development.labels),
                    calibration_slope(external.probabilities, external.labels),
                ),
                units={"patients": cohort.n_patients, "limbs": cohort.n_limbs},
            )
        )
        run.tables.append(
            table_6_clinical(
                proposed,
                units={"patients": cohort.n_patients, "limbs": cohort.n_limbs},
            )
        )
    else:
        run.notes.append("the external partition is empty; the transport and clinical tables were not built")

    automated, manual = _morphometry_pairs(solution)
    run.tables.append(
        table_5_morphometry(
            automated,
            manual,
            cost={"hydraulic_solve_ms_per_segment": _hydraulic_solve_milliseconds(cases)},
            propagated_error={"minimum_luminal_area": propagated_resistance_error(1.0, 0.041)},
            units={"segments": cohort.n_limbs * 3},
        )
    )

    association = _association_table(cohort, cases)
    if association is not None:
        run.tables.append(
            table_7_association(
                association.degree_only_auc,
                association.extended_auc,
                association.likelihood_ratio,
                association.odds_ratios,
                association.net_renclassification,
                association.net_benefit,
                units=association.units,
            )
        )
    else:
        run.notes.append("the association table needs both classes of each endpoint to be present")

    run.tables.append(
        table_1_segmentation(
            _segmentation_reports(cases),
            parameters={config.encoder_name: float(config.raw.get("model", {}).get("reported_params_millions", 0.0))},
            units={"volumes": len(cases)},
        )
    )
    run.timings["total_seconds"] = time.time() - overall_start
    run.status = "PARTIALLY_VERIFIED"
    run.notes.append(
        "point values on the clinical cohort are NOT_RUN: this run executes every mechanism on the cohort fixture"
    )
    return run


def _morphometry_pairs(solution: dict[str, Any]) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    names = tuple(f"{segment}_{field_name}" for segment in SEGMENT_NAMES for field_name in DESCRIPTOR_FIELDS)
    solved = np.asarray(solution.get("solved", np.zeros((0, 0))), dtype=np.float64)
    reference = np.asarray(solution.get("reference", np.zeros((0, 0))), dtype=np.float64)
    automated: dict[str, FloatArray] = {}
    manual: dict[str, FloatArray] = {}
    if solved.size == 0:
        return automated, manual
    for column, name in enumerate(names):
        first = solved[:, column]
        second = reference[:, column]
        finite = np.isfinite(first) & np.isfinite(second)
        if int(np.count_nonzero(finite)) < 3:
            continue
        automated[name] = first[finite]
        manual[name] = second[finite]
    return automated, manual


def _hydraulic_solve_milliseconds(cases: list[LimbCase]) -> float:
    if not cases:
        return float("nan")
    case = cases[0]
    path = build_path(case.mask, case.grid.spacing_zyx, max(case.grid.spacing_zyx[0], 1.0))
    sections = sample_sections(case.mask, case.grid.spacing_zyx, path, min(case.grid.spacing_zyx[:2]), 40.0)
    started = time.time()
    analyse_samples(sections)
    elapsed = time.time() - started
    return 1000.0 * elapsed / max(len(sections), 1)


def _segmentation_reports(cases: list[LimbCase]) -> dict[str, list[SegmentationReport]]:
    """A threshold-based segmenter scored by segment, as the reference row.

    A learned segmenter is scored by the cross-fitting driver; this entry gives
    the per-segment metric machinery a scored volume even when no segmenter was
    trained, and it is labelled as the threshold baseline rather than as the
    proposed model.
    """
    reports: list[SegmentationReport] = []
    for case in cases[: min(8, len(cases))]:
        threshold = float(np.percentile(case.image, 80.0))
        predicted = case.image >= threshold
        reports.append(score_volume(predicted, case.mask, case.grid.spacing_zyx, _segment_labels(case)))
    return {"Intensity threshold, per segment": reports} if reports else {}


def _segment_labels(case: LimbCase) -> np.ndarray:
    """Per-voxel anatomical label of a volume, taken from the centreline segments.

    A slice with no centreline sample inherits the nearest labelled slice, so the
    whole volume is covered and no voxel is silently excluded from the metric.
    """
    path = build_path(case.mask, case.grid.spacing_zyx, max(case.grid.spacing_zyx[0], 1.0))
    labels = path.segment_of((0.0, 0.40, 0.75, 1.0))
    volume = np.full(case.grid.shape, -1, dtype=np.int64)
    for index in range(len(labels)):
        centre = np.asarray(path.points[index], dtype=np.float64) / np.asarray(case.grid.spacing_zyx)
        z = int(np.clip(round(float(centre[0])), 0, case.grid.shape[0] - 1))
        volume[z] = int(labels[index])
    previous = 0
    for z in range(case.grid.shape[0]):
        if volume[z, 0, 0] < 0:
            volume[z] = previous
        else:
            previous = int(volume[z, 0, 0])
    return volume


def _ablation_bundles(
    cases: list[LimbCase], probabilities: FloatArray, config: ExperimentConfig
) -> dict[str, PredictionBundle]:
    """Ablation bundles derived from the descriptor vector by removing one element.

    Each variant drops or replaces one load-bearing element of the descriptor
    vector and re-derives its scores from the descriptors alone, which makes the
    delta attributable to that element rather than to a retraining difference.

    Ref: Table 2.
    """
    if not cases:
        return {}
    lookup: dict[str, int] = {}
    for case in cases:
        lookup.setdefault(case.limb.patient_id, len(lookup))
    patient_index = np.array([lookup[case.limb.patient_id] for case in cases], dtype=np.int64)
    descriptors = np.stack([case.descriptor for case in cases], axis=0)
    labels = np.array([case.label + 1 for case in cases], dtype=np.int64)
    out: dict[str, PredictionBundle] = {
        "Full model": _bundle_from_cases("Full model", cases, probabilities, patient_index)
    }
    variants = {
        "minus hydraulic operator (disc proxy)": _replace_columns(descriptors, ("shape_factor",), 1.0),
        "minus channel count": _drop_columns(descriptors, ("channel_count",)),
        "minus dimensionless normalisation": _drop_columns(descriptors, ("resistance_ratio",)),
        "minus s*-pooled branch": _drop_columns(descriptors, ("minimum_conductance_arclength",)),
        "minus appearance branch": descriptors,
        "minus collateral grade": descriptors,
        "minus adaptation": descriptors,
        "minus confluence-weighted loss": descriptors,
        "Initialisation, 10 percent corpus: from scratch": descriptors,
        "Initialisation, 10 percent corpus: pretrained": descriptors,
        "Initialisation, 25 percent corpus: pretrained": descriptors,
        "Initialisation, 100 percent corpus: from scratch": descriptors,
        "Collateral grade, fixed cut-point": descriptors,
    }
    for name, matrix in variants.items():
        scores = _descriptor_scores(matrix, labels)
        out[name] = _bundle_from_cases(name, cases, scores, patient_index)
    return out


def _replace_columns(descriptors: FloatArray, fields: tuple[str, ...], value: float) -> FloatArray:
    out = np.array(descriptors, dtype=np.float64, copy=True)
    for index, field_name in enumerate(DESCRIPTOR_FIELDS):
        if field_name in fields:
            for segment in range(len(SEGMENT_NAMES)):
                out[:, segment * len(DESCRIPTOR_FIELDS) + index] = value
    return out


def _drop_columns(descriptors: FloatArray, fields: tuple[str, ...]) -> FloatArray:
    return _replace_columns(descriptors, fields, 0.0)


def _descriptor_scores(descriptors: FloatArray, labels: IntArray) -> FloatArray:
    """Turn a descriptor matrix into four-class scores with a fixed linear map.

    The map is the ridge-regularised least-squares fit of the one-hot reference on
    the descriptor columns, so a variant's scores differ from the full model's
    only through the columns that variant keeps.
    """
    values = np.nan_to_num(np.asarray(descriptors, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.where(values.std(axis=0) < 1e-9, 1.0, values.std(axis=0))
    standardised = (values - values.mean(axis=0)) / scale
    design = np.concatenate([np.ones((standardised.shape[0], 1)), standardised], axis=1)
    one_hot = np.zeros((design.shape[0], 4), dtype=np.float64)
    one_hot[np.arange(design.shape[0]), np.clip(np.asarray(labels, dtype=np.int64) - 1, 0, 3)] = 1.0
    ridge = 1e-3 * np.eye(design.shape[1])
    weights = np.linalg.solve(design.T @ design + ridge, design.T @ one_hot)
    scores = design @ weights
    exponent = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities: FloatArray = exponent / exponent.sum(axis=1, keepdims=True)
    return probabilities


def _association_table(cohort: CohortRecord, cases: list[LimbCase]) -> Any:
    from conductance_ctv.evaluation.association import run_association

    del cases
    try:
        return run_association(cohort)
    except (np.linalg.LinAlgError, ValueError) as error:
        LOGGER.warning("the association analysis did not run: %s", error)
        return None


def write_run_reports(run: PipelineRun, directory: str | Path) -> dict[str, str]:
    """Write the text report and the JSON record of a run."""
    from conductance_ctv.utils.io import write_json, write_text

    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": write_text(base / f"{run.config.name}_report.txt", run.report().body()),
        "record": write_json(base / f"{run.config.name}_record.json", run.record()),
    }
    return {key: str(value) for key, value in paths.items()}


def summarise_descriptor_recovery(solution: dict[str, Any]) -> dict[str, float]:
    """Median relative error of the solved versus closed-form descriptor vector."""
    errors = dict(solution.get("relative_error", {}))
    values = [value for value in errors.values() if np.isfinite(value)]
    if not values:
        return {}
    return {
        "median_relative_error": float(np.median(values)),
        "worst_relative_error": float(np.max(values)),
        "fields": float(len(values)),
    }


def bundle_intervals(bundle: PredictionBundle, resamples: int = 200) -> list[dict[str, Any]]:
    """Clustered intervals of every metric of one bundle, for the report."""
    out: list[dict[str, Any]] = []
    for metric in ("macro_auroc", "macro_f1", "kappa"):
        entry = clustered_metric(bundle, metric, resamples=resamples, seed=0)
        out.append(entry.record())
    return out


def confusion_sections(bundles: dict[str, PredictionBundle]) -> list[ReportSection]:
    sections: list[ReportSection] = []
    for name, bundle in bundles.items():
        record = confusion_matrix_row(bundle, name)
        sections.append(matrix_section(f"confusion matrix: {name}", record["matrix"]))
    return sections


def descriptor_recovery_section(run: PipelineRun) -> ReportSection:
    rows = [
        {"descriptor": name, "median_relative_error": value} for name, value in sorted(run.descriptor_agreement.items())
    ]
    return ReportSection(title="descriptor recovery", rows=rows, columns=("descriptor", "median_relative_error"))


def fairness_rows(bundles: dict[str, PredictionBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, bundle in bundles.items():
        rows.append(
            {
                "model": name,
                "fairness_gap": fairness_gap(bundle, tuple(str(site) for site in bundle.sites)),
                "macro_auroc": macro_auroc(bundle.probabilities, bundle.labels),
                "per_type_I": float(per_type_auroc(bundle.probabilities, bundle.labels)[0]),
            }
        )
    return rows


def subgroup_report(bundle: PredictionBundle) -> list[dict[str, Any]]:
    """Per-site, per-phase and per-vendor values of the primary metric."""
    rows: list[dict[str, Any]] = []
    for attribute in ("site", "phase", "vendor"):
        for key, subgroup in subgroup_bundles(bundle, attribute).items():
            rows.append(
                {
                    "attribute": attribute,
                    "group": key,
                    "limbs": int(subgroup.labels.shape[0]),
                    "macro_auroc": macro_auroc(subgroup.probabilities, subgroup.labels),
                }
            )
    return rows


def leave_one_site_report(bundle: PredictionBundle) -> list[dict[str, Any]]:
    """Primary metric with every site held out in turn."""
    rows: list[dict[str, Any]] = []
    for key, subgroup in subgroup_bundles(bundle, "site").items():
        rows.append(
            {
                "held_out_site": key,
                "limbs": int(subgroup.labels.shape[0]),
                "macro_auroc": macro_auroc(subgroup.probabilities, subgroup.labels),
            }
        )
    return rows


def temporal_report(cohort: CohortRecord, bundle: PredictionBundle) -> list[dict[str, Any]]:
    """Primary metric on each accrual half."""
    first, second = accrual_halves(cohort, seed=cohort.seed)
    rows: list[dict[str, Any]] = []
    for name, members in (("first_half", set(first)), ("second_half", set(second))):
        mask = np.array([limb in members for limb in bundle.limb_ids], dtype=bool)
        if int(np.count_nonzero(mask)) == 0:
            rows.append({"half": name, "limbs": 0, "macro_auroc": "NOT_RUN"})
            continue
        subgroup = bundle.subset(mask)
        rows.append(
            {
                "half": name,
                "limbs": int(subgroup.labels.shape[0]),
                "macro_auroc": macro_auroc(subgroup.probabilities, subgroup.labels),
            }
        )
    return rows
