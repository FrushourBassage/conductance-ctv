"""The paper-claim to code map, and the record of every intentional deviation.

Each entry names a location in the manuscript, states the claim in one line, and
points at the file and symbol that carries it. The status of an entry is filled
in from the check that exercises it, so a claim is never marked verified by
assertion.

Ref: the verification pass described in the project context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_STATUS = "NOT_RUN"


@dataclass
class Claim:
    """One manuscript claim with the code that carries it and the check that tests it."""

    identifier: str
    paper_location: str
    claim: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    method: str
    check: str
    status: str = DEFAULT_STATUS
    evidence: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "paper_location": self.paper_location,
            "claim": self.claim,
            "files": list(self.files),
            "symbols": list(self.symbols),
            "method": self.method,
            "check": self.check,
            "status": self.status,
            "evidence": self.evidence,
        }


CLAIMS: tuple[Claim, ...] = (
    Claim(
        identifier="flow-problem",
        paper_location="Sec. 6.5, Eq. (1)",
        claim=(
            "The conductance of a section is the integral of the solution of "
            "-div(grad u) = 1 with u = 0 on the lumen boundary."
        ),
        files=(
            "src/conductance_ctv/hydraulics/dirichlet.py",
            "src/conductance_ctv/hydraulics/conductance.py",
        ),
        symbols=("solve_poisson", "solve_unit_source", "conductance_of_section"),
        method=(
            "Solve the problem on an axis-aligned rectangle, whose conductance has a separable "
            "Fourier series that the grid represents exactly, and compare."
        ),
        check="solver_rectangle_series",
    ),
    Claim(
        identifier="density-normalisation",
        paper_location="Sec. 6.6 and Algorithm 2",
        claim=(
            "A limb's descriptor vector and a backbone feature vector gathered in a 20 mm tube "
            "around s* are combined and fused; the severity is produced by the descriptor branch alone."
        ),
        files=(
            "src/conductance_ctv/models/branches.py",
            "src/conductance_ctv/models/classifier.py",
            "src/conductance_ctv/data/sampling.py",
        ),
        symbols=("DescriptorBranch", "AppearanceBranch", "FusionHead", "appearance_tube_indices"),
        method="Build the model and check the branch dimensions, the tube radius and the severity path.",
        check="model_branch_wiring",
    ),
    Claim(
        identifier="polya-szego",
        paper_location="Sec. 6.5, Proposition 1 and Eq. (2)",
        claim=(
            "G <= A^2 / (8 pi) with equality only for the disc, and G >= A^2 / (8 pi k) for k "
            "disc channels of combined area A."
        ),
        files=(
            "src/conductance_ctv/hydraulics/conductance.py",
            "src/conductance_ctv/hydraulics/shapes.py",
        ),
        symbols=("polya_szego_upper", "polya_szego_channel_lower", "disc_conductance", "septum_section"),
        method=(
            "Evaluate both bounds on digital discs, ellipses and equal-channel septa and compare with the solved"
            " conductance."
        ),
        check="polya_szego_bounds",
    ),
    Claim(
        identifier="slit-shape-factor-range",
        paper_location="Sec. 1",
        claim="For slit-like compression with an axis ratio from 0.15 to 0.30 the shape factor is about 1.8 to 3.4.",
        files=("src/conductance_ctv/hydraulics/conductance.py",),
        symbols=("shape_factor_from_axis_ratio", "axis_ratio_from_shape_factor"),
        method="Evaluate the closed form of the elliptic shape factor at the two ends of the quoted ratio range.",
        check="shape_factor_range",
    ),
    Claim(
        identifier="series-resistance",
        paper_location="Sec. 6.5 and Algorithm 1 line 10",
        claim=(
            "The series resistance of a span is the sum of ds / G over its samples, and an occluded span has G = 0 and"
            " is reported separately."
        ),
        files=(
            "src/conductance_ctv/hydraulics/series.py",
            "src/conductance_ctv/hydraulics/descriptors.py",
        ),
        symbols=("series_resistance", "occlusion_length", "describe_sections"),
        method=(
            "Compare the sum over a conductance profile against a hand-computed value and check the occluded" " branch."
        ),
        check="series_resistance_arithmetic",
    ),
    Claim(
        identifier="dimensionless-ratio",
        paper_location="Sec. 6.5 and Sec. 3.3",
        claim="rho_R = R / R_ref, formed against a lesion-free span, is dimensionless and survives transport.",
        files=("src/conductance_ctv/hydraulics/series.py",),
        symbols=("reference_resistance", "resistance_ratio", "conductance_weighted_length"),
        method=(
            "Scale a section by a constant and check that rho_R and the shape factor are unchanged while the absolute"
            " area is not."
        ),
        check="dimensionless_invariance",
    ),
    Claim(
        identifier="minimum-conductance-location",
        paper_location="Algorithm 1 line 8",
        claim="s* is the arc length of the minimum-conductance sample, not of the minimum-area sample.",
        files=("src/conductance_ctv/hydraulics/series.py", "src/conductance_ctv/hydraulics/descriptors.py"),
        symbols=("minimal_conductance_index", "minimal_area_index", "describe_sections"),
        method=(
            "Build a profile whose conductance minimum and area minimum are at different samples and check which one is"
            " selected."
        ),
        check="star_selection",
    ),
    Claim(
        identifier="descriptor-vector",
        paper_location="Sec. 6.5 descriptor list",
        claim=(
            "Per segment the vector carries the minimal luminal area, the arc length of the "
            "minimum-conductance sample, the shape factor, the channel count, the series resistance, "
            "the dimensionless resistance ratio and the conductance-weighted length."
        ),
        files=("src/conductance_ctv/hydraulics/descriptors.py",),
        symbols=("DESCRIPTOR_FIELDS", "SEGMENT_NAMES", "HydraulicSignature", "descriptor_feature_names"),
        method="Count the fields and the segments and check that the concatenated vector has the declared length.",
        check="descriptor_layout",
    ),
    Claim(
        identifier="four-types",
        paper_location="Sec. 6.1",
        claim=(
            "Type I is obstructive compression without thrombus, type II compression with "
            "collaterals, type III post-thrombotic stenosis, type IV complete obstruction."
        ),
        files=("src/conductance_ctv/data/schema.py", "src/conductance_ctv/data/fixture.py"),
        symbols=("LesionType", "LESION_TYPE_DESCRIPTION", "generate_limb"),
        method=(
            "Check the enum members, their descriptions and that the generator reproduces each type as a distinct"
            " section family."
        ),
        check="lesion_type_definitions",
    ),
    Claim(
        identifier="three-segments",
        paper_location="Sec. 6.2",
        claim=(
            "Segmentation and stenosis are quantified per segment: common iliac, external iliac and common femoral vein"
            " on each limb."
        ),
        files=("src/conductance_ctv/hydraulics/centerline.py", "src/conductance_ctv/metrics/segmentation.py"),
        symbols=("SEGMENT_NAMES", "PathSamples.segment_of", "score_volume"),
        method="Check that the path labels cover the three segments and that the scorer returns one value per segment.",
        check="segment_units",
    ),
    Claim(
        identifier="analysis-units",
        paper_location="Sec. 6.2 and Sec. 6.8",
        claim=(
            "Segment-level figures are never mistaken for patient-level figures, and every interval is a"
            " patient-clustered percentile interval."
        ),
        files=("src/conductance_ctv/metrics/bootstrap.py", "src/conductance_ctv/evaluation/protocol.py"),
        symbols=("patient_clusters", "clustered_bootstrap", "unit_counts"),
        method=(
            "Resample patients and check that the interval is wider than a limb-level resample of the same statistic."
        ),
        check="clustered_resampling",
    ),
    Claim(
        identifier="mcid",
        paper_location="Sec. 6.1",
        claim="The MCID of the primary endpoint was pre-set at 0.05 and all primary results are reported against it.",
        files=("src/conductance_ctv/evaluation/tables.py", "src/conductance_ctv/verification/checks.py"),
        symbols=("table_3_comparators",),
        method=(
            "Check that the pre-specified constant is carried in the release and used when the primary comparison is"
            " reported."
        ),
        check="prespecified_constants",
    ),
    Claim(
        identifier="prespecified-criteria",
        paper_location="Sec. 6.7",
        claim=(
            "External macro-AUROC against 0.900, ICC at or above 0.900 for the minimal luminal area, "
            "a management-relevant decision change above 0.10 and a reader non-inferiority margin of "
            "0.05 in Cohen's kappa."
        ),
        files=("src/conductance_ctv/verification/checks.py", "src/conductance_ctv/models/classifier.py"),
        symbols=("PRESPECIFIED_CRITERIA", "decision_rule", "operating_point"),
        method=(
            "Check that the four criteria are recorded with their thresholds and that the decision rule is applied to"
            " the sum of the two severe probabilities."
        ),
        check="prespecified_constants",
    ),
    Claim(
        identifier="decisive-type",
        paper_location="Table 3 caption",
        claim="The per-type areas are reported independently because the rarest and hardest type carries the decision.",
        files=("src/conductance_ctv/evaluation/tables.py", "src/conductance_ctv/metrics/classification.py"),
        symbols=("table_3_comparators", "per_type_auroc"),
        method="Check that the third table emits one column per type and a macro aggregate over all four.",
        check="per_type_reporting",
    ),
    Claim(
        identifier="unconstrained-comparison",
        paper_location="Table 3 caption and Sec. 6.8",
        claim=(
            "Comparisons are against the best unconstrained comparator, by DeLong for areas and by McNemar for paired"
            " calls."
        ),
        files=(
            "src/conductance_ctv/evaluation/protocol.py",
            "src/conductance_ctv/metrics/inference_stats.py",
        ),
        symbols=("run_protocol", "delong_auc_pair", "mcnemar"),
        method="Check that the pipeline selects the best comparator on the primary metric and reports both tests.",
        check="paired_comparisons",
    ),
    Claim(
        identifier="dice-hd95-cldice",
        paper_location="Table 1 and its caption",
        claim=(
            "Segmentation is reported by Dice, the 95th-percentile Hausdorff distance and centreline Dice, per segment."
        ),
        files=(
            "src/conductance_ctv/metrics/segmentation.py",
            "src/conductance_ctv/metrics/skeleton.py",
        ),
        symbols=("dice_score", "hausdorff_distance", "centreline_dice", "ridge_skeleton"),
        method=(
            "Check each metric against a hand-computable case and check that centreline Dice exceeds Dice on a"
            " single-trunk vessel."
        ),
        check="segmentation_metrics",
    ),
    Claim(
        identifier="confluence-weighted-loss",
        paper_location="Sec. 6.4",
        claim=(
            "The segmentation cost is weighted towards a 10 mm cylinder around the iliocaval confluence instead of a"
            " uniform voxel cost."
        ),
        files=(
            "src/conductance_ctv/data/sampling.py",
            "src/conductance_ctv/losses/segmentation.py",
        ),
        symbols=("confluence_weight_map", "CONFLUENCE_CYLINDER_MM", "ConfluenceWeightedLoss"),
        method=(
            "Check the cylinder radius and that the weight map is above one inside the cylinder and exactly one"
            " outside it."
        ),
        check="confluence_weighting",
    ),
    Claim(
        identifier="patient-crossfitting",
        paper_location="Sec. 6.4",
        claim=(
            "Fine-tuning uses a patient-level cross-fitting scheme so no evaluation limb was seen by the segmenter that"
            " scores it."
        ),
        files=("src/conductance_ctv/data/splits.py", "src/conductance_ctv/training/crossfit.py"),
        symbols=("crossfit_folds", "crossfit_segmenter", "out_of_fold_indices"),
        method="Build the folds and check they are disjoint over patients and cover the development set.",
        check="crossfit_disjointness",
    ),
    Claim(
        identifier="patient-split",
        paper_location="Sec. 6.8 and Algorithm 3 line 1",
        claim=(
            "One patient-level split fixes the development and tuning partitions and every baseline is scored on it;"
            " the external region is sealed until the final evaluation."
        ),
        files=("src/conductance_ctv/data/splits.py", "src/conductance_ctv/evaluation/protocol.py"),
        symbols=("patient_level_split", "PredictionBundle", "run_protocol"),
        method=(
            "Check that the split is disjoint over patients and that the protocol rejects misaligned comparison pairs."
        ),
        check="split_disjointness",
    ),
    Claim(
        identifier="class-balanced-loss",
        paper_location="Sec. 6.6",
        claim="The fused branches are trained under a class-balanced multinomial loss.",
        files=("src/conductance_ctv/losses/class_balanced.py",),
        symbols=("effective_number_weights", "ClassBalancedMultinomialLoss", "TypeModelLoss"),
        method=(
            "Check the effective-number weights against the closed form and check that the loss is finite and"
            " differentiable."
        ),
        check="class_balanced_weights",
    ),
    Claim(
        identifier="severity-regression",
        paper_location="Algorithm 2 line 3",
        claim="The reported severity is a descriptor-branch regression output with no appearance input.",
        files=("src/conductance_ctv/models/branches.py", "src/conductance_ctv/models/classifier.py"),
        symbols=("DescriptorOnlySeverity", "ConductanceTypeModel.forward"),
        method="Perturb the appearance input and check the descriptor severity is unchanged.",
        check="severity_independent_of_appearance",
    ),
    Claim(
        identifier="decision-rule",
        paper_location="Algorithm 2 line 4 and Sec. 6.6",
        claim=(
            "The recommendation is one when the calibrated probabilities of the post-thrombotic and occlusive types sum"
            " to at least the pre-specified threshold."
        ),
        files=("src/conductance_ctv/models/classifier.py",),
        symbols=("decision_rule", "operating_point", "threshold_sensitivity_band"),
        method="Check the rule against a direct evaluation of the sum for a batch of probability vectors.",
        check="decision_rule_arithmetic",
    ),
    Claim(
        identifier="temperature-scaling",
        paper_location="Sec. 6.6",
        claim="Calibration is temperature scaling fitted in the tuning partition only.",
        files=("src/conductance_ctv/models/classifier.py",),
        symbols=("TemperatureCalibrator", "apply_temperature_"),
        method=(
            "Fit a temperature on an over-confident model and check the negative log-likelihood decreases and"
            " the temperature exceeds one."
        ),
        check="temperature_scaling",
    ),
    Claim(
        identifier="degradation-direction",
        paper_location="Sec. 3.3 and Table 4",
        claim="Dimensionless descriptors degrade less than absolute-area descriptors when transported.",
        files=(
            "src/conductance_ctv/metrics/agreement.py",
            "src/conductance_ctv/hydraulics/descriptors.py",
        ),
        symbols=("family_of", "relative_degradation", "dimensionless_columns", "absolute_columns"),
        method=(
            "Separate the descriptor columns into the two families, check the split covers the vector, and check that"
            " the dimensionless family is invariant under an emulated scanner scale change while the absolute family is"
            " not."
        ),
        check="descriptor_family_split",
    ),
    Claim(
        identifier="propagated-error",
        paper_location="Sec. 3.2 and Table 5",
        claim=(
            "The series resistance scales as the inverse square of area, so a 4.1 percent area error becomes 8.2"
            " percent in rho_R."
        ),
        files=("src/conductance_ctv/hydraulics/series.py",),
        symbols=("propagated_resistance_error", "resistance_sensitivity"),
        method="Compare the closed form of the inverse-square propagation with a numerical perturbation.",
        check="propagated_error",
    ),
    Claim(
        identifier="icc-and-bland-altman",
        paper_location="Sec. 6.8 and Table 5",
        claim="Morphometric agreement is reported as ICC(2, 1) with Bland-Altman bias and limits of agreement.",
        files=("src/conductance_ctv/metrics/agreement.py",),
        symbols=("icc_2_1", "bland_altman", "agreement_table"),
        method=(
            "Compare the estimator with the mean-square formula evaluated by hand and check the limits against the"
            " closed form."
        ),
        check="icc_arithmetic",
    ),
    Claim(
        identifier="per-segment-parameters",
        paper_location="Table 1 parameter column",
        claim="The parameter totals of the compared architectures are reported per model.",
        files=("src/conductance_ctv/models/encoders.py",),
        symbols=("ENCODER_SPECS", "resolve_width_bracket", "parameter_table"),
        method="Check that each reported total is bracketed by its family and resolved to the nearer neighbour.",
        check="parameter_resolution",
    ),
    Claim(
        identifier="proposed-model-total",
        paper_location="Table 1 parameter column",
        claim=(
            "The proposed model is reported at 133.6 M parameters, against 131.0 M for its fine-tuned "
            "backbone alone."
        ),
        files=(
            "src/conductance_ctv/models/unet3d.py",
            "src/conductance_ctv/models/decoder.py",
            "configs/model/foundation_trunk.yaml",
        ),
        symbols=("build_segmentation_network", "AnisotropicDecoder3D", "segmentation_total_millions"),
        method=(
            "Build the declared primary configuration for real and check that its total brackets the reported "
            "one and sits on the nearer of the two achievable totals."
        ),
        check="proposed_total",
    ),
    Claim(
        identifier="public-corpora",
        paper_location="Data availability",
        claim=(
            "The TotalSegmentator computed tomography dataset release 2.0.1 under CC BY 4.0 and the "
            "AbdomenAtlas computed tomography dataset were used for pretraining; neither supplies "
            "evaluation data or a clinical variable."
        ),
        files=("src/conductance_ctv/data/manifest.py", "dataset_urls.txt"),
        symbols=("PRETRAINING_CORPORA", "pretraining_manifest", "label_index_map"),
        method=(
            "Check that each corpus is recorded with its pinned version and licence and is listed in the link "
            "list under one of its addresses; a refresh run re-probes the addresses as well."
        ),
        check="dataset_provenance",
    ),
    Claim(
        identifier="private-cohort",
        paper_location="Data availability",
        claim=(
            "The sensitive raw clinical records are not shared; the derived data and code are available from the"
            " corresponding author on request."
        ),
        files=("src/conductance_ctv/verification/checks.py",),
        symbols=("check_private_cohort_absent",),
        method=(
            "Scan the tree for any bundled clinical record or credential and confirm that only the cohort fixture"
            " cohort is present."
        ),
        check="private_cohort_absent",
    ),
    Claim(
        identifier="readers",
        paper_location="Sec. 6.3",
        claim="The type reference standard is a two-reader consensus with a third reader adjudicating disagreements.",
        files=("src/conductance_ctv/data/schema.py", "src/conductance_ctv/data/fixture.py"),
        symbols=("LimbRecord.reader_consensus", "confusion_counts", "_reader_call"),
        method=(
            "Check that two reader calls are recorded per limb, that the consensus is defined only when they agree and"
            " that a boundary disagreement rate exists."
        ),
        check="reader_records",
    ),
    Claim(
        identifier="no-pressure-terms",
        paper_location="Sec. 4.1",
        claim=(
            "The analysis is purely geometric: the descriptors are geometric resistances and no pressure is assumed"
            " anywhere."
        ),
        files=("src/conductance_ctv/hydraulics/", "src/conductance_ctv/hydraulics/series.py"),
        symbols=("series_resistance", "conductance_of_section"),
        method="Scan the hydraulics package for any pressure symbol or unit and confirm the descriptors are geometric.",
        check="geometric_only",
    ),
)


DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "paper_location": "Sec. 6.2 and Data availability",
        "departure": (
            "The clinical cohort is not bundled. The release supplies a schema-compatible cohort fixture whose "
            "sections are prescribed in closed form, so every mechanism is executable and the solved descriptors "
            "can be scored against an analytic reference."
        ),
        "reason": "The manuscript states that the raw records are not shared because of re-identification risk.",
        "impact": (
            "All cohort-level point values are NOT_RUN; the mechanisms, metrics and tables are executed on the fixture"
            " cohort."
        ),
    },
    {
        "paper_location": "Table 1 parameter column",
        "departure": (
            "The per-layer widths of each architecture family are not stated. The release declares a family per "
            "encoder and ships resolve_width_bracket, which derives the width multiplier whose built model is the "
            "nearer of the two achievable parameter totals around the reported one; the measured count is reported "
            "beside the reported one and the residual is the width granularity."
        ),
        "reason": "The manuscript reports parameter totals without the layer specification.",
        "impact": (
            "Each reported total is bracketed by its family; the measured count and the width granularity are the "
            "values the run reports, so a residual is never presented as an exact match."
        ),
    },
    {
        "paper_location": "Sec. 6.4 training details",
        "departure": (
            "Batch size, learning rate, schedule, epoch count, precision, gradient clipping and the EMA decay are "
            "engineering defaults. The manuscript reports 214 A100 GPU-hours and the parameter total only."
        ),
        "reason": "The training configuration is not reported in the manuscript.",
        "impact": "configs/train/main.yaml carries the defaults and every one is listed here.",
    },
    {
        "paper_location": "Sec. 6.4",
        "departure": (
            "The magnitude of the confluence weight is an engineering default (3.0 inside the cylinder against 1.0 "
            "outside). The manuscript states the direction of the weighting and the 10 mm radius."
        ),
        "reason": "The manuscript gives the radius and the direction, not the ratio.",
        "impact": "loss.confluence_inside_weight is configurable and defaults to 3.0.",
    },
    {
        "paper_location": "Algorithm 1 line 4",
        "departure": (
            "The resolution floor below which a component is not counted as a channel is defined as the area of the "
            "disc inscribed in a two-cell span, which the manuscript leaves unspecified."
        ),
        "reason": "The manuscript refers to a resolution floor without fixing it.",
        "impact": "hydraulics.resolution_floor_cells exposes the choice; the default is two cells.",
    },
    {
        "paper_location": "Sec. 6.5 descriptor list",
        "departure": (
            "The conductance-weighted length is the resistance-equivalent length L * (R_ref / R) over the patent "
            "samples, so it is a length in millimetres that shrinks as the lumen narrows. The manuscript lists the "
            "quantity without its normalisation; a direct weighting sum_k (G / G_ref) ds would instead return the "
            "span length exactly and carry no information about the profile."
        ),
        "reason": "The normalisation of the weighted length is not stated.",
        "impact": (
            "The descriptor is a length and scales with the geometry; the choice is recorded here and the invariance "
            "check exercises it. rho_R is unaffected, since it uses the reference resistance directly."
        ),
    },
    {
        "paper_location": "Sec. 6.5",
        "departure": (
            "A section area uses the half-cell boundary convention: the occupied set grown by half a cell, "
            "A = (N + L/2) h^2 with L the occupied-to-free interface length. This is the domain the five-point "
            "stencil discretises and it is what removes a first-order grid bias from the shape factor."
        ),
        "reason": (
            "The manuscript specifies the mask and the grid but not the boundary convention of the discretisation."
        ),
        "impact": (
            "Shape factors of digital sections are reported under this convention; the solver accuracy is measured"
            " against closed forms."
        ),
    },
    {
        "paper_location": "Table 1 clDice column",
        "departure": (
            "The centreline is extracted as the medial ridge of the exact Euclidean distance transform. The "
            "manuscript names centreline Dice as a metric and does not fix the skeletonisation algorithm."
        ),
        "reason": "The skeletonisation algorithm is not stated.",
        "impact": (
            "The property the caption attributes to clDice, that it exceeds Dice on a single-trunk vessel, is checked"
            " in the test suite."
        ),
    },
    {
        "paper_location": "Sec. 6.5 reference span",
        "departure": (
            "The lesion-free reference span is chosen as the candidate with the highest mean conductance among the "
            "other segments, with upstream and explicit modes also implemented. The manuscript says the reference "
            "comes from the opposite or the upstream segment without fixing which."
        ),
        "reason": "The selection rule is not fixed in the manuscript.",
        "impact": "hydraulics.reference_mode selects the rule; the default is max.",
    },
    {
        "paper_location": "Sec. 6.1 three segment interfaces",
        "departure": (
            "The interfaces between the common iliac, external iliac and common femoral segments sit at 0.40 and "
            "0.75 of the confluence-to-bifurcation arc. The manuscript names the three segments without stating "
            "their proportions."
        ),
        "reason": "The anatomical split is not stated as fractions.",
        "impact": "hydraulics.segment_fractions is configurable and defaults to the values above.",
    },
    {
        "paper_location": "Table 3 first row",
        "departure": (
            "The single-diameter rule emits four one-versus-rest scores built from the diameter-derived stenosis "
            "fraction and the two binary observations a diameter reading supports, a spur and a closed lumen. The "
            "manuscript reports the row without defining how a one-dimensional measurement becomes four scores."
        ),
        "reason": "The mapping from one diameter to four type scores is not stated.",
        "impact": "models.baselines.diameter_rule_scores carries the mapping and is labelled as the baseline it is.",
    },
    {
        "paper_location": "Sec. 6.9 safety arm",
        "departure": (
            "The reader arm and the safety arm need four human readers and a pre-registered adverse-event "
            "endpoint, neither of which is reproducible from a repository. The reader disagreement model is "
            "carried by the cohort fixture and the human arms are reported NOT_RUN."
        ),
        "reason": "Human readers and a prospective safety endpoint cannot be reproduced from code.",
        "impact": "Table 6 reports the fixture arm's reader deltas and marks the human arms NOT_RUN.",
    },
    {
        "paper_location": "compute profile",
        "departure": (
            "The reported wall-clock is 214 A100 GPU-hours. The environment this release was verified in has no "
            "accelerator, so the headline run was not executed at scale and the GPU-hour figure is NOT_RUN."
        ),
        "reason": "No accelerator is available in the verification environment.",
        "impact": "The training loops were executed at reduced scale; the full-scale figure is reported NOT_RUN.",
    },
    {
        "paper_location": "Data availability",
        "departure": (
            "The pretraining corpora are declared and their licences recorded, but they were not staged: "
            "TotalSegmentator release 2.0.1 alone is 23.6 GB and the transfer rate to it is throttled from here."
        ),
        "reason": "The corpora cannot be fetched within the environment's budget.",
        "impact": "Pretraining is NOT_RUN; the encoder families and the staging plan are nevertheless executable.",
    },
)


def claims_by_check() -> dict[str, list[str]]:
    """Claim identifiers grouped by the check that exercises them."""
    out: dict[str, list[str]] = {}
    for claim in CLAIMS:
        out.setdefault(claim.check, []).append(claim.identifier)
    return out


def apply_status(
    claims: tuple[Claim, ...], statuses: dict[str, str], evidence: dict[str, dict[str, Any]]
) -> list[Claim]:
    """Write check outcomes into the claims that depend on them."""
    updated: list[Claim] = []
    for claim in claims:
        claim.status = statuses.get(claim.check, DEFAULT_STATUS)
        claim.evidence = evidence.get(claim.check, {})
        updated.append(claim)
    return updated
