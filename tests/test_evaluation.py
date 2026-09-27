"""The evaluation protocol, the reported tables, the report writer and the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from conductance_ctv.data.fixture import generate_cohort
from conductance_ctv.data.schema import DEVELOPMENT_SITES, EXTERNAL_SITE, Site
from conductance_ctv.evaluation.association import (
    PRESENTATION_ENDPOINTS,
    build_columns,
    endpoint_labels,
    run_association,
    scaled_columns,
)
from conductance_ctv.evaluation.protocol import (
    PredictionBundle,
    clustered_metric,
    leave_one_site_out,
    logit_average,
    metric_function,
    paired_comparison,
    probability_average,
    run_protocol,
    significance_of_difference,
    subgroup_bundles,
    temporal_halves,
    unit_counts,
)
from conductance_ctv.evaluation.report import (
    Report,
    ReportSection,
    comparison_section,
    format_value,
    intervals_to_rows,
    matrix_section,
    render_table,
    summary_section,
)
from conductance_ctv.evaluation.tables import (
    confusion_matrix_row,
    fairness_gap,
    interaction_ratio,
    table_1_segmentation,
    table_2_components,
    table_3_comparators,
    table_4_transport,
    table_5_morphometry,
    table_6_clinical,
    table_7_association,
)
from conductance_ctv.metrics.segmentation import score_volume


def _bundle(name: str, count: int = 240, seed: int = 3, sites: tuple[Site, ...] | None = None) -> PredictionBundle:
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, 5, size=count)
    probabilities = np.eye(4)[labels - 1] * 0.45 + rng.random((count, 4)) * 0.55
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    site_values = sites if sites is not None else tuple(Site.A if index % 2 else Site.B for index in range(count))
    return PredictionBundle(
        name=name,
        limb_ids=tuple(f"L{index:04d}" for index in range(count)),
        patient_index=np.repeat(np.arange(count // 2), 2),
        labels=labels.astype(np.int64),
        probabilities=probabilities,
        sites=site_values,
        phases=tuple("direct" if index % 3 else "delayed_indirect" for index in range(count)),
        vendors=tuple(("vendor_a", "vendor_b")[index % 2] for index in range(count)),
    )


def test_prediction_bundle_take_and_subset() -> None:
    bundle = _bundle("model")
    taken = bundle.take(np.array([0, 1, 2], dtype=np.int64))
    assert len(taken.limb_ids) == 3
    assert taken.probabilities.shape == (3, 4)
    assert taken.sites == bundle.sites[:3]
    # Rows 0 and 1 belong to patient 0, so drawing row 1 twice keeps that patient.
    repeated = bundle.take(np.array([0, 0, 1], dtype=np.int64))
    assert repeated.patient_index.tolist() == [0, 0, 0]
    mask = np.zeros(len(bundle.limb_ids), dtype=bool)
    mask[:5] = True
    subset = bundle.subset(mask)
    assert len(subset.limb_ids) == 5
    with pytest.raises(ValueError):
        bundle.subset(np.zeros(3, dtype=bool))
    renamed = bundle.with_name("other")
    assert renamed.name == "other"
    assert renamed.labels.tolist() == bundle.labels.tolist()


def test_prediction_bundle_validates_shapes() -> None:
    with pytest.raises(ValueError):
        PredictionBundle(
            name="bad",
            limb_ids=("a", "b"),
            patient_index=np.zeros(3, dtype=np.int64),
            labels=np.zeros(2, dtype=np.int64),
            probabilities=np.zeros((2, 4)),
            sites=(Site.A, Site.B),
        )


def test_metric_functions_and_clustered_metrics() -> None:
    bundle = _bundle("model")
    for metric in ("macro_auroc", "macro_f1", "kappa", "balanced_accuracy"):
        value = metric_function(metric)(bundle)
        assert np.isfinite(value)
    entry = clustered_metric(bundle, "macro_auroc", resamples=60, seed=1)
    assert entry.unit == "patient"
    assert entry.result.lower <= entry.result.upper
    assert entry.record()["model"] == "model"
    with pytest.raises(KeyError):
        metric_function("nonsense")


def test_unit_counts_and_subgroups() -> None:
    bundle = _bundle("model")
    counts = unit_counts(bundle)
    assert counts["limbs"] == len(bundle.limb_ids)
    assert counts["segments"] == 3 * counts["limbs"]
    assert counts["patients"] == 120
    for attribute in ("site", "phase", "vendor"):
        groups = subgroup_bundles(bundle, attribute)
        assert groups
        assert sum(len(group.limb_ids) for group in groups.values()) == len(bundle.limb_ids)
    with pytest.raises(KeyError):
        subgroup_bundles(bundle, "nonsense")


def test_leave_one_site_out_and_temporal_halves() -> None:
    sites = tuple(Site.A if index % 3 == 0 else Site.B if index % 3 == 1 else Site.C for index in range(240))
    bundle = _bundle("model", sites=sites)
    folds = leave_one_site_out(bundle)
    assert set(folds) == {str(site) for site in Site}
    first = tuple(bundle.limb_ids[:120])
    second = tuple(bundle.limb_ids[120:])
    halves = temporal_halves(bundle, (first, second))
    assert "difference" in halves
    assert halves["first_half"]["limbs"] == 120.0


def test_paired_comparison_and_protocol() -> None:
    proposed = _bundle("Conductance-3D-CTV (proposed)", seed=4)
    comparator = _bundle("comparator", seed=4)
    paired = paired_comparison(proposed, comparator)
    assert paired.delong is not None
    assert paired.mcnemar is not None
    assert "delong" in paired.record()
    mismatched = _bundle("other", count=100, seed=4)
    with pytest.raises(ValueError):
        paired_comparison(proposed, mismatched)
    outcome = run_protocol([proposed, comparator], metrics=("macro_auroc",), resamples=40, seed=2)
    assert outcome.metrics
    assert outcome.units["limbs"] == 240
    assert len(outcome.paired) == 1
    assert significance_of_difference(0.5, 0.1) < 0.01
    assert np.isnan(significance_of_difference(0.5, 0.0))
    assert run_protocol([]).metrics == []


def test_probability_and_logit_averaging() -> None:
    bundles = [_bundle(f"run{index}", seed=index) for index in range(3)]
    averaged = probability_average(bundles)
    assert averaged.probabilities.shape == bundles[0].probabilities.shape
    assert np.allclose(averaged.probabilities.sum(axis=1), 1.0)
    assert averaged.name.endswith("avg3")
    logit = logit_average(bundles)
    assert np.allclose(logit.probabilities.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        probability_average([])
    with pytest.raises(ValueError):
        logit_average([])


def test_table_one_segmentation() -> None:
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[2:14, 6:10, 6:10] = True
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[8:] = 1
    report = score_volume(mask, mask, (1.0, 1.0, 1.0), labels)
    table = table_1_segmentation(
        {"method": [report], "empty": []},
        parameters={"method": 12.5},
        inference_seconds={"method": 0.3},
        units={"volumes": 1},
    )
    assert table.number == 1
    assert table.rows[0]["dice_CIV"] == pytest.approx(1.0)
    assert table.rows[1]["status"] == "NOT_RUN"
    assert table.status == "PASS"
    assert "params_millions" in table.columns


def test_table_two_components_and_interaction() -> None:
    full = _bundle("Full model", seed=5)
    minus = _bundle("minus", seed=6)
    table = table_2_components({"Full model": full, "minus": minus}, full_name="Full model", interaction_ratio=1.23)
    assert len(table.rows) == 2
    assert table.rows[0]["delta_macro_f1_points"] == pytest.approx(0.0)
    missing = table_2_components({"minus": minus})
    assert missing.status == "NOT_RUN"
    ratio = interaction_ratio(full, minus, minus)
    assert np.isfinite(ratio)


def test_table_three_comparators_and_fairness() -> None:
    bundles = {"a": _bundle("a", seed=7), "b": _bundle("b", seed=8)}
    gap = {name: fairness_gap(bundle, tuple(str(site) for site in bundle.sites)) for name, bundle in bundles.items()}
    table = table_3_comparators(bundles, fairness_gap=gap)
    assert table.number == 3
    assert "fairness_gap" in table.columns
    assert all("auc_I" in row for row in table.rows)
    record = confusion_matrix_row(bundles["a"], "a")
    assert record["total"] == len(bundles["a"].limb_ids)
    assert record["off_diagonal"] >= 0


def test_table_four_transport() -> None:
    development = _bundle("proposed", seed=9)
    external = _bundle("proposed", seed=10)
    table = table_4_transport(
        development,
        external,
        {"resistance_ratio": 0.941, "shape_factor": 0.928, "minimum_luminal_area": 0.937},
        {"resistance_ratio": 0.928, "shape_factor": 0.914, "minimum_luminal_area": 0.902},
        calibration=(1.02, 0.96),
    )
    assert table.number == 4
    quantities = {row["quantity"] for row in table.rows}
    assert "Type macro-AUROC" in quantities
    assert "Calibration slope" in quantities
    assert any(row["unit"] == "dimensionless" for row in table.rows)


def test_table_five_morphometry() -> None:
    automated = {"minimum_luminal_area": np.linspace(80.0, 120.0, 40)}
    manual = {"minimum_luminal_area": np.linspace(80.5, 121.0, 40)}
    table = table_5_morphometry(
        automated,
        manual,
        cost={"hydraulic_solve_ms_per_segment": 41.0},
        propagated_error={"minimum_luminal_area": 0.082},
    )
    assert table.number == 5
    assert table.rows[0]["icc"] > 0.9
    assert any(row["descriptor"] == "hydraulic_solve_ms_per_segment" for row in table.rows)
    empty = table_5_morphometry({"orphan": np.zeros(5)}, {})
    assert empty.rows[0]["status"] == "NOT_RUN"


def test_table_six_clinical() -> None:
    retrospective = _bundle("proposed", seed=11)
    table = table_6_clinical(
        retrospective,
        workflow={"Turnaround-time change": -0.042},
        safety={"Pre-specified adverse events": 0.0},
        units={"patients": 120, "limbs": 240},
    )
    assert table.number == 6
    endpoints = {row["endpoint"] for row in table.rows}
    assert "Management-relevant call, sens./spec." in endpoints
    assert "Turnaround-time change" in endpoints


def test_table_seven_association() -> None:
    table = table_7_association(
        {"dvt_history": 0.785},
        {"dvt_history": 0.835},
        {"dvt_history": {"p_value": 0.0005, "ci_low": 0.038, "ci_high": 0.062}},
        {"dvt_history:type_iii_iv": {"odds_ratio": 3.21, "ci_low": 2.68, "ci_high": 3.84, "p_value": 1e-9}},
        {"dvt_history": 0.31},
        0.041,
        units={"patients": 1492, "limbs": 2972},
    )
    assert table.number == 7
    kinds = {row["kind"] for row in table.rows}
    assert kinds == {"discrimination", "adjusted_odds_ratio"}


def test_association_on_a_generated_cohort() -> None:
    cohort = generate_cohort(120, seed=17)
    columns = build_columns(cohort)
    assert columns["stenosis_ratio"].shape[0] == cohort.n_limbs
    scaled = scaled_columns(columns)
    assert abs(float(np.mean(scaled["resistance_ratio_civ"]))) < 1e-9
    labels = endpoint_labels(cohort)
    assert set(labels) == set(PRESENTATION_ENDPOINTS)
    table = run_association(cohort, threshold=0.5)
    assert table.degree_only_auc
    assert table.odds_ratios
    assert table.units["limbs"] == cohort.n_limbs
    for endpoint in table.degree_only_auc:
        if endpoint in table.extended_auc:
            assert 0.0 <= table.degree_only_auc[endpoint] <= 1.0
            assert 0.0 <= table.extended_auc[endpoint] <= 1.0
    record = table.record()
    assert "nri" in record


def test_report_rendering_and_writing(tmp_path) -> None:
    table = table_3_comparators({"a": _bundle("a", seed=13)})
    text = render_table(table)
    assert "Table 3" in text
    assert "note:" in text
    assert format_value(1.5).strip() == "1.5000"
    assert format_value(None).strip() == "NOT_RUN"
    assert format_value(True).strip() == "yes"
    assert format_value(7).strip() == "7"
    assert format_value(float("nan")).strip() == "NOT_RUN"
    sections = [
        summary_section("summary", {"a": 1}),
        matrix_section("confusion", [[1, 0], [0, 1]], ("I", "II")),
        comparison_section(
            "intervals",
            intervals_to_rows(
                [entry.record() for entry in [clustered_metric(_bundle("a"), "macro_auroc", resamples=20, seed=1)]]
            ),
        ),
        ReportSection(title="notes", body="one line"),
    ]
    report = Report(title="run", sections=sections, tables=[table])
    body = report.body()
    assert "run" in body
    path = report.write(tmp_path / "report.txt")
    assert path.read_text(encoding="utf-8").startswith("run")


def test_site_and_development_constants() -> None:
    assert Site.C in {EXTERNAL_SITE} or str(EXTERNAL_SITE).endswith("C")
    assert DEVELOPMENT_SITES == (Site.A, Site.B)
