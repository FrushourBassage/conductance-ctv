"""Segmentation metrics, discrimination metrics and agreement metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conductance_ctv.metrics.agreement import (
    agreement_table,
    bland_altman,
    concordance_correlation,
    family_of,
    icc_2_1,
    relative_degradation,
)
from conductance_ctv.metrics.classification import (
    auroc,
    balanced_accuracy,
    brier_score,
    calibration_slope,
    cohen_kappa,
    confusion_table,
    expected_calibration_error,
    f1_per_class,
    high_risk_flag,
    kappa_interval,
    kappa_standard_error,
    macro_auroc,
    macro_f1,
    per_type_auroc,
    per_type_scores,
    sensitivity_specificity,
)
from conductance_ctv.metrics.segmentation import (
    aggregate_segment_scores,
    anatomical_gradient,
    centreline_dice,
    dice_score,
    distance_to_surface,
    hausdorff_distance,
    jaccard_score,
    score_volume,
    surface_dice,
    surface_distances,
)
from conductance_ctv.metrics.skeleton import (
    distance_transform,
    radial_thickness,
    ridge_skeleton,
    skeleton_voxels,
)


def _tube() -> np.ndarray:
    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[4:20, 10:14, 10:14] = True
    return mask


def test_dice_and_jaccard_on_known_masks() -> None:
    first = np.zeros((4, 4, 4), dtype=bool)
    first[:2] = True
    second = np.zeros((4, 4, 4), dtype=bool)
    second[1:3] = True
    assert dice_score(first, second) == pytest.approx(0.5)
    assert jaccard_score(first, second) == pytest.approx(1.0 / 3.0)
    assert dice_score(first, first) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        dice_score(first, np.zeros((5, 5, 5), dtype=bool))


def test_hausdorff_distance_of_a_shifted_tube() -> None:
    first = _tube()
    second = np.roll(first, 2, axis=0)
    distance = hausdorff_distance(first, second, (1.0, 1.0, 1.0))
    assert distance == pytest.approx(2.0, abs=0.05)
    values = surface_distances(first, second, (1.0, 1.0, 1.0))
    assert values.size > 0
    assert float(np.max(values)) >= distance - 1e-9


def test_hausdorff_of_an_empty_mask_is_undefined() -> None:
    empty = np.zeros((6, 6, 6), dtype=bool)
    assert math.isnan(hausdorff_distance(empty, _tube()[:6, :6, :6], (1.0, 1.0, 1.0)))
    assert surface_distances(empty, empty, (1.0, 1.0, 1.0)).size == 0


def test_surface_dice_is_one_for_identical_masks() -> None:
    mask = _tube()
    assert surface_dice(mask, mask, (1.0, 1.0, 1.0), tolerance=0.5) == pytest.approx(1.0)
    empty = np.zeros_like(mask)
    assert math.isnan(surface_dice(empty, mask, (1.0, 1.0, 1.0), tolerance=1.0))


def test_distance_transform_is_zero_outside_and_positive_inside() -> None:
    mask = _tube()
    field = distance_transform(mask, (1.0, 1.0, 1.0))
    assert float(field[~mask].max()) == 0.0
    assert float(field[mask].max()) == pytest.approx(2.0, abs=0.05)
    assert radial_thickness(mask, (1.0, 1.0, 1.0)) == pytest.approx(4.0, abs=0.2)
    surface = distance_to_surface(mask, (1.0, 1.0, 1.0))
    # Inside the trunk the nearest surface voxel is at most the inscribed radius
    # away, and the deepest voxels are one step from it.
    assert float(surface[mask].max()) <= 2.0 + 1e-9
    assert float(surface[field >= 1.99].min()) <= 1.0
    assert float(surface[~mask].max()) > 2.0


def test_distance_transform_of_an_empty_mask_is_zero() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert float(distance_transform(empty).max()) == 0.0


def test_ridge_skeleton_follows_a_trunk_rather_than_filling_it() -> None:
    mask = _tube()
    skeleton = ridge_skeleton(mask, (1.0, 1.0, 1.0))
    assert skeleton_voxels(skeleton) > 0
    assert skeleton_voxels(skeleton) < int(mask.sum())
    assert skeleton_voxels(skeleton & ~mask) == 0
    # A wider trunk of the same length has a broader plateau, but the ridge still
    # scales with the length and not with the volume.
    wider = np.zeros((24, 24, 24), dtype=bool)
    wider[4:20, 9:15, 9:15] = True
    wider_skeleton = ridge_skeleton(wider, (1.0, 1.0, 1.0))
    assert int(wider.sum()) > 2 * int(mask.sum())
    assert skeleton_voxels(wider_skeleton) < 2 * skeleton_voxels(skeleton)
    # The ridge spans the trunk along its axis rather than a slice of it; the two
    # end slices drop out because the strict comparison needs a neighbour on both
    # sides of an equal value.
    span = int(np.ptp(np.nonzero(skeleton)[0])) + 1
    assert 12 <= span <= 16


def test_ridge_skeleton_of_an_empty_mask_is_empty() -> None:
    empty = np.zeros((6, 6, 6), dtype=bool)
    assert skeleton_voxels(ridge_skeleton(empty)) == 0


def test_centreline_dice_exceeds_dice_on_a_trunk() -> None:
    reference = np.zeros((20, 20, 20), dtype=bool)
    reference[3:17, 9:11, 9:11] = True
    predicted = np.zeros_like(reference)
    predicted[2:18, 8:12, 8:12] = True
    assert centreline_dice(predicted, reference, (1.0, 1.0, 1.0)) > dice_score(predicted, reference)
    assert math.isnan(centreline_dice(np.zeros_like(reference), reference, (1.0, 1.0, 1.0)))


def test_score_volume_per_segment_and_gradient() -> None:
    mask = _tube()
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[:8] = 0
    labels[8:16] = 1
    labels[16:] = 2
    report = score_volume(mask, mask, (1.0, 1.0, 1.0), labels)
    assert set(report.dice) == {"CIV", "EIV", "CFV"}
    assert all(value == pytest.approx(1.0) for value in report.dice.values())
    assert report.macro_dice() == pytest.approx(1.0)
    assert len(report.records()) == 9
    gradient = anatomical_gradient(report)
    assert gradient["span"] == pytest.approx(0.0, abs=1e-9)
    whole = score_volume(mask, mask, (1.0, 1.0, 1.0))
    assert "all" in whole.dice


def test_aggregate_segment_scores_drops_undefined_values() -> None:
    mask = _tube()
    report = score_volume(mask, mask, (1.0, 1.0, 1.0))
    values = aggregate_segment_scores([report, report], "all", "dice")
    assert values.shape == (2,)
    with pytest.raises(KeyError):
        aggregate_segment_scores([report], "all", "nonsense")


def test_auroc_matches_the_rank_statistic() -> None:
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assert auroc(scores, labels) == pytest.approx(1.0)
    assert auroc(-scores, labels) == pytest.approx(0.0)
    assert auroc(np.array([1.0, 1.0, 1.0, 1.0]), labels) == pytest.approx(0.5)
    assert math.isnan(auroc(scores, np.ones(4, dtype=np.int64)))
    with pytest.raises(ValueError):
        auroc(scores, np.ones(3, dtype=np.int64))


def test_per_type_and_macro_auroc() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(1, 5, size=200)
    probabilities = np.eye(4)[labels - 1] * 0.4 + rng.random((200, 4)) * 0.6
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    per_type = per_type_auroc(probabilities, labels)
    assert per_type.shape == (4,)
    assert np.all(per_type > 0.5)
    assert macro_auroc(probabilities, labels) == pytest.approx(float(np.mean(per_type)))
    assert per_type_scores(probabilities).shape == (200, 4)
    with pytest.raises(ValueError):
        per_type_scores(np.zeros((10, 3)))


def test_macro_f1_and_balanced_accuracy() -> None:
    labels = np.array([1, 1, 2, 3, 4], dtype=np.int64)
    predictions = np.array([1, 2, 2, 3, 4], dtype=np.int64)
    per_class = f1_per_class(predictions, labels)
    assert per_class.shape == (4,)
    assert macro_f1(predictions, labels) == pytest.approx(float(np.mean(per_class)))
    assert balanced_accuracy(predictions, labels) == pytest.approx(0.875)
    assert balanced_accuracy(predictions, predictions) == pytest.approx(1.0)


def test_cohen_kappa_and_its_interval() -> None:
    labels = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int64)
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)
    assert cohen_kappa(np.array([1, 1, 1, 1, 2, 2, 2, 2]), labels) < 1.0
    error = kappa_standard_error(labels, labels)
    assert error == pytest.approx(0.0, abs=1e-6) or math.isnan(error)
    low, high = kappa_interval(labels, labels)
    assert low <= 1.0 <= high or math.isnan(low)


def test_confusion_table_rows() -> None:
    labels = np.array([1, 2, 3, 4], dtype=np.int64)
    rows = confusion_table(labels, labels)
    assert len(rows) == 16
    assert sum(int(row["count"]) for row in rows) == 4


def test_sensitivity_specificity_and_high_risk_flag() -> None:
    calls = np.array([1, 1, 0, 0], dtype=np.int64)
    truth = np.array([1, 0, 1, 0], dtype=np.int64)
    sensitivity, specificity = sensitivity_specificity(calls, truth)
    assert sensitivity == pytest.approx(0.5)
    assert specificity == pytest.approx(0.5)
    probabilities = np.array([[0.1, 0.1, 0.4, 0.4], [0.7, 0.1, 0.1, 0.1]])
    assert high_risk_flag(probabilities, 0.5).tolist() == [1, 0]


def test_calibration_slope_behaves_as_expected() -> None:
    rng = np.random.default_rng(5)
    count = 4000
    risk = rng.random(count) * 0.8 + 0.1
    labels = np.where(rng.random(count) < risk, 3, 1)
    honest = np.stack([(1 - risk) / 2, (1 - risk) / 2, risk / 3, risk * 2 / 3], axis=1)
    assert calibration_slope(honest, labels) == pytest.approx(1.0, abs=0.15)
    inflated = np.clip(risk * 1.8 - 0.3, 1e-4, 1 - 1e-4)
    over = np.stack([(1 - inflated) / 2, (1 - inflated) / 2, inflated / 3, inflated * 2 / 3], axis=1)
    assert calibration_slope(over, labels) < 0.75


def test_brier_score_and_expected_calibration_error() -> None:
    labels = np.array([1, 2, 3, 4], dtype=np.int64)
    perfect = np.eye(4)[labels - 1]
    assert brier_score(perfect, labels) == pytest.approx(0.0)
    assert expected_calibration_error(perfect, labels) == pytest.approx(0.0)
    uniform = np.full((4, 4), 0.25)
    assert brier_score(uniform, labels) > 0.0
    # A uniform predictor at 0.25 top-label confidence is right one time in four,
    # so its top-label calibration error is zero rather than half.
    assert expected_calibration_error(uniform, labels) == pytest.approx(0.0, abs=1e-9)


def test_icc_matches_the_mean_square_formula() -> None:
    ratings = np.array(
        [[10.0, 11.0], [12.0, 12.5], [9.0, 8.5], [14.0, 15.0], [11.0, 10.0], [13.0, 13.5]],
        dtype=np.float64,
    )
    result = icc_2_1(ratings)
    n_subjects, n_raters = ratings.shape
    grand = float(ratings.mean())
    row_means = ratings.mean(axis=1)
    column_means = ratings.mean(axis=0)
    between_subjects = float(n_raters * np.sum((row_means - grand) ** 2) / (n_subjects - 1))
    between_raters = float(n_subjects * np.sum((column_means - grand) ** 2) / (n_raters - 1))
    total = float(np.sum((ratings - grand) ** 2))
    residual = float(
        (total - n_raters * np.sum((row_means - grand) ** 2) - n_subjects * np.sum((column_means - grand) ** 2))
        / ((n_subjects - 1) * (n_raters - 1))
    )
    hand = (between_subjects - residual) / (
        between_subjects + (n_raters - 1) * residual + n_raters * (between_raters - residual) / n_subjects
    )
    assert result.value == pytest.approx(hand, rel=1e-12)
    assert result.subjects == n_subjects and result.raters == n_raters
    assert len(result.record()) == 6
    with pytest.raises(ValueError):
        icc_2_1(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        icc_2_1(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_icc_is_one_for_identical_raters() -> None:
    values = np.arange(20, dtype=np.float64)
    result = icc_2_1(np.stack([values, values], axis=1))
    assert result.value == pytest.approx(1.0)


def test_bland_altman_limits_match_the_closed_form() -> None:
    automated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    manual = automated + np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    result = bland_altman(automated, manual)
    difference = automated - manual
    assert result.bias == pytest.approx(float(difference.mean()))
    assert result.standard_deviation == pytest.approx(float(difference.std(ddof=1)))
    assert result.upper == pytest.approx(result.bias + 1.959963984540054 * result.standard_deviation)
    assert result.crosses_zero
    assert int(result.record()["samples"]) == 5


def test_bland_altman_handles_an_empty_series() -> None:
    result = bland_altman(np.zeros(0), np.zeros(0))
    assert math.isnan(result.bias)
    assert not result.crosses_zero


def test_concordance_correlation_and_agreement_table() -> None:
    values = np.arange(10, dtype=np.float64)
    assert concordance_correlation(np.stack([values, values], axis=1)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        concordance_correlation(np.stack([values, values, values], axis=1))
    table = agreement_table({"area": values}, {"area": values + 0.1})
    assert "area" in table
    assert table["area"]["icc"] > 0.9
    assert agreement_table({"area": values}, {"other": values}) == {}


def test_relative_degradation_and_family() -> None:
    assert relative_degradation(1.0, 0.9) == pytest.approx(0.1)
    assert math.isnan(relative_degradation(0.0, 0.5))
    assert family_of("resistance_ratio") == "dimensionless"
    assert family_of("shape_factor") == "dimensionless"
    assert family_of("minimum_luminal_area") == "absolute"
