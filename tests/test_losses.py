"""The segmentation objective, the class-balanced objective and the registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conductance_ctv.losses import (
    SEGMENTATION_LOSSES,
    ClassBalancedMultinomialLoss,
    ConfluenceWeightedLoss,
    DeepSupervisionLoss,
    GroupFairnessPenalty,
    SeverityRegressionLoss,
    TypeModelLoss,
    build_segmentation_loss,
    build_type_loss,
    class_counts,
    effective_number_weights,
    generalized_dice,
    segmentation_inside_weight,
    soft_cldice,
    soft_dice,
    soft_dice_per_channel,
)


def _pair(batch: int = 2, shape: tuple[int, int, int] = (8, 16, 16)) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(batch, 2, *shape)
    target = torch.zeros(batch, 2, *shape)
    target[:, 1, 2:6, 6:10, 6:10] = 1.0
    target[:, 0] = 1.0 - target[:, 1]
    return logits, target


def test_soft_dice_is_one_on_a_perfect_prediction() -> None:
    target = torch.zeros(1, 1, 4, 4, 4)
    target[:, :, 1:3, 1:3, 1:3] = 1.0
    assert float(soft_dice(target, target)) == pytest.approx(1.0, abs=1e-4)
    assert float(soft_dice(torch.zeros_like(target), target)) < 1e-4


def test_soft_dice_is_weighted_by_the_cost_map() -> None:
    probabilities = torch.full((1, 1, 4, 4, 4), 0.5)
    target = torch.zeros_like(probabilities)
    target[:, :, :2] = 1.0
    weight = torch.ones_like(probabilities)
    weight[:, :, :2] = 5.0
    assert float(soft_dice(probabilities, target, weight)) != float(soft_dice(probabilities, target))


def test_soft_dice_per_channel_and_generalized() -> None:
    logits, target = _pair()
    probabilities = torch.softmax(logits, dim=1)
    per_channel = soft_dice_per_channel(probabilities, target)
    assert per_channel.shape == (2, 2)
    value = generalized_dice(probabilities, target)
    assert 0.0 <= float(value) <= 1.0


def test_soft_cldice_is_bounded_and_differentiable() -> None:
    logits, target = _pair()
    logits.requires_grad_(True)
    probabilities = torch.sigmoid(logits[:, 1:2])
    score = soft_cldice(probabilities, target[:, 1:2], iterations=3)
    assert 0.0 <= float(score.detach()) <= 1.0
    score.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(torch.sum(torch.abs(logits.grad))) >= 0.0
    with pytest.raises(ValueError):
        soft_cldice(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))


def test_confluence_loss_reports_its_parts() -> None:
    logits, target = _pair()
    criterion = ConfluenceWeightedLoss(use_cldice=True)
    parts = criterion(logits, target)
    assert set(parts) == {"dice_loss", "cross_entropy", "cldice_loss", "total"}
    assert torch.isfinite(parts["total"])
    assert float(parts["dice_loss"]) >= 0.0
    with pytest.raises(ValueError):
        criterion(torch.zeros(2, 4, 4, 4), target)


def test_confluence_loss_backward_produces_finite_gradients() -> None:
    logits, target = _pair()
    logits.requires_grad_(True)
    criterion = ConfluenceWeightedLoss()
    weight = torch.ones(2, 1, 8, 16, 16)
    weight[:, :, :3] = 3.0
    parts = criterion(logits, target, weight)
    parts["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(torch.sum(torch.abs(logits.grad))) > 0.0


def test_deep_supervision_sums_the_stages() -> None:
    target = torch.zeros(1, 2, 16, 16, 16)
    target[:, 1, 4:8, 4:8, 4:8] = 1.0
    target[:, 0] = 1.0 - target[:, 1]
    stages = [torch.randn(1, 2, size, size, size) for size in (16, 8, 4)]
    criterion = DeepSupervisionLoss(ConfluenceWeightedLoss(), weights=(0.5, 0.3, 0.2))
    parts = criterion(stages, target)
    assert "total" in parts
    assert "stage0_total" in parts
    with pytest.raises(ValueError):
        DeepSupervisionLoss(ConfluenceWeightedLoss(), weights=(0.5, 0.5))(stages, target)


def test_effective_number_weights_match_the_closed_form() -> None:
    counts = [900.0, 600.0, 300.0, 100.0]
    beta = 0.999
    weights = effective_number_weights(counts, beta)
    closed = np.array([(1.0 - beta) / (1.0 - beta**value) for value in counts])
    closed = closed * (4.0 / closed.sum())
    assert np.allclose(weights.numpy(), closed, atol=1e-5)
    assert float(weights.min()) > 0.0
    with pytest.raises(ValueError):
        effective_number_weights(counts, beta=1.5)
    with pytest.raises(ValueError):
        effective_number_weights(np.ones((2, 2)))


def test_effective_number_weights_skip_absent_classes() -> None:
    weights = effective_number_weights([10.0, 0.0, 5.0, 1.0], 0.999)
    assert float(weights[1]) == 0.0


def test_class_counts_and_the_weighted_loss() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 3])
    counts = class_counts(labels, 4)
    assert counts.tolist() == [3.0, 2.0, 1.0, 1.0]
    criterion = ClassBalancedMultinomialLoss()
    criterion.fit_weights(labels)
    assert float(criterion.class_weights[0]) < float(criterion.class_weights[3])
    logits = torch.randn(4, 4)
    loss = criterion(logits, torch.tensor([0, 1, 2, 3]))
    assert torch.isfinite(loss)
    assert len(criterion.record()["class_weights"]) == 4


def test_severity_regression_loss_is_finite_on_an_occluded_target() -> None:
    criterion = SeverityRegressionLoss()
    prediction = torch.randn(5)
    target = torch.tensor([1.0, 2.0, float("inf"), 1.5, 3.0])
    censored = torch.tensor([False, False, True, False, False])
    loss = criterion(prediction, target, censored)
    assert torch.isfinite(loss)
    assert float(criterion(prediction, target)) > 0.0


def test_group_fairness_penalty_is_zero_for_one_group() -> None:
    penalty = GroupFairnessPenalty(0.5)
    probabilities = torch.softmax(torch.randn(6, 4), dim=1)
    single = torch.zeros(6, dtype=torch.long)
    assert float(penalty(probabilities, single)) == 0.0
    two_groups = torch.tensor([0, 0, 0, 1, 1, 1])
    assert float(penalty(probabilities, two_groups)) >= 0.0


def test_type_model_loss_combines_the_terms() -> None:
    torch.manual_seed(0)
    criterion = TypeModelLoss(fairness_weight=0.2)
    logits = torch.randn(8, 4)
    probabilities = torch.softmax(logits, dim=1)
    labels = torch.randint(0, 4, (8,))
    severity_prediction = torch.randn(8)
    severity_target = torch.rand(8) + 1.0
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    parts = criterion(logits, probabilities, severity_prediction, labels, severity_target, groups)
    assert set(parts) == {"classification", "severity", "fairness", "total"}
    assert torch.isfinite(parts["total"])
    weights = criterion.fit_class_weights(labels)
    assert weights.shape == (4,)


def test_loss_registry_builds_every_named_objective() -> None:
    for name in SEGMENTATION_LOSSES:
        objective = build_segmentation_loss(name, deep_supervision=name == "confluence_dice_ce")
        assert isinstance(objective, torch.nn.Module)
    with pytest.raises(KeyError):
        build_segmentation_loss("nonsense")
    type_objective = build_type_loss(fairness_weight=0.1)
    assert isinstance(type_objective, TypeModelLoss)


def test_the_uniform_ablation_asks_for_a_flat_cost_map() -> None:
    assert segmentation_inside_weight("confluence_dice_ce", 3.0) == 3.0
    assert segmentation_inside_weight("uniform_dice_ce", 3.0) == 1.0
    logits, target = _pair()
    weight = torch.ones(2, 1, 8, 16, 16)
    weight[:, :, :4] = 4.0
    weighted = build_segmentation_loss("confluence_dice_ce")(logits, target, weight)
    flat = build_segmentation_loss("uniform_dice_ce")(logits, target, torch.ones_like(weight))
    assert float(weighted["cross_entropy"]) != float(flat["cross_entropy"])
    assert float(weighted["cross_entropy"]) > float(flat["cross_entropy"])
