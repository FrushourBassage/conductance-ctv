"""Overlap losses for the lumen segmentation.

The segmentation objective is a soft Dice term plus a cross-entropy term on a
weight map that raises the cost inside the confluence cylinder, where the lumen
collapses and where a uniform voxel cost under-weights the region that carries
the minimum area.

Ref: Sec. 6.4 ("adjusting the error function to allow placement of the data
around the critical iliocaval confluence area in a 10 mm cylinder instead of
employing the original error function"); Table 2 (the confluence-weighted loss
ablation).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

_SMOOTH = 1e-6


def soft_dice(
    probabilities: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None, smooth: float = _SMOOTH
) -> torch.Tensor:
    """Soft Dice over the foreground channel, optionally voxel-weighted."""
    if weight is None:
        numerator = 2.0 * torch.sum(probabilities * targets) + smooth
        denominator = torch.sum(probabilities) + torch.sum(targets) + smooth
        return numerator / denominator
    weighted = weight
    numerator = 2.0 * torch.sum(weighted * probabilities * targets) + smooth
    denominator = torch.sum(weighted * probabilities) + torch.sum(weighted * targets) + smooth
    return numerator / denominator


def soft_dice_per_channel(probabilities: torch.Tensor, targets: torch.Tensor, smooth: float = _SMOOTH) -> torch.Tensor:
    """Dice of each channel of a one-hot pair."""
    dims = tuple(range(2, probabilities.dim()))
    numerator = 2.0 * torch.sum(probabilities * targets, dim=dims) + smooth
    denominator = torch.sum(probabilities, dim=dims) + torch.sum(targets, dim=dims) + smooth
    return numerator / denominator


def generalized_dice(probabilities: torch.Tensor, targets: torch.Tensor, smooth: float = _SMOOTH) -> torch.Tensor:
    """Class-weighted Dice, with weights proportional to the inverse class volume."""
    dims = tuple(range(2, probabilities.dim()))
    volumes = torch.sum(targets, dim=dims)
    weights = 1.0 / (volumes**2 + smooth)
    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    numerator = 2.0 * torch.sum(weights * torch.sum(probabilities * targets, dim=dims), dim=1)
    denominator = torch.sum(weights * torch.sum(probabilities + targets, dim=dims), dim=1) + smooth
    return torch.mean(numerator / denominator)


def soft_cldice(
    probabilities: torch.Tensor, targets: torch.Tensor, iterations: int = 8, smooth: float = _SMOOTH
) -> torch.Tensor:
    """Differentiable centreline Dice, using soft skeletonisation.

    The soft skeleton is the iterative difference between a morphological opening
    and the mask itself, which is the differentiable surrogate of the true
    skeleton; it is used in the objective while the reported metric uses an exact
    integer skeleton.

    Ref: Table 1 (clDice column), Sec. 2.3 (centreline overlap for thin vessels).
    """
    if probabilities.dim() != 5:
        raise ValueError("soft_cldice expects a five-dimensional tensor")
    kernel = 3
    opened = -F.max_pool3d(-probabilities, kernel, stride=1, padding=kernel // 2)
    opened = F.max_pool3d(opened, kernel, stride=1, padding=kernel // 2)
    skeleton = torch.clamp(probabilities - opened, min=0.0)
    for _ in range(iterations):
        dilated = F.max_pool3d(skeleton, kernel, stride=1, padding=kernel // 2)
        eroded = -F.max_pool3d(-dilated, kernel, stride=1, padding=kernel // 2)
        skeleton = torch.clamp(probabilities * eroded, min=0.0)
    topology_precision = torch.sum(skeleton * targets) / (torch.sum(skeleton) + smooth)
    topology_sensitivity = torch.sum(skeleton * probabilities) / (torch.sum(skeleton) + smooth)
    return 2.0 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity + smooth)


class ConfluenceWeightedLoss(nn.Module):
    """Dice plus cross-entropy on the confluence-weighted cost map.

    The weighting itself is geometric and therefore belongs to the dataset, which
    builds the cost map from the centreline; this objective consumes whatever map
    it is handed, so the same code path serves the confluence-weighted cost and
    the uniform-cost ablation.

    Ref: Sec. 6.4; Table 1 row for the proposed model.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        cross_entropy_weight: float = 1.0,
        use_cldice: bool = False,
        cldice_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.use_cldice = bool(use_cldice)
        self.cldice_weight = float(cldice_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if logits.shape[1] < 2:
            raise ValueError("the segmentation loss expects a background and a lumen channel")
        if logits.shape != targets.shape:
            raise ValueError("the logits and the one-hot target must share a shape")
        if weight is not None and tuple(weight.shape) != (logits.shape[0], 1, *logits.shape[2:]):
            raise ValueError("the cost map must have one channel and match the batch and spatial shape")
        probabilities = torch.softmax(logits, dim=1)
        lumen = probabilities[:, 1:2]
        foreground = targets[:, 1:2] if targets.shape[1] > 1 else targets
        dice = 1.0 - soft_dice(lumen, foreground, weight)
        if weight is None:
            cross_entropy = F.binary_cross_entropy(lumen.clamp(1e-6, 1 - 1e-6), foreground, reduction="mean")
        else:
            cross_entropy = F.binary_cross_entropy(
                lumen.clamp(1e-6, 1 - 1e-6), foreground, weight=weight, reduction="mean"
            )
        total = self.dice_weight * dice + self.cross_entropy_weight * cross_entropy
        parts = {"dice_loss": dice, "cross_entropy": cross_entropy}
        if self.use_cldice:
            centreline = 1.0 - soft_cldice(lumen, foreground)
            total = total + self.cldice_weight * centreline
            parts["cldice_loss"] = centreline
        parts["total"] = total
        return parts


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of the per-stage losses of a multi-scale decoder."""

    def __init__(self, base: nn.Module, weights: Sequence[float] = (0.4, 0.3, 0.3)) -> None:
        super().__init__()
        self.base = base
        self.weights = tuple(float(value) for value in weights)

    def forward(
        self,
        logits: Sequence[torch.Tensor],
        targets: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(logits) != len(self.weights):
            raise ValueError("the number of stages and supervision weights must agree")
        total = torch.zeros((), dtype=logits[0].dtype, device=logits[0].device)
        parts: dict[str, torch.Tensor] = {}
        for position, (stage_logits, stage_weight) in enumerate(zip(logits, self.weights, strict=True)):
            scaled = F.interpolate(stage_logits, size=targets.shape[2:], mode="trilinear", align_corners=False)
            stage_parts = self.base(scaled, targets, weight)
            total = total + stage_weight * stage_parts["total"]
            parts[f"stage{position}_total"] = stage_parts["total"]
        parts["total"] = total
        return parts
