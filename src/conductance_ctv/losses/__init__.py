"""Loss selection, so an experiment names its objective rather than building it."""

from __future__ import annotations

from torch import nn

from conductance_ctv.losses.class_balanced import (
    ClassBalancedMultinomialLoss,
    GroupFairnessPenalty,
    SeverityRegressionLoss,
    TypeModelLoss,
    class_counts,
    effective_number_weights,
)
from conductance_ctv.losses.segmentation import (
    ConfluenceWeightedLoss,
    DeepSupervisionLoss,
    generalized_dice,
    soft_cldice,
    soft_dice,
    soft_dice_per_channel,
)

SEGMENTATION_LOSSES: tuple[str, ...] = ("confluence_dice_ce", "confluence_dice_ce_cldice", "uniform_dice_ce")


def segmentation_inside_weight(name: str, configured: float = 3.0) -> float:
    """Weight of the cost inside the confluence cylinder for a named objective.

    The uniform-cost ablation of Table 2 is a flat map, which is the same map with
    a weight of one.
    """
    return 1.0 if name == "uniform_dice_ce" else float(configured)


def build_segmentation_loss(
    name: str = "confluence_dice_ce",
    dice_weight: float = 1.0,
    cross_entropy_weight: float = 1.0,
    deep_supervision: bool = False,
    supervision_weights: tuple[float, ...] = (0.4, 0.3, 0.3),
) -> nn.Module:
    """Build the segmentation objective named by an experiment.

    ``uniform_dice_ce`` is the ablation that drops the confluence weighting; the
    cost map itself comes from :func:`segmentation_inside_weight`, because the
    geometry and not the objective decides where the weight is raised.

    Ref: Table 2 (the confluence-weighted loss ablation).
    """
    if name == "confluence_dice_ce":
        base: nn.Module = ConfluenceWeightedLoss(
            dice_weight=dice_weight,
            cross_entropy_weight=cross_entropy_weight,
            use_cldice=False,
        )
    elif name == "confluence_dice_ce_cldice":
        base = ConfluenceWeightedLoss(
            dice_weight=dice_weight,
            cross_entropy_weight=cross_entropy_weight,
            use_cldice=True,
        )
    elif name == "uniform_dice_ce":
        base = ConfluenceWeightedLoss(
            dice_weight=dice_weight,
            cross_entropy_weight=cross_entropy_weight,
            use_cldice=False,
        )
    else:
        raise KeyError(f"unknown segmentation loss {name!r}")
    if deep_supervision:
        return DeepSupervisionLoss(base, supervision_weights)
    return base


def build_type_loss(
    beta: float = 0.999,
    label_smoothing: float = 0.0,
    fairness_weight: float = 0.0,
    severity_weight: float = 0.3,
    classification_weight: float = 1.0,
) -> TypeModelLoss:
    """Build the four-type objective, optionally with the group constraint."""
    return TypeModelLoss(
        classification_weight=classification_weight,
        severity_weight=severity_weight,
        fairness_weight=fairness_weight,
        beta=beta,
        label_smoothing=label_smoothing,
    )


__all__ = [
    "SEGMENTATION_LOSSES",
    "ClassBalancedMultinomialLoss",
    "ConfluenceWeightedLoss",
    "DeepSupervisionLoss",
    "GroupFairnessPenalty",
    "SeverityRegressionLoss",
    "TypeModelLoss",
    "build_segmentation_loss",
    "segmentation_inside_weight",
    "build_type_loss",
    "class_counts",
    "effective_number_weights",
    "generalized_dice",
    "soft_cldice",
    "soft_dice",
    "soft_dice_per_channel",
]
