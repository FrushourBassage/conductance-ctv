"""Class-balanced multinomial loss for the four-type head.

The four types are strongly unbalanced and the rarest one carries the decision,
so the multinomial loss is re-weighted by the effective number of samples of each
class rather than by its raw frequency. A severity regression on the
dimensionless resistance ratio is trained alongside it in log space, because the
ratio is positive and heavy-tailed.

Ref: Sec. 6.6 ("class-balanced multinomial loss"), Table 3 (the rarest and
hardest type carries the decision).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def effective_number_weights(counts: Sequence[float], beta: float = 0.999, normalise: bool = True) -> torch.Tensor:
    """Inverse effective-number weights of the class-balanced loss.

    ``w_y = (1 - beta) / (1 - beta ** n_y)`` with ``n_y`` the number of training
    samples of class ``y``; a class with no sample receives weight zero.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly inside (0, 1)")
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    weights = np.zeros_like(values)
    present = values > 0
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, values[present]))
    if normalise and weights.sum() > 0.0:
        weights = weights * (values.shape[0] / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def class_counts(labels: Sequence[int] | torch.Tensor, n_classes: int = 4) -> np.ndarray:
    values = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    return np.bincount(values.astype(np.int64), minlength=n_classes)[:n_classes].astype(np.float64)


class ClassBalancedMultinomialLoss(nn.Module):
    """Weighted cross-entropy with optional label smoothing.

    Ref: Sec. 6.6.
    """

    class_weights: torch.Tensor

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        beta: float = 0.999,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if class_weights is None:
            self.register_buffer("class_weights", torch.ones(4, dtype=torch.float32))
        else:
            self.register_buffer("class_weights", class_weights.to(dtype=torch.float32))

    def fit_weights(self, labels: Sequence[int] | torch.Tensor, n_classes: int = 4) -> torch.Tensor:
        counts = [float(value) for value in class_counts(labels, n_classes)]
        weights = effective_number_weights(counts, self.beta)
        self.class_weights.copy_(weights)
        return weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights: torch.Tensor = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss: torch.Tensor = F.cross_entropy(
            logits,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        return loss

    def record(self) -> dict[str, list[float]]:
        return {"class_weights": [float(value) for value in self.class_weights]}


class SeverityRegressionLoss(nn.Module):
    """Smooth L1 on the log ratio, which is the scale the descriptor lives on.

    The dimensionless resistance ratio is one for a healthy span and grows with
    the hydraulic burden, so its logarithm is the quantity with a symmetric error
    structure. A floor keeps the logarithm finite for an occluded segment, whose
    resistance is infinite and whose severity is reported as a censored value.

    Ref: Sec. 6.5 (rho_R is dimensionless and at least one) and Algorithm 2 line 3.
    """

    def __init__(self, beta: float = 0.5, floor: float = 1.0, censored_weight: float = 0.25) -> None:
        super().__init__()
        self.beta = float(beta)
        self.floor = float(floor)
        self.censored_weight = float(censored_weight)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, censored: torch.Tensor | None = None
    ) -> torch.Tensor:
        clean = torch.nan_to_num(target, nan=self.floor, posinf=self.floor * 1e6, neginf=self.floor)
        logged_target = torch.log(torch.clamp(clean, min=self.floor))
        logged_prediction = torch.log(torch.clamp(F.softplus(prediction) + self.floor, min=self.floor))
        elementwise = F.smooth_l1_loss(logged_prediction, logged_target, beta=self.beta, reduction="none")
        if censored is None:
            return elementwise.mean()
        weights = torch.where(censored, self.censored_weight, torch.ones_like(elementwise))
        return (elementwise * weights).sum() / weights.sum().clamp_min(1e-6)


class GroupFairnessPenalty(nn.Module):
    """Penalty on the spread of the severe-call rate across pre-specified groups.

    The constrained variant of Table 3 trades primary-utility performance for a
    smaller fairness gap, so the constraint has to be visible in the objective and
    not only at the decision rule.

    Ref: Table 3 (fairness-debiased row) and Supplementary Table S18.
    """

    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, probabilities: torch.Tensor, groups: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        severe = probabilities[:, 2] + probabilities[:, 3]
        calls = (severe >= threshold).to(dtype=probabilities.dtype)
        unique = torch.unique(groups)
        if unique.numel() <= 1:
            return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
        rates = torch.stack([torch.mean(calls[groups == group]) for group in unique])
        return self.weight * torch.var(rates)


class TypeModelLoss(nn.Module):
    """Joint objective: classification, severity and the optional group penalty."""

    def __init__(
        self,
        classification_weight: float = 1.0,
        severity_weight: float = 0.3,
        fairness_weight: float = 0.0,
        beta: float = 0.999,
        label_smoothing: float = 0.0,
        severity_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.classification = ClassBalancedMultinomialLoss(beta=beta, label_smoothing=label_smoothing)
        self.severity = SeverityRegressionLoss(beta=severity_beta)
        self.fairness = GroupFairnessPenalty(fairness_weight)
        self.classification_weight = float(classification_weight)
        self.severity_weight = float(severity_weight)
        self.fairness_weight = float(fairness_weight)

    def forward(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        severity_prediction: torch.Tensor,
        labels: torch.Tensor,
        severity_target: torch.Tensor,
        groups: torch.Tensor | None = None,
        censored: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        classification = self.classification(logits, labels)
        severity = self.severity(severity_prediction, severity_target, censored)
        total = self.classification_weight * classification + self.severity_weight * severity
        parts = {"classification": classification, "severity": severity}
        if groups is not None and self.fairness_weight > 0.0:
            penalty = self.fairness(probabilities, groups, threshold)
            total = total + penalty
            parts["fairness"] = penalty
        parts["total"] = total
        return parts

    def fit_class_weights(self, labels: Sequence[int] | torch.Tensor) -> torch.Tensor:
        return self.classification.fit_weights(labels)
