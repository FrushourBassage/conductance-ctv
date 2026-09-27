"""The conductance type model, its calibration and its pre-specified decision rule.

The model fuses the descriptor branch with the appearance branch, is trained
under the class-balanced multinomial loss, and reports a calibrated probability
for each of the four types together with a severity score. The decision rule
turns the calibrated probabilities into one clinical recommendation through a
threshold that is fixed before the sealed region is opened, and the threshold is
applied to the sum of the post-thrombotic and the occlusive probability.

Ref: Sec. 6.6 and Algorithm 2; Sec. 6.7 (the pre-specified criterion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from conductance_ctv.models.branches import (
    AppearanceBranch,
    DescriptorBranch,
    DescriptorOnlySeverity,
    FusionHead,
)
from conductance_ctv.models.unet3d import LumenSegmentationNet, appearance_descriptor
from conductance_ctv.utils.types import FloatArray, IntArray

TYPE_III_INDEX = 2
TYPE_IV_INDEX = 3


@dataclass(frozen=True)
class TypeModelOutput:
    """Everything Algorithm 2 returns for a batch."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    severity_raw: torch.Tensor
    severity_descriptor: torch.Tensor
    predicted: torch.Tensor


class ConductanceTypeModel(nn.Module):
    """Segmentation backbone, descriptor branch, appearance branch and fusion head."""

    def __init__(
        self,
        segmenter: LumenSegmentationNet,
        descriptor_hidden: int = 128,
        descriptor_embedding: int = 96,
        appearance_embedding: int = 96,
        fusion_hidden: int = 128,
        dropout: float = 0.1,
        n_classes: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.descriptor_branch = DescriptorBranch(
            hidden=descriptor_hidden, embedding=descriptor_embedding, dropout=dropout
        )
        backbone_channels = int(getattr(segmenter.encoder, "feature_channels", 512))
        self.appearance_branch = AppearanceBranch(
            input_dim=backbone_channels, embedding=appearance_embedding, dropout=dropout
        )
        self.fusion = FusionHead(
            descriptor_embedding=descriptor_embedding,
            appearance_embedding=appearance_embedding,
            n_classes=n_classes,
            hidden=fusion_hidden,
            dropout=dropout,
        )
        self.severity = DescriptorOnlySeverity(descriptor_embedding)
        self.n_classes = n_classes
        if freeze_backbone:
            for parameter in self.segmenter.encoder.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        image: torch.Tensor,
        descriptors: torch.Tensor,
        tube_mask: torch.Tensor | None = None,
    ) -> TypeModelOutput:
        appearance = appearance_descriptor(self.segmenter.encoder, image, tube_mask)
        descriptor_embedding = self.descriptor_branch(descriptors)
        appearance_embedding = self.appearance_branch(appearance)
        fused = self.fusion(descriptor_embedding, appearance_embedding)
        scaled = self.fusion.temperature_scale(fused["logits"])
        probabilities = torch.softmax(scaled, dim=1)
        return TypeModelOutput(
            logits=fused["logits"],
            probabilities=probabilities,
            severity_raw=fused["severity_raw"],
            severity_descriptor=self.severity(descriptor_embedding),
            predicted=torch.argmax(probabilities, dim=1),
        )

    def segment(self, image: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.segmenter(image)
        return logits

    def trainable_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict[str, object]]:
        """Discriminative learning rates: a lower one for the adapted backbone."""
        backbone_ids = {id(parameter) for parameter in self.segmenter.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        heads = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone, "lr": backbone_lr, "name": "backbone"},
            {"params": heads, "lr": head_lr, "name": "heads"},
        ]

    def freeze_segmentation(self, freeze: bool = True) -> None:
        for parameter in self.segmenter.parameters():
            parameter.requires_grad = not freeze


class TemperatureCalibrator:
    """Single-parameter temperature fitted on the tuning partition only.

    Ref: Sec. 6.6 ("temperature-scaled calibration in the tuning partition only").
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: FloatArray, labels: IntArray, iterations: int = 200) -> float:
        """Grid-then-refine search on the negative log-likelihood, which is one-dimensional."""
        values = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError("logits must have shape (N, C)")
        if values.shape[0] != targets.shape[0]:
            raise ValueError("logits and labels must agree on the row count")
        rows = np.arange(targets.shape[0])
        left, right = 1e-2, 1e2
        for _ in range(iterations):
            third = (right - left) / 3.0
            first = left + third
            second = right - third
            if self._nll(values, targets, rows, first) <= self._nll(values, targets, rows, second):
                right = second
            else:
                left = first
        self.temperature = float(max(1e-6, 0.5 * (left + right)))
        return self.temperature

    @staticmethod
    def _nll(values: FloatArray, targets: IntArray, rows: IntArray, temperature: float) -> float:
        scaled = values / max(temperature, 1e-6)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-np.mean(log_prob[rows, targets]))

    def apply(self, logits: FloatArray) -> FloatArray:
        return np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-6)

    def probabilities(self, logits: FloatArray) -> FloatArray:
        scaled = self.apply(logits)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        normalised: FloatArray = exponent / exponent.sum(axis=1, keepdims=True)
        return normalised

    def record(self) -> dict[str, float]:
        return {"temperature": self.temperature}


def apply_temperature_(module: ConductanceTypeModel, temperature: float) -> None:
    """Write a fitted temperature into the fusion head's buffer."""
    with torch.no_grad():
        module.fusion.temperature.fill_(float(temperature))


def decision_rule(probabilities: FloatArray | torch.Tensor, threshold: float) -> np.ndarray:
    """``y = 1[p_III + p_IV >= theta]``.

    Ref: Algorithm 2 line 4 and the pre-specified operating point of Sec. 6.7.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("the threshold must lie in [0, 1]")
    values = (
        probabilities.detach().cpu().numpy() if isinstance(probabilities, torch.Tensor) else np.asarray(probabilities)
    )
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("probabilities must have shape (N, 4)")
    severe = values[:, TYPE_III_INDEX] + values[:, TYPE_IV_INDEX]
    return (severe >= float(threshold)).astype(np.int64)


def operating_point(
    probabilities: FloatArray, reference: FloatArray, target_sensitivity: float | None = None
) -> dict[str, float]:
    """Threshold, sensitivity and specificity of a candidate operating point.

    The threshold is chosen on the tuning partition to reach a target sensitivity
    on the management-relevant call; the returned operating point is what the
    pre-specified success criterion is then evaluated against.

    The sweep is a prefix sum over the severity order, so every achievable
    operating point is enumerated rather than sampled on a grid.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(reference, dtype=np.int64)
    severe = values[:, TYPE_III_INDEX] + values[:, TYPE_IV_INDEX]
    truth = (labels >= 3).astype(np.float64)
    order = np.argsort(-severe, kind="mergesort")
    ordered_severe = severe[order]
    ordered_truth = truth[order]
    total_positive = float(ordered_truth.sum())
    total_negative = float(ordered_truth.shape[0] - total_positive)
    if total_positive <= 0.0 or total_negative <= 0.0:
        raise ValueError("the operating point is undefined without both classes present")
    cumulative_positive = np.cumsum(ordered_truth)
    cumulative_negative = np.cumsum(1.0 - ordered_truth)
    sensitivity = cumulative_positive / total_positive
    specificity = 1.0 - cumulative_negative / total_negative
    target = float(np.mean(truth)) if target_sensitivity is None else float(target_sensitivity)
    index = int(np.argmin(np.abs(sensitivity - target)))
    threshold = float(ordered_severe[index])
    if index + 1 < ordered_severe.shape[0]:
        threshold = 0.5 * (float(ordered_severe[index]) + float(ordered_severe[index + 1]))
    return {
        "threshold": threshold,
        "sensitivity": float(sensitivity[index]),
        "specificity": float(specificity[index]),
        "youden": float(sensitivity[index] + specificity[index] - 1.0),
        "index": float(index),
    }


def threshold_sensitivity_band(
    probabilities: FloatArray, reference: FloatArray, centre: float, width: float = 0.10
) -> dict[str, float]:
    """Spread of sensitivity and specificity inside a threshold band.

    The manuscript reports that inside a band of plus or minus 0.10 the
    sensitivity moves by at most 0.028, the specificity by at most 0.032 and the
    Youden index stays constant; this function computes exactly those three
    spreads for whatever probabilities it is given.

    Ref: Sec. 6.6 and Supplementary Table S23.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(reference, dtype=np.int64)
    severe = values[:, TYPE_III_INDEX] + values[:, TYPE_IV_INDEX]
    truth = (labels >= 3).astype(np.int64)
    sensitivities: list[float] = []
    specificities: list[float] = []
    youdens: list[float] = []
    for offset in (-width, -0.5 * width, 0.0, 0.5 * width, width):
        threshold = float(np.clip(centre + offset, 0.0, 1.0))
        calls = (severe >= threshold).astype(np.int64)
        tp = float(np.sum(calls * truth))
        fp = float(np.sum(calls * (1 - truth)))
        fn = float(np.sum((1 - calls) * truth))
        tn = float(np.sum((1 - calls) * (1 - truth)))
        sensitivity = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        youdens.append(sensitivity + specificity - 1.0)
    return {
        "sensitivity_range": float(max(sensitivities) - min(sensitivities)),
        "specificity_range": float(max(specificities) - min(specificities)),
        "youden_range": float(max(youdens) - min(youdens)),
    }


def group_thresholds(
    probabilities: FloatArray,
    reference: FloatArray,
    groups: np.ndarray,
    global_threshold: float,
    tolerance: float = 0.35,
) -> dict[str, float]:
    """Per-group thresholds under an equalised-call constraint.

    The fairness-debiased variant of Table 3 is a constrained model: its gain is
    confined to the fairness-gap column, so the constraint has to be applied at
    the decision rule and not only in the loss. Each group's threshold stays
    inside a tolerance band around the global one and is nudged to equalise the
    severe-call rate across groups.

    Ref: Table 3 (fairness-debiased row) and Supplementary Table S18.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    severe = values[:, TYPE_III_INDEX] + values[:, TYPE_IV_INDEX]
    unique = np.unique(groups)
    unconditional = float(np.mean((severe >= global_threshold).astype(np.float64)))
    lower = max(0.0, global_threshold * (1.0 - tolerance))
    upper = min(1.0, global_threshold * (1.0 + tolerance) + tolerance)
    out: dict[str, float] = {}
    for group in unique:
        mask = groups == group
        if not bool(mask.any()):
            continue
        best_threshold = global_threshold
        best_gap = np.inf
        for threshold in np.linspace(lower, upper, 21):
            rate = float(np.mean((severe[mask] >= threshold).astype(np.float64)))
            if abs(rate - unconditional) < best_gap:
                best_gap = abs(rate - unconditional)
                best_threshold = float(threshold)
        out[str(group)] = best_threshold
    return out
