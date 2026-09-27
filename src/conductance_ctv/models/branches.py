"""The two branches of the type model and their fusion.

The descriptor branch reads the hydraulic magnitudes and is meant to be read
literally; the appearance branch reads texture the geometry does not carry. Both
are fused and trained jointly under the class-balanced multinomial loss, and the
severity regression is produced by the descriptor branch alone so that the
reported ``rho_R`` never depends on appearance.

Ref: Sec. 6.6 ("both branches are fused together ... the descriptor branch
conveys physical magnitudes and is understood literally; the appearance branch
conveys an internal texture not captured by geometry"), Algorithm 2 lines 1-3.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from conductance_ctv.hydraulics.descriptors import DESCRIPTOR_FIELDS, SEGMENT_NAMES

DESCRIPTOR_DIM = len(SEGMENT_NAMES) * len(DESCRIPTOR_FIELDS)
SEVERITY_INDEX = SEGMENT_NAMES.index("CIV") * len(DESCRIPTOR_FIELDS) + DESCRIPTOR_FIELDS.index("resistance_ratio")


def descriptor_dimension() -> int:
    """Length of the concatenated descriptor vector, seven fields per segment."""
    return DESCRIPTOR_DIM


class DescriptorBranch(nn.Module):
    """Per-segment encoding of the hydraulic descriptor vector, then its aggregate.

    The vector is reshaped to one row per segment so the branch can separate
    within-segment magnitudes from between-segment relations, which is what the
    reference-resistance ratio already encodes.

    Ref: Sec. 6.5 descriptor list and Sec. 6.6.
    """

    def __init__(self, hidden: int = 128, embedding: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.segments = len(SEGMENT_NAMES)
        self.fields = len(DESCRIPTOR_FIELDS)
        self.embedding = embedding
        self.normalise = nn.LayerNorm(self.fields)
        self.within = nn.Sequential(
            nn.Linear(self.fields, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding),
            nn.GELU(),
        )
        self.across = nn.Sequential(
            nn.Linear(self.segments * embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding),
            nn.GELU(),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.shape[-1] != DESCRIPTOR_DIM:
            raise ValueError(f"expected {DESCRIPTOR_DIM} descriptor columns, got {descriptors.shape[-1]}")
        rows = descriptors.reshape(descriptors.shape[0], self.segments, self.fields)
        encoded = self.within(self.normalise(rows))
        aggregated: torch.Tensor = self.across(encoded.reshape(descriptors.shape[0], -1))
        return aggregated


class AppearanceBranch(nn.Module):
    """Projection of the pooled appearance feature into the fusion width."""

    def __init__(self, input_dim: int, hidden: int = 128, embedding: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected: torch.Tensor = self.project(features)
        return projected


class FusionHead(nn.Module):
    """Concatenate the two branch embeddings, then classify and regress."""

    temperature: torch.Tensor

    def __init__(
        self,
        descriptor_embedding: int,
        appearance_embedding: int,
        n_classes: int = 4,
        hidden: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        combined = descriptor_embedding + appearance_embedding
        self.trunk = nn.Sequential(
            nn.LayerNorm(combined),
            nn.Linear(combined, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, n_classes)
        self.severity = nn.Linear(hidden, 1)
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(
        self, descriptor_embedding: torch.Tensor, appearance_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        fused = self.trunk(torch.cat([descriptor_embedding, appearance_embedding], dim=1))
        logits: torch.Tensor = self.classifier(fused)
        severity: torch.Tensor = self.severity(fused).squeeze(-1)
        return {"logits": logits, "severity_raw": severity}

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled logits, used before any probability is reported."""
        scaled: torch.Tensor = logits / self.temperature.clamp_min(1e-6)
        return scaled


class DescriptorOnlySeverity(nn.Module):
    """Severity regression from the descriptor branch, with no appearance input.

    Eq. (2) of Algorithm 2 reports the severity from the descriptor branch alone
    so the number stays interpretable and traceable to a measurement.
    """

    def __init__(self, descriptor_embedding: int, hidden: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(descriptor_embedding),
            nn.Linear(descriptor_embedding, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, descriptor_embedding: torch.Tensor) -> torch.Tensor:
        severity: torch.Tensor = self.head(descriptor_embedding).squeeze(-1)
        return severity


def branch_shapes(descriptor_dim: int = DESCRIPTOR_DIM, appearance_dim: int = 512) -> dict[str, tuple[int, ...]]:
    return {"descriptor": (descriptor_dim,), "appearance": (appearance_dim,)}


def descriptor_slice(segment: str, field: str) -> int:
    """Column index of one descriptor field of one segment."""
    if segment not in SEGMENT_NAMES:
        raise KeyError(f"unknown segment {segment!r}")
    if field not in DESCRIPTOR_FIELDS:
        raise KeyError(f"unknown descriptor field {field!r}")
    return SEGMENT_NAMES.index(segment) * len(DESCRIPTOR_FIELDS) + DESCRIPTOR_FIELDS.index(field)


def names_for(indices: Sequence[int]) -> list[str]:
    """Human-readable names of descriptor columns, for the report writers."""
    return [
        f"{SEGMENT_NAMES[index // len(DESCRIPTOR_FIELDS)]}_{DESCRIPTOR_FIELDS[index % len(DESCRIPTOR_FIELDS)]}"
        for index in indices
    ]
