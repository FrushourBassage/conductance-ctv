"""Anisotropic decoder.

A venous-phase acquisition is far coarser through the plane than inside it, so
the decoder restores in-plane resolution at every stage while leaving the
through-plane axis at the resolution it already has. The final upsampling step is
correspondingly anisotropic; an isotropic decoder would have to invent
through-plane detail that the acquisition never measured.

Ref: Sec. 6.4 ("the usage of an anisotropic decoder in place of the final
upsampling stage ... because the acquisition is anisotropic in the through-plane
direction").
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from conductance_ctv.models.encoders import ConvNormAct

IN_PLANE_SCALE: tuple[int, int, int] = (1, 2, 2)


def anisotropic_upsample(inputs: torch.Tensor, scale: tuple[int, int, int] = IN_PLANE_SCALE) -> torch.Tensor:
    """Upsample in-plane only, keeping the through-plane resolution.

    The last stage of a decoder needs a factor of two in every axis to undo a
    factor-of-two stride; restricting the factor to the two in-plane axes is what
    makes the decoder anisotropic.
    """
    return F.interpolate(inputs, scale_factor=scale, mode="trilinear", align_corners=False)


class DecoderStage(nn.Module):
    """One decoder stage: anisotropic upsample, skip concatenation, two convolutions."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale: tuple[int, int, int] = IN_PLANE_SCALE,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.skip_channels = int(skip_channels)
        self.block = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvNormAct(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        """Upsample, concatenate the skip and convolve.

        A missing skip is substituted by zeros of the declared width so the same
        weights serve both the skip-connected path and the degenerate path used
        when a decoder is driven without an encoder ladder.
        """
        upsampled = anisotropic_upsample(inputs, self.scale)
        if skip is None:
            skip = torch.zeros(
                upsampled.shape[0],
                self.skip_channels,
                *upsampled.shape[2:],
                dtype=upsampled.dtype,
                device=upsampled.device,
            )
        elif upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="trilinear", align_corners=False)
        convolved: torch.Tensor = self.block(torch.cat([upsampled, skip], dim=1))
        return convolved


class AnisotropicDecoder3D(nn.Module):
    """Decoder that walks an encoder's channel ladder in reverse and upsamples in plane.

    ``skip_channels`` is the encoder channel list from the stem to the deepest
    stage. The decoder walks it backwards, so the adapter between an encoder
    family and this decoder is a single list of integers.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        feature_channels: int,
        decoder_widths: Sequence[int] | None = None,
        out_channels: int = 2,
        deep_supervision: bool = False,
        decoder_width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        ladder = list(encoder_channels)
        if not ladder:
            raise ValueError("a decoder needs at least one encoder channel count")
        if decoder_width_mult <= 0.0:
            raise ValueError("the decoder width multiplier must be positive")
        base = (
            list(decoder_widths)
            if decoder_widths is not None
            else [max(16, min(256, value // 2)) for value in reversed(ladder)]
        )
        # The decoder widths are not stated by the manuscript, so they are a
        # declared default; the multiplier is the free variable that brings the
        # full network onto its reported parameter total.
        widths = [max(8, int(round(value * decoder_width_mult / 8.0)) * 8) for value in base]
        if len(widths) != len(ladder):
            raise ValueError("the decoder width ladder must match the encoder ladder")
        self.bottleneck = ConvNormAct(feature_channels, widths[0], kernel_size=3)
        stages: list[nn.Module] = []
        current = widths[0]
        # The deepest encoder activation feeds the bottleneck, so the stages consume
        # the skips from the second-deepest outwards: ladder[-2], ladder[-3], ...,
        # ladder[0]. Pairing them from the deepest one instead would shift every
        # stage by one level of the ladder.
        skip_ladder = list(reversed(ladder[:-1]))
        for index, skip in enumerate(skip_ladder):
            width = widths[index + 1]
            stages.append(DecoderStage(current, skip, width, scale=IN_PLANE_SCALE))
            current = width
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv3d(current, out_channels, kernel_size=1)
        self.deep_supervision = deep_supervision
        self.auxiliary: nn.ModuleList | None = (
            nn.ModuleList([nn.Conv3d(width, out_channels, 1) for width in widths[1:]]) if deep_supervision else None
        )
        self.ladder = ladder
        self.decoder_widths = widths
        self.skip_ladder = skip_ladder
        self.decoder_width_mult = float(decoder_width_mult)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.bottleneck(features)
        for stage in self.stages:
            out = stage(out, None)
        logits: torch.Tensor = self.head(out)
        return logits

    def forward_with_skips(self, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        """Full decoder path with the encoder skip connections."""
        if not skips:
            raise ValueError("at least one skip tensor is required")
        out = self.bottleneck(skips[-1])
        for stage, skip in zip(self.stages, reversed(list(skips[:-1])), strict=True):
            out = stage(out, skip)
        logits: torch.Tensor = self.head(out)
        return logits

    def forward_multiscale(self, skips: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Deep-supervision outputs, coarsest last."""
        out = self.bottleneck(skips[-1])
        outputs: list[torch.Tensor] = []
        heads = self.auxiliary
        for position, (stage, skip) in enumerate(zip(self.stages, reversed(list(skips[:-1])), strict=True)):
            out = stage(out, skip)
            if heads is not None and position < len(heads):
                aux: torch.Tensor = heads[position](out)
                outputs.append(aux)
        final: torch.Tensor = self.head(out)
        outputs.append(final)
        return outputs
