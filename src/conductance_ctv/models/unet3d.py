"""Lumen segmentation network: an encoder family plus its decoder.

The proposed segmenter pairs the foundation trunk with the anisotropic decoder.
The comparator columns of Table 1 pair the same decoder with the residual, the
densely connected and the attention encoders, and pair the transformer trunk with
a patch-grid upsampling head, so every column differs in the encoder rather than
in the objective.

Ref: Sec. 6.4; Table 1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from conductance_ctv.models.decoder import IN_PLANE_SCALE, AnisotropicDecoder3D, anisotropic_upsample
from conductance_ctv.models.encoders import (
    AttentionCNN3D,
    DenseNet3D,
    FoundationTrunk3D,
    ResNet3D,
    ViT3D,
    build_encoder,
    count_parameters_millions,
    encoder_spec,
    norm3d,
)

LUMEN_CLASSES = 2

SEGMENTATION_ENCODERS: tuple[str, ...] = (
    "resnet50_unet",
    "densenet121_unet",
    "attention3d_unet",
    "foundation_trunk",
    "universal_transferred",
    "universal_prompted",
)

ConvolutionalEncoder = ResNet3D | DenseNet3D | AttentionCNN3D | FoundationTrunk3D


class PatchGridHead(nn.Module):
    """Segmentation head for a patch transformer: grid upsampling back to voxels."""

    def __init__(self, dim: int, out_channels: int, patch: tuple[int, int, int], stages: int = 2) -> None:
        super().__init__()
        self.patch = patch
        widths = [max(32, dim // (2**index)) for index in range(1, stages + 1)]
        blocks: list[nn.Module] = []
        current = dim
        for width in widths:
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(current, width, 3, padding=1, bias=False),
                    norm3d(width),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            )
            current = width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv3d(current, out_channels, 1)
        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Upsample(scale_factor=IN_PLANE_SCALE, mode="trilinear", align_corners=False),
                    nn.Conv3d(current, current, 3, padding=1, bias=False),
                    norm3d(current),
                    nn.LeakyReLU(0.01, inplace=True),
                )
                for _ in range(2)
            ]
        )

    def forward(self, grid: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        out = grid
        for block in self.blocks:
            out = block(out)
        out = self.upsample(out)
        if tuple(out.shape[2:]) != tuple(target_shape):
            out = F.interpolate(out, size=tuple(target_shape), mode="trilinear", align_corners=False)
        grid_logits: torch.Tensor = self.head(out)
        return grid_logits


class LumenSegmentationNet(nn.Module):
    """Encoder plus decoder producing per-voxel lumen logits."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        patch_head: PatchGridHead | None = None,
        encoder_name: str = "custom",
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_head = patch_head
        self.encoder_name = encoder_name
        self.deep_supervision = deep_supervision

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        if isinstance(encoder, ViT3D):
            head = self.patch_head
            if head is None:
                raise RuntimeError("a patch transformer needs a patch head")
            logits: torch.Tensor = head(encoder.token_grid(inputs), inputs.shape[2:])
            return logits
        if isinstance(encoder, (ResNet3D, DenseNet3D, AttentionCNN3D, FoundationTrunk3D)):
            return self._forward_from_pyramid(encoder, inputs)
        raise TypeError(f"unknown encoder type {type(encoder).__name__}")

    def _forward_from_pyramid(self, encoder: ConvolutionalEncoder, inputs: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder
        if not isinstance(decoder, AnisotropicDecoder3D):
            raise RuntimeError("the network has neither a decoder nor a patch head")
        skips = encoder.forward_skips(inputs)
        if self.deep_supervision:
            coarse = decoder.forward_multiscale(skips)[-1]
            if tuple(coarse.shape[2:]) != tuple(inputs.shape[2:]):
                coarse = F.interpolate(coarse, size=inputs.shape[2:], mode="trilinear", align_corners=False)
            return coarse
        logits: torch.Tensor = decoder.forward_with_skips(skips)
        if tuple(logits.shape[2:]) != tuple(inputs.shape[2:]):
            logits = F.interpolate(logits, size=inputs.shape[2:], mode="trilinear", align_corners=False)
        return logits

    def multiscale(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Deep-supervision logits at every decoder stage, coarsest last."""
        encoder = self.encoder
        decoder = self.decoder
        if not isinstance(encoder, (ResNet3D, DenseNet3D, AttentionCNN3D, FoundationTrunk3D)) or not isinstance(
            decoder, AnisotropicDecoder3D
        ):
            return [self.forward(inputs)]
        skips = encoder.forward_skips(inputs)
        outputs = decoder.forward_multiscale(skips)
        return [F.interpolate(value, size=inputs.shape[2:], mode="trilinear", align_corners=False) for value in outputs]


@dataclass(frozen=True)
class SegmentationBuild:
    """A built segmenter with the parameter count actually measured."""

    network: LumenSegmentationNet
    encoder_name: str
    parameters_millions: float


def build_segmentation_network(
    encoder_name: str,
    out_channels: int = LUMEN_CLASSES,
    width_mult: float = 1.0,
    deep_supervision: bool = False,
    decoder_width_mult: float = 1.0,
) -> SegmentationBuild:
    """Pair an encoder family with its decoder."""
    spec = encoder_spec(encoder_name)
    kwargs = dict(spec.kwargs)
    kwargs["width_mult"] = width_mult
    encoder = build_encoder(spec.family, **kwargs)
    if isinstance(encoder, ViT3D):
        head = PatchGridHead(encoder.dim, out_channels, encoder.patch)
        network = LumenSegmentationNet(encoder=encoder, decoder=None, patch_head=head, encoder_name=encoder_name)
        return SegmentationBuild(network, encoder_name, count_parameters_millions(network))
    if not isinstance(encoder, (ResNet3D, DenseNet3D, AttentionCNN3D, FoundationTrunk3D)):
        raise TypeError(f"unknown encoder type {type(encoder).__name__}")
    ladder = tuple(int(value) for value in encoder.ladder)
    decoder = AnisotropicDecoder3D(
        encoder_channels=ladder,
        feature_channels=int(encoder.feature_channels),
        out_channels=out_channels,
        deep_supervision=deep_supervision,
        decoder_width_mult=decoder_width_mult,
    )
    network = LumenSegmentationNet(
        encoder=encoder,
        decoder=decoder,
        patch_head=None,
        encoder_name=encoder_name,
        deep_supervision=deep_supervision,
    )
    return SegmentationBuild(network, encoder_name, count_parameters_millions(network))


def encoder_from_name(encoder_name: str, width_mult: float = 1.0) -> nn.Module:
    spec = encoder_spec(encoder_name)
    kwargs = dict(spec.kwargs)
    kwargs["width_mult"] = width_mult
    return build_encoder(spec.family, **kwargs)


def decoder_width_factory(
    encoder_name: str,
    width_mult: float,
    out_channels: int = LUMEN_CLASSES,
    deep_supervision: bool = False,
) -> Callable[[float], nn.Module]:
    """One-argument builder of the full network at a fixed encoder width."""

    def build(decoder_width_mult: float) -> nn.Module:
        return build_segmentation_network(
            encoder_name,
            out_channels=out_channels,
            width_mult=width_mult,
            deep_supervision=deep_supervision,
            decoder_width_mult=decoder_width_mult,
        ).network

    return build


def segmentation_total_millions(
    encoder_name: str,
    width_mult: float = 1.0,
    decoder_width_mult: float = 1.0,
    out_channels: int = LUMEN_CLASSES,
) -> float:
    """Parameter total of the whole segmentation network, encoder and decoder."""
    with torch.device("meta"):
        return build_segmentation_network(
            encoder_name,
            out_channels=out_channels,
            width_mult=width_mult,
            decoder_width_mult=decoder_width_mult,
        ).parameters_millions


def feature_map(encoder: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Deepest feature map of any encoder family, shaped as a tensor."""
    if isinstance(encoder, ViT3D):
        return encoder.token_grid(inputs)
    if isinstance(encoder, (ResNet3D, DenseNet3D, AttentionCNN3D, FoundationTrunk3D)):
        activated: torch.Tensor = encoder(inputs)
        return activated
    raise TypeError(f"unknown encoder type {type(encoder).__name__}")


def appearance_descriptor(
    encoder: nn.Module, inputs: torch.Tensor, tube_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Pooled appearance vector gathered inside the tube, one row per limb.

    When a tube mask is supplied the pooling is restricted to it, which is the
    "feature vector gathered in a 20 mm tube with a focus on s*" of Sec. 6.6;
    without one the whole patch is pooled.

    Ref: Sec. 6.6.
    """
    features = feature_map(encoder, inputs)
    if tube_mask is None:
        whole: torch.Tensor = F.adaptive_avg_pool3d(features, 1).flatten(1)
        return whole
    if tube_mask.dim() == 4:
        tube_mask = tube_mask.unsqueeze(1)
    mask = tube_mask.to(dtype=features.dtype)
    if mask.shape[2:] != features.shape[2:]:
        mask = F.interpolate(mask, size=features.shape[2:], mode="nearest")
    denominator = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    summed = (features * mask).sum(dim=(2, 3, 4))
    pooled = summed / denominator
    fallback = F.adaptive_avg_pool3d(features, 1).flatten(1)
    empty = (denominator.flatten() < 1.0).unsqueeze(1)
    gathered: torch.Tensor = torch.where(empty, fallback, pooled)
    return gathered


def upsampling_ratio(encoder: nn.Module, reference_input: torch.Tensor) -> float:
    """Voxel-to-feature ratio of an encoder, used to size the decoder output."""
    features = feature_map(encoder, reference_input)
    return float(reference_input.shape[2]) / float(features.shape[2])


def anisotropic_upsample_to(inputs: torch.Tensor, target: Sequence[int]) -> torch.Tensor:
    """Anisotropic upsample followed by a shape clamp, for the report writers."""
    upsampled: torch.Tensor = anisotropic_upsample(inputs)
    if tuple(upsampled.shape[2:]) != tuple(target):
        upsampled = F.interpolate(upsampled, size=tuple(target), mode="trilinear", align_corners=False)
    return upsampled
