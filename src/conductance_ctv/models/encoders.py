"""Three-dimensional encoder families for the segmentation and type backbones.

Five families are declared, matching the columns of Table 1: a residual encoder,
a densely connected encoder, an attention encoder, a vision-transformer trunk and
a hierarchical convolutional trunk that plays the role of the foundation
backbone. Each family exposes ``feature_channels``, the width of the tensor the
decoder or the type head consumes.

The manuscript does not state per-layer widths, so each family carries a
``width_mult`` knob and :func:`resolve_width_for_target` derives the multiplier
that lands the built model on a reported parameter total. The measured total is
always reported alongside, never the requested one.

Ref: Table 1 (architecture families and parameter totals), Sec. 6.4 (adapted
backbone), Sec. 6.6 (backbone feature vector).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_MODALITIES = 1


def _round_to(value: float, divisor: int) -> int:
    return max(divisor, int(round(value / divisor)) * divisor)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Total parameter count, in units of one parameter."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def count_parameters_millions(module: nn.Module) -> float:
    return count_parameters(module) / 1e6


def count_parameters_at_width(
    family: str,
    kwargs: Mapping[str, object],
    multiplier: float = 1.0,
) -> float:
    """Parameter total of a family at a width multiplier, without allocating it.

    The width search below evaluates the family at multipliers up to 16, where the
    real tensors would be several hundred gigabytes; a meta-device build carries
    the shapes and dtypes only, so the count is exact and the memory is not.
    """
    arguments: dict[str, object] = dict(kwargs)
    arguments["width_mult"] = multiplier
    with torch.device("meta"):
        return count_parameters_millions(build_encoder(family, **arguments))


def width_factory(family: str, base: Mapping[str, object]) -> Callable[[float], nn.Module]:
    """One-argument builder at a fixed family and base width set.

    Binding both here keeps the builder independent of the loop that produced
    them, so a search that calls it many times cannot read a later iteration.
    """
    frozen: dict[str, object] = dict(base)

    def build(multiplier: float) -> nn.Module:
        arguments = dict(frozen)
        arguments["width_mult"] = multiplier
        return build_encoder(family, **arguments)

    return build


def norm3d(channels: int, max_groups: int = 8) -> nn.Module:
    """Normalisation that tolerates a one-voxel spatial extent.

    A four-times-strided trunk reaches a 1 x 1 x 1 feature map on a small patch,
    and instance normalisation rejects that in training mode. Group normalisation
    with the same parameter count per channel behaves identically on a larger map
    and stays defined at the coarsest stage.
    """
    groups = max_groups if channels % max_groups == 0 else 1
    return nn.GroupNorm(groups, channels, affine=True)


class ConvNormAct(nn.Module):
    """Convolution, normalisation and activation, the unit every encoder repeats."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int, int] = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False
        )
        self.norm = norm3d(out_channels)
        self.activation: nn.Module = nn.LeakyReLU(0.01, inplace=True) if activation else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activated: torch.Tensor = self.activation(self.norm(self.conv(inputs)))
        return activated


class SqueezeExcite3D(nn.Module):
    """Channel attention used by the residual and attention families."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.reduce = nn.Conv3d(channels, hidden, 1)
        self.expand = nn.Conv3d(hidden, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.expand(F.relu(self.reduce(self.pool(inputs)))))
        return inputs * weights


class Bottleneck3D(nn.Module):
    """The residual bottleneck of the ResNet family, in three dimensions."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        width: int,
        stride: int | tuple[int, int, int] = 1,
        cardinality: int = 1,
    ) -> None:
        super().__init__()
        out_channels = width * self.expansion
        self.reduce = ConvNormAct(in_channels, width, kernel_size=1, stride=1)
        self.spatial = nn.Conv3d(width, width, 3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.spatial_norm = norm3d(width)
        self.expand = ConvNormAct(width, out_channels, kernel_size=1, stride=1, activation=False)
        self.attention = SqueezeExcite3D(out_channels)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        out = self.reduce(inputs)
        out = self.activation(self.spatial_norm(self.spatial(out)))
        out = self.attention(self.expand(out))
        combined: torch.Tensor = self.activation(out + residual)
        return combined


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        blocks: int,
        stride: int | tuple[int, int, int],
        cardinality: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [Bottleneck3D(in_channels, width, stride, cardinality)]
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(width * Bottleneck3D.expansion, width, 1, cardinality))
        self.blocks = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        staged: torch.Tensor = self.blocks(inputs)
        return staged


class ResNet3D(nn.Module):
    """Residual encoder with the ResNet-50 stage layout, in three dimensions."""

    def __init__(
        self,
        in_channels: int = SUPPORTED_MODALITIES,
        widths: Sequence[int] = (64, 128, 256, 512),
        blocks: Sequence[int] = (3, 4, 6, 3),
        width_mult: float = 1.0,
        stem_channels: int | None = None,
    ) -> None:
        super().__init__()
        if len(widths) != len(blocks):
            raise ValueError("widths and block counts must have the same length")
        scaled = [_round_to(width * width_mult, 8) for width in widths]
        stem = stem_channels if stem_channels is not None else _round_to(64 * width_mult, 8)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem, kernel_size=3, stride=(1, 2, 2)),
            ConvNormAct(stem, stem, kernel_size=3, stride=1),
        )
        stages: list[nn.Module] = []
        current = stem
        for index, (width, depth) in enumerate(zip(scaled, blocks, strict=True)):
            stride = 1 if index == 0 else (2, 2, 2)
            stages.append(ResidualStage(current, width, depth, stride))
            current = width * Bottleneck3D.expansion
        self.stages = nn.Sequential(*stages)
        self.feature_channels = current
        self.feature_dim = current
        self.ladder = (stem, *[width * Bottleneck3D.expansion for width in scaled])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        deepest: torch.Tensor = self.stages(self.stem(inputs))
        return deepest

    def forward_skips(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Encoder activations from the stem down, coarsest last."""
        out = self.stem(inputs)
        collected = [out]
        for stage in self.stages:
            out = stage(out)
            collected.append(out)
        return collected


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.bottleneck = ConvNormAct(in_channels, 4 * growth_rate, kernel_size=1)
        self.conv = nn.Conv3d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
        self.norm = norm3d(growth_rate)
        self.activation = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.activation(self.norm(self.conv(self.bottleneck(inputs)))))
        return torch.cat([inputs, out], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_channels: int, layers: int, growth_rate: int, dropout: float = 0.0) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = in_channels
        for _ in range(layers):
            blocks.append(DenseLayer3D(current, growth_rate, dropout))
            current += growth_rate
        self.layers = nn.Sequential(*blocks)
        self.out_channels = current

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dense: torch.Tensor = self.layers(inputs)
        return dense


class Transition3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = norm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool3d(2, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compressed: torch.Tensor = self.pool(self.conv(F.leaky_relu(self.norm(inputs), 0.01)))
        return compressed


class DenseNet3D(nn.Module):
    """Densely connected encoder with the DenseNet-121 block layout."""

    def __init__(
        self,
        in_channels: int = SUPPORTED_MODALITIES,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        compression: float = 0.5,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        growth = _round_to(growth_rate * width_mult, 4)
        stem_features = _round_to(init_features * width_mult, 8)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_features, kernel_size=3, stride=(1, 2, 2)),
            ConvNormAct(stem_features, stem_features, kernel_size=3, stride=1),
        )
        blocks: list[nn.Module] = []
        ladder: list[int] = [stem_features]
        current = stem_features
        for index, layers in enumerate(block_config):
            blocks.append(DenseBlock3D(current, layers, growth))
            current += layers * growth
            ladder.append(current)
            if index != len(block_config) - 1:
                out_channels = _round_to(current * compression, 8)
                blocks.append(Transition3D(current, out_channels))
                current = out_channels
                ladder.append(current)
        self.blocks = nn.Sequential(*blocks)
        self.final_norm = norm3d(current)
        self.feature_channels = current
        self.feature_dim = current
        self.ladder = tuple(ladder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.blocks(self.stem(inputs))
        return F.leaky_relu(self.final_norm(out), 0.01)

    def forward_skips(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        collected = [self.stem(inputs)]
        for block in self.blocks:
            collected.append(block(collected[-1]))
        collected[-1] = F.leaky_relu(self.final_norm(collected[-1]), 0.01)
        return collected


class AttentionBlock3D(nn.Module):
    """Channel-and-spatial attention block of the three-dimensional CNN family."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_reduce = nn.Conv3d(channels, max(4, channels // 8), 1)
        self.channel_expand = nn.Conv3d(max(4, channels // 8), channels, 1)
        self.spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([torch.mean(inputs, dim=1, keepdim=True), torch.amax(inputs, dim=1, keepdim=True)], dim=1)
        spatial_weight = torch.sigmoid(self.spatial(pooled))
        pooled_channel = F.adaptive_avg_pool3d(inputs, 1)
        channel_weight = torch.sigmoid(self.channel_expand(F.leaky_relu(self.channel_reduce(pooled_channel), 0.01)))
        return inputs * channel_weight * spatial_weight


class AttentionStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, stride: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for index in range(depth):
            layers.append(ConvNormAct(current, out_channels, kernel_size=3, stride=stride if index == 0 else 1))
            current = out_channels
        layers.append(AttentionBlock3D(out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attended: torch.Tensor = self.layers(inputs)
        return attended


class AttentionCNN3D(nn.Module):
    """Three-dimensional CNN encoder built from attention stages."""

    def __init__(
        self,
        in_channels: int = SUPPORTED_MODALITIES,
        widths: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 3, 2),
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have the same length")
        scaled = [_round_to(width * width_mult, 8) for width in widths]
        stages: list[nn.Module] = []
        current = in_channels
        for index, (width, depth) in enumerate(zip(scaled, depths, strict=True)):
            stages.append(AttentionStage(current, width, depth, stride=1 if index == 0 else 2))
            current = width
        self.stages = nn.Sequential(*stages)
        self.feature_channels = current
        self.feature_dim = current
        self.ladder = tuple(scaled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        staged: torch.Tensor = self.stages(inputs)
        return staged

    def forward_skips(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        collected: list[torch.Tensor] = []
        out = inputs
        for stage in self.stages:
            out = stage(out)
            collected.append(out)
        return collected


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch, stride=patch)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.proj(inputs)
        tokens: torch.Tensor = projected.flatten(2).transpose(1, 2)
        return tokens


class TransformerBlock3D(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(inputs)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        out = inputs + attended
        residual: torch.Tensor = out + self.mlp(self.norm2(out))
        return residual


class ViT3D(nn.Module):
    """Patch transformer trunk, the family used for the universal-model columns."""

    def __init__(
        self,
        in_channels: int = SUPPORTED_MODALITIES,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        patch: tuple[int, int, int] = (4, 8, 8),
        prompt_tokens: int = 0,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        scaled_dim = max(heads, _round_to(dim * width_mult, heads))
        self.dim = scaled_dim
        self.patch = patch
        self.patch_embed = PatchEmbed3D(in_channels, scaled_dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, scaled_dim))
        self.prompt_tokens = prompt_tokens
        self.prompt: nn.Parameter | None = (
            nn.Parameter(torch.zeros(1, prompt_tokens, scaled_dim)) if prompt_tokens > 0 else None
        )
        self.blocks = nn.ModuleList([TransformerBlock3D(scaled_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(scaled_dim)
        self.feature_channels = scaled_dim
        self.feature_dim = scaled_dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.prompt is not None:
            nn.init.trunc_normal_(self.prompt, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs)
        batch = tokens.shape[0]
        prefix = [self.cls_token.expand(batch, -1, -1)]
        if self.prompt is not None:
            prefix.append(self.prompt.expand(batch, -1, -1))
        tokens = torch.cat([*prefix, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        normalised: torch.Tensor = self.norm(tokens)
        return normalised

    def pooled(self, inputs: torch.Tensor) -> torch.Tensor:
        """Class-token embedding, the feature the type head consumes."""
        embedding: torch.Tensor = self.forward(inputs)[:, 0]
        return embedding

    def grid_shape(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Patch grid of an input volume, per axis."""
        return (
            max(1, int(shape[0]) // int(self.patch[0])),
            max(1, int(shape[1]) // int(self.patch[1])),
            max(1, int(shape[2]) // int(self.patch[2])),
        )

    def token_grid(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embedded patches laid back out as a feature map, for the dense head."""
        tokens = self.forward(inputs)[:, 1 + self.prompt_tokens :]
        grid = self.grid_shape((inputs.shape[2], inputs.shape[3], inputs.shape[4]))
        batch = tokens.shape[0]
        laid_out: torch.Tensor = tokens.transpose(1, 2).reshape(batch, self.dim, *grid)
        return laid_out

    def skip_ladder(self) -> tuple[int, ...]:
        return (self.dim,)


class FoundationTrunk3D(nn.Module):
    """Hierarchical convolutional trunk standing in for the pretrained backbone.

    The family is the one the manuscript fine-tunes for the proposed model: a
    stem followed by four residual stages with a projection to a fixed feature
    width, so the decoder and the appearance branch see a single interface.
    """

    def __init__(
        self,
        in_channels: int = SUPPORTED_MODALITIES,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        blocks: Sequence[int] = (2, 2, 4, 2),
        width_mult: float = 1.0,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        if len(blocks) + 1 != len(widths):
            raise ValueError("the trunk needs one more width than it has stages")
        scaled = [_round_to(width * width_mult, 8) for width in widths]
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, scaled[0], kernel_size=3, stride=(1, 2, 2)),
            ConvNormAct(scaled[0], scaled[0], kernel_size=3, stride=1),
        )
        stages: list[nn.Module] = []
        current = scaled[0]
        for index, depth in enumerate(blocks):
            width = scaled[index + 1]
            stages.append(ResidualStage(current, width // Bottleneck3D.expansion, depth, 2))
            current = width
        self.stages = nn.Sequential(*stages)
        self.projection = nn.Conv3d(current, feature_dim, 1)
        self.feature_channels = feature_dim
        self.feature_dim = feature_dim
        self.ladder = (scaled[0], *scaled[1:-1], feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected: torch.Tensor = self.projection(self.stages(self.stem(inputs)))
        return projected

    def forward_skips(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Stem and stage activations, with the deepest one projected.

        The projection is the interface the decoder and the appearance branch
        both consume, so it replaces the raw deepest stage in the ladder.
        """
        out = self.stem(inputs)
        collected = [out]
        for index, stage in enumerate(self.stages):
            out = stage(out)
            collected.append(self.projection(out) if index == len(self.stages) - 1 else out)
        return collected


@dataclass(frozen=True)
class EncoderSpec:
    """A named encoder family with its constructor arguments."""

    name: str
    family: str
    kwargs: dict[str, float | int | tuple[int, ...]]
    reported_params_millions: float


ENCODER_SPECS: tuple[EncoderSpec, ...] = (
    EncoderSpec("resnet50_unet", "resnet", {"widths": (64, 128, 256, 512), "blocks": (3, 4, 6, 3)}, 46.3),
    EncoderSpec("densenet121_unet", "densenet", {"growth_rate": 32, "block_config": (6, 12, 24, 16)}, 28.7),
    EncoderSpec("attention3d_unet", "attention", {"widths": (64, 128, 256, 512), "depths": (3, 3, 4, 3)}, 38.1),
    EncoderSpec(
        "foundation_trunk",
        "foundation",
        {"widths": (64, 128, 256, 512, 1024), "blocks": (2, 2, 4, 2)},
        131.0,
    ),
    EncoderSpec("universal_transferred", "vit", {"dim": 384, "depth": 12, "heads": 6}, 96.4),
    EncoderSpec("universal_prompted", "vit", {"dim": 384, "depth": 12, "heads": 6, "prompt_tokens": 8}, 142.0),
)


def build_encoder(family: str, **kwargs: object) -> nn.Module:
    """Construct an encoder by family name."""
    if family == "resnet":
        return ResNet3D(**kwargs)  # type: ignore[arg-type]
    if family == "densenet":
        return DenseNet3D(**kwargs)  # type: ignore[arg-type]
    if family == "attention":
        return AttentionCNN3D(**kwargs)  # type: ignore[arg-type]
    if family == "foundation":
        return FoundationTrunk3D(**kwargs)  # type: ignore[arg-type]
    if family == "vit":
        return ViT3D(**kwargs)  # type: ignore[arg-type]
    raise KeyError(f"unknown encoder family {family!r}")


def encoder_spec(name: str) -> EncoderSpec:
    for spec in ENCODER_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown encoder {name!r}")


@dataclass(frozen=True)
class WidthResolution:
    """Outcome of matching a family to a reported parameter total.

    The width multiplier drives integer channel counts, so the parameter total is
    a step function of the multiplier rather than a continuous one: most reported
    totals fall strictly between two achievable totals. The record therefore
    carries both neighbours of the target instead of a single number, and the
    residual is reported as the width-ladder granularity rather than presented as
    an exact match.
    """

    multiplier: float
    achieved_millions: float
    target_millions: float
    lower_millions: float
    upper_millions: float

    @property
    def relative_gap(self) -> float:
        return abs(self.achieved_millions - self.target_millions) / self.target_millions

    @property
    def ladder_granularity(self) -> float:
        """Gap between the two achievable totals that bracket the target."""
        return (self.upper_millions - self.lower_millions) / self.target_millions

    def record(self) -> dict[str, float]:
        return {
            "multiplier": self.multiplier,
            "achieved_millions": self.achieved_millions,
            "target_millions": self.target_millions,
            "lower_millions": self.lower_millions,
            "upper_millions": self.upper_millions,
            "relative_gap": self.relative_gap,
            "ladder_granularity": self.ladder_granularity,
        }


def resolve_width_bracket(
    factory: Callable[[float], nn.Module],
    target_millions: float,
    low: float = 0.1,
    high: float = 16.0,
    iterations: int = 64,
) -> WidthResolution:
    """Locate the two achievable totals a reported total falls between.

    The parameter total is measured from a meta-device build at every probe, so
    the returned figures come from the model rather than from the target. Raises
    when the target lies outside the family's reachable range, which is the one
    failure the caller has to answer for.

    Ref: Table 1 parameter column.
    """
    if target_millions <= 0.0:
        raise ValueError("the target must be a positive parameter total")
    if not 0.0 < low < high:
        raise ValueError("the search interval must be positive and ordered")

    cache: dict[float, float] = {}

    def achieved(multiplier: float) -> float:
        key = round(multiplier, 12)
        if key not in cache:
            with torch.device("meta"):
                cache[key] = count_parameters_millions(factory(multiplier))
        return cache[key]

    low_value = achieved(low)
    high_value = achieved(high)
    if not low_value <= target_millions <= high_value:
        raise ValueError(
            f"the target {target_millions:.1f} M lies outside the reachable range [{low_value:.1f}, {high_value:.1f}] M"
        )

    for _ in range(iterations):
        middle = 0.5 * (low + high)
        value = achieved(middle)
        if value < target_millions:
            low = middle
        else:
            high = middle
    lower_total = achieved(low)
    upper_total = achieved(high)
    if abs(lower_total - target_millions) <= abs(upper_total - target_millions):
        multiplier, total = low, lower_total
    else:
        multiplier, total = high, upper_total
    return WidthResolution(
        multiplier=multiplier,
        achieved_millions=total,
        target_millions=target_millions,
        lower_millions=min(lower_total, upper_total),
        upper_millions=max(lower_total, upper_total),
    )


def parameter_table(
    specs: Iterable[EncoderSpec] = ENCODER_SPECS, resolve: bool = False
) -> dict[str, dict[str, object]]:
    """Measured parameter totals for every declared family.

    With ``resolve`` the width multiplier that brackets each reported total is
    searched for as well; that search builds each family repeatedly, so it is
    opt-in rather than the default.
    """
    out: dict[str, dict[str, object]] = {}
    for spec in specs:
        factory = width_factory(spec.family, spec.kwargs)
        measured = count_parameters_millions(factory(1.0))
        entry: dict[str, object] = {
            "reported_millions": spec.reported_params_millions,
            "measured_millions": measured,
        }
        if resolve:
            try:
                resolution = resolve_width_bracket(factory, spec.reported_params_millions)
            except ValueError as error:
                entry["resolution"] = None
                entry["note"] = str(error)
            else:
                entry["resolution"] = resolution.record()
        out[spec.name] = entry
    return out


def receptive_span_voxels(depth: int, stride_pattern: Sequence[int] = (1, 2, 2, 2)) -> int:
    """Number of input voxels a deepest-stage unit sees, for patch sizing."""
    span = 1
    for stride in stride_pattern[:depth]:
        span *= stride
    return span * (2 * depth + 1)


def channels_divisible(channels: int, divisor: int = 8) -> bool:
    return channels % divisor == 0


def stem_reduction(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Feature-map shape after the anisotropic stem, which keeps the first axis."""
    return (shape[0], int(math.ceil(shape[1] / 2)), int(math.ceil(shape[2] / 2)))
