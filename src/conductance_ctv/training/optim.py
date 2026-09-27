"""Optimisers, learning-rate schedules, mixed precision, EMA and checkpoints."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def build_optimizer(
    parameters: Iterable[nn.Parameter] | list[dict[str, object]],
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """Build an optimiser by name."""
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise KeyError(f"unknown optimiser {name!r}")


class WarmupCosineSchedule:
    """Linear warmup followed by a cosine decay, with a floor on the learning rate.

    Both branches are implemented here rather than delegated to a library
    schedule so the warmup step count and the floor are explicit, and so
    :meth:`get_lr` is defined for every step index.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        start_factor: float = 0.1,
    ) -> None:
        if total_steps < 1:
            raise ValueError("the total step count must be at least one")
        if warmup_steps < 0:
            raise ValueError("the warmup step count must be non-negative")
        if not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError("the learning-rate floor must lie in (0, 1]")
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.start_factor = float(start_factor)
        self._step = 0

    @property
    def current_step(self) -> int:
        return self._step

    def get_lr(self, step: int | None = None) -> float:
        index = self._step if step is None else int(step)
        if self.warmup_steps > 0 and index < self.warmup_steps:
            ramp = (1.0 - self.start_factor) * index / float(self.warmup_steps) + self.start_factor
            return self.base_lr * ramp
        progress = (index - self.warmup_steps) / float(max(self.total_steps - self.warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def apply(self, optimizer: torch.optim.Optimizer) -> float:
        value = self.get_lr()
        for group in optimizer.param_groups:
            if "lr_scale" in group:
                group["lr"] = value * float(group["lr_scale"])
            else:
                group["lr"] = value
        return value

    def step(self) -> float:
        self._step += 1
        return self.get_lr()

    def state_dict(self) -> dict[str, float]:
        return {
            "base_lr": self.base_lr,
            "warmup_steps": float(self.warmup_steps),
            "total_steps": float(self.total_steps),
            "min_lr_ratio": self.min_lr_ratio,
            "step": float(self._step),
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._step = int(state.get("step", 0))


class PolynomialSchedule:
    """Polynomial decay with an exponent, the alternative to the cosine branch."""

    def __init__(self, base_lr: float, total_steps: int, power: float = 0.9, min_lr_ratio: float = 0.0) -> None:
        if total_steps < 1:
            raise ValueError("the total step count must be at least one")
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.power = float(power)
        self.min_lr_ratio = float(min_lr_ratio)
        self._step = 0

    def get_lr(self, step: int | None = None) -> float:
        index = self._step if step is None else int(step)
        progress = min(max(index / float(self.total_steps), 0.0), 1.0)
        factor = (1.0 - progress) ** self.power
        value: float = self.base_lr * max(factor, self.min_lr_ratio)
        return value

    def apply(self, optimizer: torch.optim.Optimizer) -> float:
        value = self.get_lr()
        for group in optimizer.param_groups:
            group["lr"] = value * float(group.get("lr_scale", 1.0))
        return value

    def step(self) -> float:
        self._step += 1
        return self.get_lr()


@dataclass(frozen=True)
class PrecisionContext:
    """Mixed-precision configuration, resolved from the experiment config."""

    enabled: bool
    dtype: torch.dtype
    scaler: Any | None

    @classmethod
    def build(cls, precision: str) -> PrecisionContext:
        if precision in ("fp32", "float32"):
            return cls(False, torch.float32, None)
        if precision in ("bf16", "bfloat16"):
            return cls(True, torch.bfloat16, None)
        if precision in ("fp16", "float16"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # torch.amp.GradScaler is absent from this build's stubs.
            scaler: Any = torch.amp.GradScaler(device, enabled=torch.cuda.is_available())  # type: ignore[attr-defined]
            return cls(True, torch.float16, scaler)
        raise KeyError(f"unknown precision {precision!r}")

    def autocast(self) -> torch.autocast:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.autocast(device_type=device, dtype=self.dtype, enabled=self.enabled)

    def record(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "dtype": str(self.dtype).replace("torch.", ""),
            "scaler": self.scaler is not None,
        }


class ModelEMA:
    """Exponential moving average of the parameters, with a warmup-corrected decay."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 100) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("the decay must lie strictly inside (0, 1)")
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.step_count = 0
        self.shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        decay = self.decay * (1.0 - math.exp(-self.step_count / max(self.warmup, 1)))
        decay = min(self.decay, decay)
        for name, tensor in model.state_dict().items():
            stored = self.shadow.get(name)
            if stored is None or not stored.is_floating_point():
                self.shadow[name] = tensor.detach().clone()
                continue
            stored.mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)

    def store(self, model: nn.Module) -> None:
        self.backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "step": self.step_count, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        decay = state.get("decay")
        if isinstance(decay, (int, float)):
            self.decay = float(decay)
        steps = state.get("step")
        if isinstance(steps, int):
            self.step_count = steps
        shadow = state.get("shadow")
        if isinstance(shadow, dict):
            self.shadow = {str(key): value for key, value in shadow.items()}


def gradient_norm(parameters: Iterable[nn.Parameter], norm_type: float = 2.0) -> float:
    """Total gradient norm over a parameter iterable, for the clipping report."""
    norms = [float(torch.linalg.vector_norm(p.grad.detach(), norm_type)) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    if norm_type == float("inf"):
        return max(norms)
    return float(sum(value**norm_type for value in norms) ** (1.0 / norm_type))


def clip_gradients(parameters: Iterable[nn.Parameter], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients in place and return the norm before clipping."""
    values = [p for p in parameters if p.grad is not None]
    if not values or max_norm <= 0.0:
        return gradient_norm(values, norm_type)
    total = torch.nn.utils.clip_grad_norm_(values, max_norm, norm_type=norm_type)
    return float(total)


def parameter_counts(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
        "frozen_millions": (total - trainable) / 1e6,
    }


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Bytes held by the optimiser state, for the footprint report."""
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total
