"""Training loops for the segmenter and the type model.

Both loops share one shape: a deterministic seed, an optional mixed-precision
context, gradient accumulation, gradient clipping, an optional exponential moving
average, an atomic checkpoint per epoch, and a history that records what was
actually computed. The type loop adds the temperature calibration, which is fitted
on the tuning partition after training and never on the evaluation partition.

Ref: Sec. 6.4 and Sec. 6.6; the compute profile recorded in the project context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from conductance_ctv.data.dataset import LimbCase, SegmentationDataset, TypeDataset
from conductance_ctv.data.transforms import DescriptorStandardiser, SegmentationAugment
from conductance_ctv.losses import build_segmentation_loss, build_type_loss, segmentation_inside_weight
from conductance_ctv.metrics import macro_auroc, macro_f1
from conductance_ctv.models.branches import SEVERITY_INDEX
from conductance_ctv.models.classifier import ConductanceTypeModel, TemperatureCalibrator, apply_temperature_
from conductance_ctv.models.unet3d import build_segmentation_network
from conductance_ctv.training.checkpoint import CheckpointInfo, config_digest, save_checkpoint
from conductance_ctv.training.optim import (
    ModelEMA,
    PrecisionContext,
    WarmupCosineSchedule,
    build_optimizer,
    clip_gradients,
    parameter_counts,
)
from conductance_ctv.training.seed import derived_seed, set_seed
from conductance_ctv.utils.logging_setup import get_logger

LOGGER = get_logger("training.trainer")

# Column of the dimensionless resistance ratio of the common iliac segment in the
# concatenated descriptor vector, used as the severity regression target.
_CIV_RESISTANCE_RATIO_COLUMN = SEVERITY_INDEX


@dataclass
class TrainingHistory:
    """Per-epoch losses and validation values of one training run."""

    train_loss: list[float] = field(default_factory=list)
    validation: list[dict[str, float]] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    steps: int = 0
    stopped_early: bool = False

    def record(self) -> dict[str, object]:
        return {
            "epochs": len(self.train_loss),
            "steps": self.steps,
            "first_loss": self.train_loss[0] if self.train_loss else None,
            "last_loss": self.train_loss[-1] if self.train_loss else None,
            "best_loss": min(self.train_loss) if self.train_loss else None,
            "stopped_early": self.stopped_early,
            "train_loss": self.train_loss,
            "learning_rate": self.learning_rate,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters, resolved from the experiment configuration."""

    epochs: int = 200
    batch_size: int = 8
    grad_accumulation: int = 1
    learning_rate: float = 3e-4
    backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    warmup_epochs: float = 5.0
    optimizer: str = "adamw"
    scheduler: str = "warmup_cosine"
    precision: str = "bf16"
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    num_workers: int = 0
    seed: int = 20260101
    patience: int = 0
    deep_supervision: bool = False
    loss: str = "confluence_dice_ce"
    confluence_inside_weight: float = 3.0
    class_balance_beta: float = 0.999
    fairness_weight: float = 0.0
    severity_weight: float = 0.3
    label_smoothing: float = 0.0
    freeze_backbone: bool = False
    augment: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accumulation

    def record(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accumulation": self.grad_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": self.learning_rate,
            "backbone_learning_rate": self.backbone_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "precision": self.precision,
            "grad_clip": self.grad_clip,
            "ema_decay": self.ema_decay,
            "seed": self.seed,
            "deep_supervision": self.deep_supervision,
            "loss": self.loss,
            "confluence_inside_weight": self.confluence_inside_weight,
            "class_balance_beta": self.class_balance_beta,
            "fairness_weight": self.fairness_weight,
            "severity_weight": self.severity_weight,
            "freeze_backbone": self.freeze_backbone,
        }


def _one_hot_lumen(mask: torch.Tensor) -> torch.Tensor:
    """Turn a single-channel lumen mask into a background-and-lumen pair."""
    foreground = mask[:, 0]
    return torch.stack([1.0 - foreground, foreground], dim=1)


def _loader(
    dataset: SegmentationDataset | TypeDataset, config: TrainConfig, shuffle: bool, seed: int
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(derived_seed(seed, "loader"))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=False,
        generator=generator,
    )


class SegmentationTrainer:
    """Trains one segmenter on one fold and scores it on the held-out fold."""

    parameters: dict[str, float]

    def __init__(
        self,
        config: TrainConfig,
        encoder_name: str,
        width_mult: float = 1.0,
        decoder_width_mult: float = 1.0,
    ) -> None:
        self.config = config
        self.encoder_name = encoder_name
        self.width_mult = width_mult
        self.decoder_width_mult = decoder_width_mult
        self.parameters = {}

    def build(self) -> nn.Module:
        built = build_segmentation_network(
            self.encoder_name,
            width_mult=self.width_mult,
            deep_supervision=self.config.deep_supervision,
            decoder_width_mult=self.decoder_width_mult,
        )
        self.parameters = parameter_counts(built.network)
        return built.network

    def fit(
        self,
        train_cases: list[LimbCase],
        validation_cases: list[LimbCase] | None = None,
        checkpoint_path: str | Path | None = None,
        max_steps: int | None = None,
    ) -> tuple[nn.Module, TrainingHistory]:
        """Train and return the model with the history of what happened."""
        set_seed(self.config.seed)
        model = self.build()
        augmentation = SegmentationAugment(through_plane_sigma=0.0, noise_sigma=5.0) if self.config.augment else None
        inside_weight = segmentation_inside_weight(self.config.loss, self.config.confluence_inside_weight)
        trainset = SegmentationDataset(
            train_cases, augment=augmentation, seed=self.config.seed, inside_weight=inside_weight
        )
        loader = _loader(trainset, self.config, shuffle=True, seed=self.config.seed)
        validation_loader = (
            _loader(
                SegmentationDataset(validation_cases, inside_weight=inside_weight),
                self.config,
                shuffle=False,
                seed=self.config.seed,
            )
            if validation_cases
            else None
        )

        criterion = build_segmentation_loss(self.config.loss, deep_supervision=self.config.deep_supervision)
        optimizer = build_optimizer(
            model.parameters(), self.config.optimizer, self.config.learning_rate, self.config.weight_decay
        )
        steps_per_epoch = max(1, len(loader) // max(1, self.config.grad_accumulation))
        total_steps = max(1, steps_per_epoch * self.config.epochs)
        schedule = WarmupCosineSchedule(
            self.config.learning_rate,
            int(self.config.warmup_epochs * steps_per_epoch),
            total_steps,
        )
        precision = PrecisionContext.build(self.config.precision)
        ema = ModelEMA(model, self.config.ema_decay) if self.config.ema_decay > 0.0 else None
        history = TrainingHistory()
        best = float("inf")
        stale = 0

        for epoch in range(self.config.epochs):
            started = time.time()
            model.train()
            running = 0.0
            batches = 0
            optimizer.zero_grad(set_to_none=True)
            for position, batch in enumerate(loader):
                image = batch["image"]
                mask = _one_hot_lumen(batch["mask"])
                weight = batch["weight"]
                with precision.autocast():
                    logits = model(image)
                    parts = criterion(logits, mask, weight)
                    loss = parts["total"] / self.config.grad_accumulation
                if precision.scaler is not None:
                    precision.scaler.scale(loss).backward()
                else:
                    loss.backward()
                running += float(parts["total"].detach())
                batches += 1
                if (position + 1) % self.config.grad_accumulation == 0:
                    if precision.scaler is not None:
                        precision.scaler.unscale_(optimizer)
                    clip_gradients(model.parameters(), self.config.grad_clip)
                    if precision.scaler is not None:
                        precision.scaler.step(optimizer)
                        precision.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    schedule.step()
                    schedule.apply(optimizer)
                    history.steps += 1
                    if ema is not None:
                        ema.update(model)
                    if max_steps is not None and history.steps >= max_steps:
                        break
            epoch_loss = running / max(batches, 1)
            history.train_loss.append(epoch_loss)
            history.learning_rate.append(schedule.get_lr())
            history.seconds.append(time.time() - started)
            if validation_loader is not None:
                history.validation.append(self.evaluate(model, validation_loader, criterion, precision))
            if checkpoint_path is not None:
                info = CheckpointInfo(
                    epoch=epoch,
                    global_step=history.steps,
                    seed=self.config.seed,
                    metrics={"train_loss": epoch_loss},
                    config_hash=config_digest(self.config.record()),
                )
                save_checkpoint(checkpoint_path, model, optimizer, schedule, ema, info)
            if epoch_loss < best - 1e-6:
                best = epoch_loss
                stale = 0
            else:
                stale += 1
                if self.config.patience > 0 and stale >= self.config.patience:
                    history.stopped_early = True
                    break
            if max_steps is not None and history.steps >= max_steps:
                break
        if ema is not None:
            ema.copy_to(model)
        return model, history

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader[dict[str, torch.Tensor]],
        criterion: nn.Module | None = None,
        precision: PrecisionContext | None = None,
    ) -> dict[str, float]:
        model.eval()
        context = precision if precision is not None else PrecisionContext.build(self.config.precision)
        total = 0.0
        batches = 0
        intersection = 0.0
        union = 0.0
        with torch.no_grad():
            for batch in loader:
                image = batch["image"]
                mask = _one_hot_lumen(batch["mask"])
                with context.autocast():
                    logits = model(image)
                    if criterion is not None:
                        parts = criterion(logits, mask, batch["weight"])
                        total += float(parts["total"])
                        batches += 1
                predicted = torch.argmax(logits, dim=1, keepdim=True).float()
                foreground = mask[:, 1:2]
                intersection += float(torch.sum(predicted * foreground))
                union += float(torch.sum(predicted)) + float(torch.sum(foreground))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return {"loss": total / max(batches, 1), "dice": dice}


class TypeTrainer:
    """Trains the two-branch type model and fits its temperature."""

    def __init__(
        self,
        config: TrainConfig,
        encoder_name: str = "foundation_trunk",
        width_mult: float = 1.0,
        decoder_width_mult: float = 1.0,
    ) -> None:
        self.config = config
        self.encoder_name = encoder_name
        self.width_mult = width_mult
        self.decoder_width_mult = decoder_width_mult

    def build(self) -> ConductanceTypeModel:
        built = build_segmentation_network(
            self.encoder_name, width_mult=self.width_mult, decoder_width_mult=self.decoder_width_mult
        )
        return ConductanceTypeModel(
            segmenter=built.network,
            freeze_backbone=self.config.freeze_backbone,
        )

    def fit(
        self,
        train_cases: list[LimbCase],
        validation_cases: list[LimbCase],
        checkpoint_path: str | Path | None = None,
        max_steps: int | None = None,
    ) -> tuple[ConductanceTypeModel, TrainingHistory, DescriptorStandardiser]:
        set_seed(self.config.seed)
        model = self.build()
        standardiser = DescriptorStandardiser().fit(np.stack([case.descriptor for case in train_cases], axis=0))
        augment = SegmentationAugment(through_plane_sigma=0.0, noise_sigma=5.0) if self.config.augment else None
        trainset = TypeDataset(train_cases, standardiser=standardiser, augment=augment, seed=self.config.seed)
        validset = TypeDataset(validation_cases, standardiser=standardiser)
        loader = _loader(trainset, self.config, shuffle=True, seed=self.config.seed)

        criterion = build_type_loss(
            beta=self.config.class_balance_beta,
            label_smoothing=self.config.label_smoothing,
            fairness_weight=self.config.fairness_weight,
            severity_weight=self.config.severity_weight,
        )
        criterion.fit_class_weights([case.label for case in train_cases])
        backbone_lr = (
            self.config.backbone_learning_rate
            if self.config.backbone_learning_rate is not None
            else self.config.learning_rate * 0.2
        )
        groups_list = model.trainable_parameter_groups(backbone_lr, self.config.learning_rate)
        for group in groups_list:
            group_lr = group.get("lr")
            scale = (
                float(group_lr) / max(self.config.learning_rate, 1e-12) if isinstance(group_lr, (int, float)) else 1.0
            )
            group["lr_scale"] = scale
        optimizer = build_optimizer(
            groups_list, self.config.optimizer, self.config.learning_rate, self.config.weight_decay
        )
        steps_per_epoch = max(1, len(loader) // max(1, self.config.grad_accumulation))
        total_steps = max(1, steps_per_epoch * self.config.epochs)
        schedule = WarmupCosineSchedule(
            self.config.learning_rate, int(self.config.warmup_epochs * steps_per_epoch), total_steps
        )
        precision = PrecisionContext.build(self.config.precision)
        ema = ModelEMA(model, self.config.ema_decay) if self.config.ema_decay > 0.0 else None
        history = TrainingHistory()

        for epoch in range(self.config.epochs):
            started = time.time()
            model.train()
            running = 0.0
            batches = 0
            optimizer.zero_grad(set_to_none=True)
            for position, batch in enumerate(loader):
                descriptors = batch["descriptor"]
                appearance = batch["appearance"]
                labels = batch["label"]
                site = batch["site"]
                severity_target = batch["descriptor_raw"][:, _CIV_RESISTANCE_RATIO_COLUMN].clone()
                censored = ~torch.isfinite(severity_target)
                severity_target = torch.nan_to_num(severity_target, nan=1.0, posinf=1e6, neginf=1.0)
                with precision.autocast():
                    output = model(appearance, descriptors)
                    parts = criterion(
                        output.logits,
                        output.probabilities,
                        output.severity_descriptor,
                        labels,
                        severity_target,
                        groups=site,
                        censored=censored,
                    )
                    loss = parts["total"] / self.config.grad_accumulation
                if precision.scaler is not None:
                    precision.scaler.scale(loss).backward()
                else:
                    loss.backward()
                running += float(parts["total"].detach())
                batches += 1
                if (position + 1) % self.config.grad_accumulation == 0:
                    if precision.scaler is not None:
                        precision.scaler.unscale_(optimizer)
                    clip_gradients(model.parameters(), self.config.grad_clip)
                    if precision.scaler is not None:
                        precision.scaler.step(optimizer)
                        precision.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    schedule.step()
                    schedule.apply(optimizer)
                    history.steps += 1
                    if ema is not None:
                        ema.update(model)
                    if max_steps is not None and history.steps >= max_steps:
                        break
            history.train_loss.append(running / max(batches, 1))
            history.learning_rate.append(schedule.get_lr())
            history.seconds.append(time.time() - started)
            history.validation.append(self.evaluate(model, validset))
            if checkpoint_path is not None:
                info = CheckpointInfo(
                    epoch=epoch,
                    global_step=history.steps,
                    seed=self.config.seed,
                    metrics=history.validation[-1],
                    config_hash=config_digest(self.config.record()),
                )
                save_checkpoint(checkpoint_path, model, optimizer, schedule, ema, info)
            if max_steps is not None and history.steps >= max_steps:
                break
        if ema is not None:
            ema.copy_to(model)
        return model, history, standardiser

    def evaluate(self, model: ConductanceTypeModel, dataset: TypeDataset) -> dict[str, float]:
        loader = _loader(dataset, self.config, shuffle=False, seed=self.config.seed)
        model.eval()
        probability_rows: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        label_rows: list[int] = []
        with torch.no_grad():
            for batch in loader:
                output = model(batch["appearance"], batch["descriptor"])
                probability_rows.append(output.probabilities.cpu().numpy().astype(np.float64))
                label_rows.extend(int(value) + 1 for value in batch["label"].tolist())
        if not probability_rows:
            return {"macro_auroc": float("nan"), "macro_f1": float("nan")}
        probabilities = np.concatenate(probability_rows, axis=0)
        labels = np.asarray(label_rows, dtype=np.int64)
        predicted = np.argmax(probabilities, axis=1) + 1
        return {
            "macro_auroc": macro_auroc(probabilities, labels),
            "macro_f1": macro_f1(predicted, labels),
        }


def calibrate_on_tuning(
    model: ConductanceTypeModel,
    tuning_dataset: TypeDataset,
    config: TrainConfig,
) -> TemperatureCalibrator:
    """Fit the temperature on the tuning partition and write it into the head."""
    loader = _loader(tuning_dataset, config, shuffle=False, seed=config.seed)
    model.eval()
    logits: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["appearance"], batch["descriptor"])
            logits.append(output.logits.cpu().numpy().astype(np.float64))
            labels.extend(int(value) for value in batch["label"].tolist())
    if not logits:
        return TemperatureCalibrator(1.0)
    calibrator = TemperatureCalibrator(1.0)
    calibrator.fit(np.concatenate(logits, axis=0), np.asarray(labels, dtype=np.int64))
    apply_temperature_(model, calibrator.temperature)
    return calibrator


def predict_probabilities(
    model: ConductanceTypeModel, dataset: TypeDataset, config: TrainConfig
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Calibrated probabilities and reference labels of a partition."""
    loader = _loader(dataset, config, shuffle=False, seed=config.seed)
    model.eval()
    rows: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["appearance"], batch["descriptor"])
            rows.append(output.probabilities.cpu().numpy().astype(np.float64))
            labels.extend(int(value) + 1 for value in batch["label"].tolist())
    if not rows:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(rows, axis=0), np.asarray(labels, dtype=np.int64)


def summarise_training(history: TrainingHistory, parameters: dict[str, float]) -> dict[str, object]:
    """Training record for a report, with the measured parameter count."""
    payload = history.record()
    payload["parameters"] = parameters
    payload["loss_decreased"] = (
        bool(history.train_loss and history.train_loss[-1] < history.train_loss[0]) if history.train_loss else False
    )
    payload["relative_loss_drop"] = (
        float((history.train_loss[0] - history.train_loss[-1]) / abs(history.train_loss[0]))
        if history.train_loss and abs(history.train_loss[0]) > 1e-12
        else 0.0
    )
    LOGGER.info(
        "training finished after %d steps; loss %.4f to %.4f",
        history.steps,
        history.train_loss[0] if history.train_loss else float("nan"),
        history.train_loss[-1] if history.train_loss else float("nan"),
    )
    return payload
