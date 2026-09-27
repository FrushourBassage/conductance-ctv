"""Patient-level cross-fitting of the segmenter.

The manuscript fine-tunes the backbone so that no evaluation limb was ever seen
by the segmenter that scores it: a patient-level cross-fitting scheme over the
development sites. The folds are disjoint over patients, and a fold's model sees
neither the patients of its own validation fold nor, when the external site is
later scored, any external patient.

Ref: Sec. 6.4 ("a patient-level cross-fitting scheme was implemented at both
Sites A and B in a manner such that no segmenting was performed on an evaluation
limb that had been seen by any segmenter").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from conductance_ctv.data.dataset import LimbCase, SegmentationDataset, build_cases
from conductance_ctv.data.schema import DEVELOPMENT_SITES, CohortRecord, GridSpec
from conductance_ctv.data.splits import crossfit_folds
from conductance_ctv.training.checkpoint import checkpoint_hash
from conductance_ctv.training.optim import PrecisionContext
from conductance_ctv.training.trainer import SegmentationTrainer, TrainConfig, TrainingHistory
from conductance_ctv.utils.logging_setup import get_logger
from conductance_ctv.utils.types import BoolArray, FloatArray, IntArray

LOGGER = get_logger("training.crossfit")


@dataclass
class FoldOutcome:
    """What one cross-fitting fold produced."""

    fold: int
    validation_patients: tuple[str, ...]
    train_cases: int
    validation_cases: int
    history: TrainingHistory
    parameters: dict[str, float]
    checkpoint: str | None = None


@dataclass
class CrossFitResult:
    """Every fold's outcome plus the out-of-fold predictions they combine into."""

    folds: list[FoldOutcome] = field(default_factory=list)
    out_of_fold_masks: dict[str, BoolArray] = field(default_factory=dict)
    out_of_fold_dice: dict[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        return {
            "n_folds": len(self.folds),
            "folds": [
                {
                    "fold": outcome.fold,
                    "validation_patients": len(outcome.validation_patients),
                    "train_cases": outcome.train_cases,
                    "validation_cases": outcome.validation_cases,
                    "steps": outcome.history.steps,
                    "checkpoint": outcome.checkpoint,
                }
                for outcome in self.folds
            ],
            "out_of_fold_volumes": len(self.out_of_fold_masks),
            "mean_out_of_fold_dice": (
                float(np.mean(list(self.out_of_fold_dice.values()))) if self.out_of_fold_dice else None
            ),
        }


def crossfit_segmenter(
    cohort: CohortRecord,
    cases: list[LimbCase],
    config: TrainConfig,
    encoder_name: str = "foundation_trunk",
    n_folds: int = 5,
    width_mult: float = 1.0,
    checkpoint_dir: str | Path | None = None,
    max_steps: int | None = None,
) -> CrossFitResult:
    """Train one segmenter per patient fold and collect out-of-fold predictions.

    The folds are drawn over the development patients only; the external site is
    never part of a training fold.
    """
    development = tuple(sorted({case.limb.patient_id for case in cases if case.limb.site in DEVELOPMENT_SITES}))
    if len(development) < n_folds:
        development = tuple(sorted({case.limb.patient_id for case in cases}))
    folds = crossfit_folds(development, n_folds, seed=config.seed)
    result = CrossFitResult()
    precision = PrecisionContext.build(config.precision)

    for index, fold in enumerate(folds):
        held = set(fold)
        train_cases = [case for case in cases if case.limb.patient_id not in held]
        validation_cases = [case for case in cases if case.limb.patient_id in held]
        if not train_cases or not validation_cases:
            LOGGER.warning("fold %d has an empty side and is skipped", index)
            continue
        trainer = SegmentationTrainer(config, encoder_name, width_mult=width_mult)
        checkpoint = Path(checkpoint_dir) / f"fold{index}.pt" if checkpoint_dir is not None else None
        model, history = trainer.fit(train_cases, validation_cases, checkpoint_path=checkpoint, max_steps=max_steps)
        model.eval()
        loader = torch.utils.data.DataLoader(
            SegmentationDataset(validation_cases, patch_shape=_patch_shape(cases)), batch_size=config.batch_size
        )
        with torch.no_grad():
            for batch in loader:
                image = batch["image"]
                with precision.autocast():
                    logits = model(image)
                predicted = (torch.argmax(logits, dim=1) > 0).cpu().numpy().astype(bool)
                reference = batch["mask"][:, 0].cpu().numpy().astype(bool)
                for row, case_index in enumerate(batch["index"].tolist()):
                    case = validation_cases[case_index]
                    result.out_of_fold_masks[case.limb.limb_id] = predicted[row]
                    result.out_of_fold_dice[case.limb.limb_id] = _dice(predicted[row], reference[row])
        result.folds.append(
            FoldOutcome(
                fold=index,
                validation_patients=fold,
                train_cases=len(train_cases),
                validation_cases=len(validation_cases),
                history=history,
                parameters=dict(trainer.parameters),
                checkpoint=checkpoint_hash(checkpoint) if checkpoint is not None else None,
            )
        )
    return result


def _patch_shape(cases: list[LimbCase]) -> tuple[int, int, int]:
    if not cases:
        return (32, 64, 64)
    return (min(32, cases[0].grid.shape[0]), 64, 64)


def _dice(predicted: BoolArray, reference: BoolArray) -> float:
    intersection = float(np.count_nonzero(predicted & reference))
    total = float(np.count_nonzero(predicted)) + float(np.count_nonzero(reference))
    if total <= 0.0:
        return float("nan")
    return 2.0 * intersection / total


def fit_full_development_model(
    cases: list[LimbCase],
    config: TrainConfig,
    encoder_name: str = "foundation_trunk",
    width_mult: float = 1.0,
    checkpoint_path: str | Path | None = None,
    max_steps: int | None = None,
) -> tuple[torch.nn.Module, TrainingHistory, dict[str, float]]:
    """Train one segmenter on every development case, for the external transfer.

    The external evaluation uses a model fitted on all development patients,
    which is the setting the transported metrics of Table 4 are reported in.
    """
    trainer = SegmentationTrainer(config, encoder_name, width_mult=width_mult)
    model, history = trainer.fit(cases, None, checkpoint_path=checkpoint_path, max_steps=max_steps)
    return model, history, dict(trainer.parameters)


def out_of_fold_indices(patient_ids: list[str], folds: tuple[tuple[str, ...], ...]) -> IntArray:
    """Fold index of every case, for the out-of-fold aggregation of Table 2."""
    lookup: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for patient in fold:
            lookup[patient] = index
    missing = [patient for patient in patient_ids if patient not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} patients are in no fold")
    return np.array([lookup[patient] for patient in patient_ids], dtype=np.int64)


def pooled_out_of_fold_predictions(fold_probabilities: list[FloatArray], indices: IntArray) -> FloatArray:
    """Concatenate per-fold predictions in the original case order."""
    if not fold_probabilities:
        return np.zeros((0, 4), dtype=np.float64)
    width = fold_probabilities[0].shape[1]
    out = np.zeros((indices.shape[0], width), dtype=np.float64)
    for fold, values in enumerate(fold_probabilities):
        rows = np.flatnonzero(indices == fold)
        if rows.size == 0:
            continue
        out[rows] = values[: rows.size]
    return out


def cases_by_patient(cases: list[LimbCase]) -> dict[str, list[LimbCase]]:
    out: dict[str, list[LimbCase]] = {}
    for case in cases:
        out.setdefault(case.limb.patient_id, []).append(case)
    return out


def render_development_cases(
    cohort: CohortRecord, grid: GridSpec | None = None, cache_dir: str | Path | None = None
) -> list[LimbCase]:
    """Render every limb of a cohort, delegating to the shared case builder."""
    if grid is None:
        return build_cases(cohort, cache_dir=cache_dir)
    return build_cases(cohort, grid=grid, cache_dir=cache_dir)
