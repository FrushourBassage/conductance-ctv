"""Training entry point for the segmenter and the type model.

Usage:
    python -m conductance_ctv.cli.train experiment=main stage=segmentation
    python -m conductance_ctv.cli.train experiment=main stage=type train.epochs=50

Ref: Sec. 6.4 and Sec. 6.6; the compute profile recorded in the project context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductance_ctv.data.dataset import TypeDataset, build_cases
from conductance_ctv.data.splits import patient_level_split
from conductance_ctv.evaluation.pipeline import generate_pipeline_cohort
from conductance_ctv.training.checkpoint import save_checkpoint
from conductance_ctv.training.trainer import (
    SegmentationTrainer,
    TypeTrainer,
    calibrate_on_tuning,
    summarise_training,
)
from conductance_ctv.utils.config import load_experiment
from conductance_ctv.utils.io import write_json
from conductance_ctv.utils.logging_setup import configure_logging, get_logger

LOGGER = get_logger("cli.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conductance-ctv-train",
        description="Train the lumen segmenter or the four-type model named by an experiment.",
        epilog="Overrides use dotted paths, for example train.learning_rate=0.0001.",
    )
    parser.add_argument("experiment", nargs="?", default="main")
    parser.add_argument("--stage", choices=("segmentation", "type", "both"), default="both")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many optimiser steps")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("override", nargs="*", help="key=value overrides applied after the configuration is resolved")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    config = load_experiment(arguments.experiment, tuple(arguments.override))
    LOGGER.info("resolved experiment %s from %s", config.name, config.path)
    cohort = generate_pipeline_cohort(config)
    cases = build_cases(cohort, grid=config.grid)
    split = patient_level_split(cohort, seed=config.seed)
    train_mask, tune_mask, _ = split.masks(cohort)
    train_cases = [case for case, keep in zip(cases, train_mask, strict=True) if keep]
    tune_cases = [case for case, keep in zip(cases, tune_mask, strict=True) if keep]
    LOGGER.info(
        "cohort: %d patients, %d limbs, %d train cases, %d tune cases",
        cohort.n_patients,
        cohort.n_limbs,
        len(train_cases),
        len(tune_cases),
    )

    arguments.out.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "experiment": config.documented(),
        "cohort_patients": cohort.n_patients,
        "cohort_limbs": cohort.n_limbs,
        "train_cases": len(train_cases),
        "tune_cases": len(tune_cases),
    }

    if arguments.stage in ("segmentation", "both"):
        segmentation_trainer = SegmentationTrainer(config.train, config.encoder_name, width_mult=config.width_mult)
        segmentation_checkpoint = arguments.out / f"{config.name}_segmentation.pt"
        _, segmentation_history = segmentation_trainer.fit(
            train_cases, tune_cases, checkpoint_path=segmentation_checkpoint, max_steps=arguments.steps
        )
        segmentation_summary = summarise_training(segmentation_history, dict(segmentation_trainer.parameters))
        segmentation_summary["checkpoint"] = str(segmentation_checkpoint)
        record["segmentation"] = segmentation_summary

    if arguments.stage in ("type", "both"):
        type_trainer = TypeTrainer(config.train, config.encoder_name, width_mult=config.width_mult)
        type_checkpoint = arguments.out / f"{config.name}_type.pt"
        type_model, type_history, standardiser = type_trainer.fit(
            train_cases, tune_cases, checkpoint_path=type_checkpoint, max_steps=arguments.steps
        )
        calibrator = calibrate_on_tuning(type_model, TypeDataset(tune_cases, standardiser=standardiser), config.train)
        type_summary = summarise_training(type_history, {"type_model": float(type_history.steps)})
        type_summary["temperature"] = calibrator.temperature
        type_summary["checkpoint"] = str(type_checkpoint)
        type_summary["standardiser"] = standardiser.record()
        record["type"] = type_summary
        save_checkpoint(
            arguments.out / f"{config.name}_type_final.pt",
            type_model,
            info=None,
            extra={"temperature": calibrator.temperature, "standardiser": standardiser.record()},
        )

    destination = arguments.report if arguments.report is not None else arguments.out / f"{config.name}_training.json"
    write_json(destination, record)
    LOGGER.info("training record written to %s", destination)
    if arguments.report is None:
        print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
