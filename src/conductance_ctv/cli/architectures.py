"""Encoder inventory: measured parameter totals and the resolved width multipliers.

Usage:
    python -m conductance_ctv.cli.architectures
    python -m conductance_ctv.cli.architectures --resolve --json

Ref: Table 1 parameter column.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch

from conductance_ctv.models.encoders import (
    ENCODER_SPECS,
    count_parameters_millions,
    resolve_width_bracket,
    width_factory,
)
from conductance_ctv.models.unet3d import build_segmentation_network
from conductance_ctv.utils.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conductance-ctv-architectures",
        description=(
            "Report the measured parameter count of every encoder family and the width multiplier that matches its"
            " reported total."
        ),
    )
    parser.add_argument("--resolve", action="store_true", help="search for the width multiplier of each reported total")
    parser.add_argument(
        "--with-decoder", action="store_true", help="measure the full segmentation network instead of the encoder"
    )
    parser.add_argument("--json", action="store_true")
    return parser


def segmentation_encoder_factory(name: str) -> Callable[[float], torch.nn.Module]:
    """One-argument builder for the encoder inside a segmentation network."""

    def build(multiplier: float) -> torch.nn.Module:
        return build_segmentation_network(name, width_mult=multiplier).network.encoder

    return build


def measure(resolve: bool = False, with_decoder: bool = False) -> list[dict[str, Any]]:
    """Measured parameter total per family, with the width bracket when asked.

    Channel counts are integers, so a reported total generally falls between two
    achievable ones; the row carries both neighbours and the residual the ladder
    implies rather than a single rounded match.
    """
    rows: list[dict[str, Any]] = []
    for spec in ENCODER_SPECS:
        factory = segmentation_encoder_factory(spec.name) if with_decoder else width_factory(spec.family, spec.kwargs)
        measured = count_parameters_millions(factory(1.0))
        row: dict[str, Any] = {
            "encoder": spec.name,
            "family": spec.family,
            "reported_millions": spec.reported_params_millions,
            "measured_millions": measured,
            "default_width_mult": 1.0,
        }
        if resolve:
            try:
                resolution = resolve_width_bracket(factory, spec.reported_params_millions)
            except ValueError as error:
                row["resolution"] = None
                row["note"] = str(error)
            else:
                row["resolved_width_mult"] = resolution.multiplier
                row["resolved_millions"] = resolution.achieved_millions
                row["relative_gap"] = resolution.relative_gap
                row["ladder_granularity"] = resolution.ladder_granularity
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    rows = measure(arguments.resolve, arguments.with_decoder)
    if arguments.json:
        print(json.dumps(rows, indent=2))
        return 0
    header = (
        f"{'encoder':24s}{'family':12s}{'reported':>10s}{'measured':>10s}{'mult':>8s}"
        f"{'achieved':>10s}{'gap':>9s}{'ladder':>9s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        values = (
            row.get("resolved_width_mult"),
            row.get("resolved_millions"),
            row.get("relative_gap"),
            row.get("ladder_granularity"),
        )
        formatted = "".join(
            f"{(value if value is not None else float('nan')):{width}.{digits}f}"
            for value, width, digits in zip(values, (8, 10, 9, 9), (3, 2, 4, 4), strict=True)
        )
        print(
            f"{row['encoder']:24s}{row['family']:12s}{row['reported_millions']:10.1f}"
            f"{row['measured_millions']:10.2f}{formatted}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
