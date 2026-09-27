"""Descriptor entry point: solve Eq. (1) on a section and report its signature.

Usage:
    python -m conductance_ctv.cli.hydraulics --shape ellipse --a 20 --b 6
    python -m conductance_ctv.cli.hydraulics --mask lumen.npz --spacing 0.6

Ref: Sec. 6.5 and Algorithm 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from conductance_ctv.hydraulics.conductance import analyse_section, shape_factor_from_axis_ratio
from conductance_ctv.hydraulics.descriptors import signature_from_mask
from conductance_ctv.hydraulics.grid import CrossSection
from conductance_ctv.hydraulics.shapes import (
    disc_channel_section,
    disc_section,
    ellipse_section,
    rectangle_section,
    septum_section,
)
from conductance_ctv.utils.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conductance-ctv-hydraulics",
        description="Solve the unit-source Dirichlet problem on a lumen section and report its hydraulic signature.",
    )
    parser.add_argument("--shape", choices=("disc", "ellipse", "rectangle", "septum"), default="ellipse")
    parser.add_argument("--a", type=float, default=20.0, help="first semi-extent in mm")
    parser.add_argument("--b", type=float, default=6.0, help="second semi-extent in mm")
    parser.add_argument("--channels", type=int, default=3, help="channel count for the septum shape")
    parser.add_argument("--area", type=float, default=1800.0, help="combined area in mm^2 for the septum shape")
    parser.add_argument("--step", type=float, default=0.5, help="in-plane grid step in mm")
    parser.add_argument("--tolerance", type=float, default=1e-11, help="SOR update tolerance")
    parser.add_argument("--max-sweeps", type=int, default=20000)
    parser.add_argument("--mask", type=Path, help="an npz holding a three-dimensional boolean lumen mask")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="voxel spacing z y x in mm")
    parser.add_argument("--json", action="store_true", help="emit the signature as JSON")
    return parser


def section_from_arguments(arguments: argparse.Namespace) -> CrossSection:
    """Build the section the arguments describe."""
    if arguments.shape == "disc":
        return disc_section(arguments.a, spacing=arguments.step)
    if arguments.shape == "ellipse":
        return ellipse_section(arguments.a, arguments.b, spacing=arguments.step)
    if arguments.shape == "rectangle":
        return rectangle_section(arguments.a, arguments.b, spacing=arguments.step)
    if arguments.shape == "septum":
        return septum_section(arguments.area, arguments.channels, spacing=arguments.step)
    raise KeyError(f"unknown shape {arguments.shape!r}")


def describe_section_arguments(
    shape: str = "ellipse",
    a: float = 20.0,
    b: float = 6.0,
    channels: int = 3,
    area: float = 1800.0,
    step: float = 0.5,
) -> dict[str, Any]:
    """Signature of a section described by its geometry, without touching the parser."""
    if shape == "disc":
        section = disc_section(a, spacing=step)
    elif shape == "ellipse":
        section = ellipse_section(a, b, spacing=step)
    elif shape == "rectangle":
        section = rectangle_section(a, b, spacing=step)
    elif shape == "septum":
        section = septum_section(area, channels, spacing=step)
    elif shape == "channels":
        section = disc_channel_section([a, b], [(0.0, 0.0), (0.0, 2.0 * max(a, b) + 1.0)], spacing=step)
    else:
        raise KeyError(f"unknown shape {shape!r}")
    analysis = analyse_section(section)
    return {
        "shape": shape,
        "area_mm2": analysis.area,
        "occupancy_area_mm2": analysis.occupancy_area,
        "interface_edges": analysis.interface_edges,
        "conductance_mm4": analysis.conductance,
        "shape_factor": analysis.shape_factor,
        "channels": analysis.channels,
        "channel_areas_mm2": list(analysis.channel_areas),
        "below_floor": analysis.below_floor,
        "resolution_limited": analysis.resolution_limited,
        "iterations": analysis.iterations,
        "residual": analysis.residual,
        "converged": analysis.converged,
        "elliptic_closed_form_shape_factor": (
            shape_factor_from_axis_ratio(min(a, b) / max(a, b)) if shape in ("ellipse", "disc") else float("nan")
        ),
    }


def read_mask(path: Path) -> tuple[np.ndarray[Any, Any], tuple[float, float, float]]:
    payload = np.load(path)
    if "mask" not in payload:
        raise KeyError(f"{path.name} does not contain a 'mask' array")
    mask = np.asarray(payload["mask"]).astype(bool)
    if mask.ndim != 3:
        raise ValueError("the mask must be three-dimensional")
    spacing = tuple(float(value) for value in payload["spacing"]) if "spacing" in payload else (1.0, 1.0, 1.0)
    return mask, spacing  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    if arguments.mask is not None:
        mask, spacing = read_mask(arguments.mask)
        signature = signature_from_mask(mask, spacing, step=arguments.spacing[0])
        if arguments.json:
            print(json.dumps(signature.record(), indent=2, default=str))
            return 0
        print(f"limb signature over {signature.arclength.shape[0]} arc-length samples")
        header = (
            f"{'segment':8s}{'MLA_mm2':>12s}{'s*_mm':>10s}{'kappa':>10s}"
            f"{'nch':>6s}{'R':>14s}{'rho_R':>10s}{'L_w_mm':>10s}{'L_occl':>9s}"
        )
        print(header)
        print("-" * len(header))
        for segment in signature.segments:
            print(
                f"{segment.name:8s}{segment.minimum_luminal_area:12.2f}{segment.minimum_conductance_arclength:10.2f}"
                f"{segment.shape_factor:10.4f}{segment.channel_count:6.0f}{segment.series_resistance:14.4f}"
                f"{segment.resistance_ratio:10.4f}{segment.conductance_weighted_length:10.2f}"
                f"{segment.occlusion_length:9.2f}"
            )
        return 0
    report = describe_section_arguments(
        arguments.shape, arguments.a, arguments.b, arguments.channels, arguments.area, arguments.step
    )
    if arguments.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    for key, value in report.items():
        print(f"{key:36s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
