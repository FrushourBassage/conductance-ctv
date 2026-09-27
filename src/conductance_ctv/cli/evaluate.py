"""Evaluation entry point: run an experiment and write its tables and report.

Usage:
    python -m conductance_ctv.cli.evaluate experiment=main
    python -m conductance_ctv.cli.evaluate experiment=_smoke --steps 2 --out reports

Ref: Algorithm 3 and Tables 1 to 7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductance_ctv.evaluation.pipeline import run_pipeline, write_run_reports
from conductance_ctv.evaluation.report import render_table
from conductance_ctv.utils.logging_setup import configure_logging, get_logger

LOGGER = get_logger("cli.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conductance-ctv-evaluate",
        description="Execute one experiment end to end and write its per-table report.",
    )
    parser.add_argument("experiment", nargs="?", default="main")
    parser.add_argument("--steps", type=int, default=None, help="cap the number of optimiser steps")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--quiet", action="store_true", help="write the report without printing the tables")
    parser.add_argument("override", nargs="*", help="key=value overrides applied after the configuration is resolved")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    LOGGER.info("running experiment %s", arguments.experiment)
    run = run_pipeline(arguments.experiment, tuple(arguments.override), max_steps=arguments.steps)
    written = write_run_reports(run, arguments.out)
    if not arguments.quiet:
        for table in sorted(run.tables, key=lambda item: item.number):
            print(render_table(table))
        print("notes:")
        for note in run.notes:
            print(f" - {note}")
    print(json.dumps({"status": run.status, "artefacts": written, "timings": run.timings}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
