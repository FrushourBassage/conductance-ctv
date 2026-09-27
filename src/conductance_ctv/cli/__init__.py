"""Command-line entry points for the release."""

from conductance_ctv.cli.architectures import main as architectures_main
from conductance_ctv.cli.evaluate import main as evaluate_main
from conductance_ctv.cli.hydraulics import main as hydraulics_main
from conductance_ctv.cli.train import main as train_main
from conductance_ctv.cli.verify import main as verify_main

__all__ = [
    "architectures_main",
    "evaluate_main",
    "hydraulics_main",
    "train_main",
    "verify_main",
]
