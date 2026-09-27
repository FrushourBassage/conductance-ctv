"""Deterministic seeding for every random source the pipeline touches."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

DEFAULT_SEED = 20260101


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and torch, and return the seed that was applied.

    ``PYTHONHASHSEED`` cannot be changed once the interpreter is running, so the
    value is recorded here and is expected to be set by the launch script for a
    fully reproducible run.
    """
    if seed < 0:
        raise ValueError("the seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def rng_state() -> dict[str, Any]:
    """Capture the state of every random source, for a checkpoint."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state captured by :func:`rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def derived_seed(seed: int, stream: str, offset: int = 0) -> int:
    """A reproducible child seed for a named stream.

    Using a named stream keeps the data order, the augmentation and the model
    initialisation from competing for the same sequence, which is what makes a
    per-seed rerun differ in exactly one place.
    """
    digest = 0
    for character in stream:
        digest = (digest * 131 + ord(character)) % (2**31 - 1)
    return int((seed * 1000003 + digest + offset) % (2**31 - 1))
