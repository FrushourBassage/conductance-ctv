"""Shared fixtures. The source tree is importable without an install step."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conductance_ctv.data.fixture import generate_cohort  # noqa: E402
from conductance_ctv.data.schema import GridSpec  # noqa: E402

SMALL_GRID = GridSpec(shape=(32, 24, 24), spacing_zyx=(1.5, 1.2, 1.2))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def small_grid() -> GridSpec:
    return SMALL_GRID


@pytest.fixture(scope="session")
def small_cohort():
    return generate_cohort(24, seed=101, grid=SMALL_GRID)


@pytest.fixture(scope="session")
def small_cases(small_cohort):
    from conductance_ctv.data.dataset import build_cases

    return build_cases(small_cohort, grid=SMALL_GRID)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(4242)
