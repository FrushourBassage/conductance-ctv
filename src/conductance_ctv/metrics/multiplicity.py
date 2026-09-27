"""Multiplicity control for the secondary endpoints.

The secondary criteria are tested in a pre-established order and the sequence
stops at the first non-significant step, so no secondary endpoint is reported as
significant once an earlier one failed. A Holm step-down is provided as well for
the case where the ordering is not fixed in advance.

Ref: Sec. 6.8 ("Multiplicity concerning the secondary criteria is managed by
conducting the testing in a pre-established order and halting whenever a
non-significant step is met").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductance_ctv.utils.types import FloatArray


@dataclass(frozen=True)
class StepOutcome:
    """One step of a multiplicity-controlled sequence."""

    name: str
    p_value: float
    alpha: float
    rejected: bool
    stopped: bool

    def record(self) -> dict[str, float | str | bool]:
        return {
            "endpoint": self.name,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "rejected": self.rejected,
            "stopped": self.stopped,
        }


def fixed_sequence(names: list[str], p_values: FloatArray, alpha: float = 0.05) -> list[StepOutcome]:
    """Pre-ordered testing that halts at the first non-significant step."""
    values = np.asarray(p_values, dtype=np.float64)
    if len(names) != values.shape[0]:
        raise ValueError("each endpoint needs exactly one p-value")
    outcomes: list[StepOutcome] = []
    stopped = False
    for name, value in zip(names, values, strict=True):
        if stopped:
            outcomes.append(StepOutcome(name, float(value), alpha, False, True))
            continue
        rejected = bool(value <= alpha)
        # The failing step is the one the sequence stopped at, so it carries the
        # stop flag as well as the non-rejection.
        outcomes.append(StepOutcome(name, float(value), alpha, rejected, not rejected))
        if not rejected:
            stopped = True
    return outcomes


def holm_step_down(names: list[str], p_values: FloatArray, alpha: float = 0.05) -> list[StepOutcome]:
    """Holm step-down, which controls the family-wise error without an ordering."""
    values = np.asarray(p_values, dtype=np.float64)
    if len(names) != values.shape[0]:
        raise ValueError("each endpoint needs exactly one p-value")
    order = np.argsort(values, kind="mergesort")
    total = values.shape[0]
    outcomes: list[StepOutcome] = [
        StepOutcome(name, float(p), alpha, False, False) for name, p in zip(names, values, strict=True)
    ]
    stopped = False
    for rank, index in enumerate(order):
        if stopped:
            outcomes[int(index)] = StepOutcome(names[int(index)], float(values[int(index)]), alpha, False, True)
            continue
        threshold = alpha / float(total - rank)
        rejected = bool(values[int(index)] <= threshold)
        outcomes[int(index)] = StepOutcome(names[int(index)], float(values[int(index)]), alpha, rejected, not rejected)
        if not rejected:
            stopped = True
    return outcomes


def benjamini_hochberg(names: list[str], p_values: FloatArray, alpha: float = 0.05) -> list[StepOutcome]:
    """Benjamini-Hochberg step-up, for a false-discovery-rate reading of the family."""
    values = np.asarray(p_values, dtype=np.float64)
    if len(names) != values.shape[0]:
        raise ValueError("each endpoint needs exactly one p-value")
    order = np.argsort(-values, kind="mergesort")
    total = values.shape[0]
    outcomes: list[StepOutcome] = [
        StepOutcome(name, float(p), alpha, False, False) for name, p in zip(names, values, strict=True)
    ]
    below = False
    for rank, index in enumerate(order):
        threshold = alpha * float(total - rank) / total
        if not below and values[int(index)] <= threshold:
            below = True
        outcomes[int(index)] = StepOutcome(names[int(index)], float(values[int(index)]), threshold, below, False)
    return outcomes


def sequence_report(outcomes: list[StepOutcome], method: str = "fixed_sequence") -> dict[str, object]:
    return {
        "method": method,
        "steps": [outcome.record() for outcome in outcomes],
        "rejected_count": sum(1 for outcome in outcomes if outcome.rejected),
        "stopped_at": next((outcome.name for outcome in outcomes if outcome.stopped), None),
    }
