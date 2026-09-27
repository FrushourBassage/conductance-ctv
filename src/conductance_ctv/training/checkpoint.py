"""Atomic checkpointing, with the seed and the random state carried inside."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from conductance_ctv.training.seed import restore_rng_state, rng_state
from conductance_ctv.utils.io import sha256_bytes, sha256_file

_FILE_MODE = 0o644
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CheckpointInfo:
    """What a checkpoint records about the run that produced it."""

    epoch: int
    global_step: int
    seed: int
    metrics: dict[str, float]
    config_hash: str
    format_version: int = _FORMAT_VERSION

    def record(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "seed": self.seed,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "format_version": self.format_version,
        }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    ema: object | None = None,
    info: CheckpointInfo | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a checkpoint by writing a temporary file and replacing the target.

    The replacement is atomic on the same filesystem, so a run that is
    interrupted mid-write leaves either the previous checkpoint or the new one
    and never a truncated file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "model": model.state_dict(),
        "rng": rng_state(),
        "info": info.record() if info is not None else {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if ema is not None and hasattr(ema, "state_dict"):
        payload["ema"] = ema.state_dict()
    if extra is not None:
        payload["extra"] = dict(extra)

    descriptor, temporary_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, _FILE_MODE)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    ema: object | None = None,
    restore_random_state: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, restoring the seed-bearing random state by default."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("the checkpoint does not contain a mapping")
    if int(payload.get("format_version", 0)) != _FORMAT_VERSION:
        raise ValueError("the checkpoint format version does not match this release")
    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and "ema" in payload and hasattr(ema, "load_state_dict"):
        ema.load_state_dict(payload["ema"])
    if restore_random_state and "rng" in payload:
        restore_rng_state(payload["rng"])
    return payload


def checkpoint_hash(path: str | Path) -> str:
    return sha256_file(path)


def content_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Digest of a state dictionary's tensor payloads rather than its container.

    ``torch.save`` embeds container metadata, so hashing the file would change on
    every write even when the parameters are identical. Hashing the concatenated
    payloads is stable across rewrites of the same values.
    """
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def config_digest(config: Mapping[str, Any]) -> str:
    """Stable digest of a configuration mapping, used to detect a mismatched resume."""
    import json

    body = json.dumps(config, sort_keys=True, default=str)
    return sha256_bytes(body.encode("utf-8"))


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    base = Path(directory)
    if not base.exists():
        return None
    candidates = sorted(base.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None
