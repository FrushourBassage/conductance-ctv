"""Distributed helpers.

The training loops are written so that the single-process and the multi-process
paths run the same code: the only differences are the sampler, the gradient
averaging that the process group performs, and the log and checkpoint writes,
which are restricted to the rank-zero process.

Ref: the compute profile recorded in the project context; the manuscript reports
training hours on A100 devices without stating the device count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, Sampler


@dataclass(frozen=True)
class DistributedContext:
    """Rank topology of the current process."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def record(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self.backend,
        }


def initialize_distributed(backend: str = "nccl") -> DistributedContext:
    """Initialise the process group when the launcher provided the environment."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1 or not dist.is_available():
        return DistributedContext(False, 0, 1, 0, backend)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistributedContext(True, rank, world_size, local_rank, backend)


def wrap_model(model: nn.Module, context: DistributedContext, find_unused: bool = False) -> nn.Module:
    """Wrap a model in distributed data parallelism when the group is active."""
    if not context.enabled:
        return model
    device = torch.device("cuda", context.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return DistributedDataParallel(
        model.to(device),
        device_ids=[context.local_rank] if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused,
    )


def build_sampler(
    dataset_size: int, context: DistributedContext, shuffle: bool = True, seed: int = 0
) -> Sampler[int] | None:
    """Distributed sampler, or ``None`` to let the loader use its own shuffling."""
    if not context.enabled:
        return None
    return DistributedSampler(
        range(dataset_size),  # type: ignore[arg-type]
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
    )


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, context: DistributedContext) -> float:
    """Average a scalar across ranks, so the logged loss is the global one."""
    if not context.enabled or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64)
    device = torch.device("cuda", context.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    tensor = tensor.to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size)


def effective_batch_size(micro_batch: int, grad_accumulation: int, world_size: int) -> int:
    """Effective batch of a run, the product the manuscript's profile is defined on."""
    if min(micro_batch, grad_accumulation, world_size) < 1:
        raise ValueError("batch, accumulation and world size must all be at least one")
    return int(micro_batch) * int(grad_accumulation) * int(world_size)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
