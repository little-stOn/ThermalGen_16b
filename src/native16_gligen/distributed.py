from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn import Module


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        value = self.rank == 0
        return bool(value)


def initialize_distributed(timeout_minutes: int = 30) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        if distributed:
            dist.init_process_group("gloo", timeout=timedelta(minutes=timeout_minutes))
        return DistributedContext(rank, local_rank, world_size, distributed, device)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))
    return DistributedContext(rank, local_rank, world_size, distributed, device)


def validate_v100_precision(precision: str, device: torch.device) -> None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    if precision == "bf16" and major < 8:
        raise ValueError("bf16 is not supported on V100/Volta. Set train.precision=fp16")
    if precision == "fp16" and major < 7:
        raise ValueError(f"fp16 training requires compute capability 7.0+, got {major}.{minor}")
    return None


def build_mixed_precision(precision: str) -> MixedPrecision | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=dtype,
        cast_forward_inputs=True,
    )
    return policy


def wrap_fsdp(module: Module, cfg: Any, context: DistributedContext) -> Module:
    if not context.distributed or not bool(cfg.enabled):
        module.to(context.device)
        return module
    mixed_precision = build_mixed_precision(str(cfg.precision))
    cpu_offload = CPUOffload(offload_params=bool(cfg.cpu_offload))
    minimum_parameters = int(getattr(cfg, "auto_wrap_min_params", 0))
    auto_wrap_policy = None
    if minimum_parameters > 0:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=minimum_parameters,
        )
    wrapped = FullyShardedDataParallel(
        module,
        device_id=context.device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=True,
        forward_prefetch=bool(cfg.forward_prefetch),
    )
    return wrapped


def synchronize(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    return None


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.distributed and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= float(context.world_size)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Distributed reduction produced a non-finite value")
    return result


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return None
