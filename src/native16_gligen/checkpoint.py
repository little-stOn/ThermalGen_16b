from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.nn import Module

from native16_gligen.distributed import DistributedContext, synchronize


def collect_trainable_parameter_names(module: Module) -> set[str]:
    names = set()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            names.add(name)
    if not names:
        raise RuntimeError("No trainable parameters were selected")
    return names


def filter_model_state(state: dict[str, torch.Tensor], trainable_names: set[str]) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        keep = name in trainable_names
        if keep:
            filtered[name] = tensor.detach().cpu()
    if not filtered:
        raise RuntimeError("The filtered checkpoint state is empty")
    return filtered


def load_checkpoint_file(path: str | Path, *, model_only: bool = False) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Inference only needs model tensors. mmap keeps optimizer tensors lazy so
    # a multi-gigabyte training checkpoint is not read in full from NFS.
    if model_only:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        except (TypeError, RuntimeError):
            payload = torch.load(checkpoint_path, map_location="cpu")
    else:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    payload["checkpoint_path"] = str(checkpoint_path)
    return payload


def load_model_weights(module: Module, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    incompatible = module.load_state_dict(payload["model"], strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        joined = ", ".join(unexpected[:10])
        raise RuntimeError(f"Checkpoint contains unexpected model keys: {joined}")
    return missing, unexpected


def save_checkpoint(
    path: str | Path,
    module: Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: Any,
    global_step: int,
    epoch: int,
    trainable_names: set[str],
    class_to_idx: dict[str, int],
    config_dict: dict[str, Any],
    context: DistributedContext,
    model_stack: dict[str, Any] | None = None,
) -> None:
    """Persist a resumable delta together with its verified parent model stack."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(module, FullyShardedDataParallel):
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(module, StateDictType.FULL_STATE_DICT, state_config):
            full_state = module.state_dict()
            optimizer_state = FullyShardedDataParallel.optim_state_dict(module, optimizer)
    else:
        full_state = module.state_dict()
        optimizer_state = optimizer.state_dict()
    if model_stack is not None:
        parent_fingerprint = model_stack.get("parent_fingerprint")
        if not isinstance(parent_fingerprint, str) or len(parent_fingerprint) != 64:
            raise ValueError("Checkpoint model_stack must declare a 64-character parent_fingerprint")
    if context.is_main:
        filtered_state = filter_model_state(full_state, trainable_names)
        payload = {
            "format_version": 2,
            "model": filtered_state,
            "optimizer": optimizer_state,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "trainable_names": sorted(trainable_names),
            "class_to_idx": dict(class_to_idx),
            "config": json.loads(json.dumps(config_dict)),
            "model_stack": json.loads(json.dumps(model_stack)) if model_stack is not None else None,
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output_path)
    synchronize(context)
    return None


def restore_optimizer_state(
    module: Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
    context: DistributedContext,
) -> None:
    full_state = payload.get("optimizer")
    if full_state is None:
        return None
    if isinstance(module, FullyShardedDataParallel):
        source = full_state if context.is_main else None
        sharded = FullyShardedDataParallel.scatter_full_optim_state_dict(source, module)
        optimizer.load_state_dict(sharded)
    else:
        optimizer.load_state_dict(full_state)
    # FSDP can materialize scalar gated parameters as a one-element tensor
    # while the saved Adam moments remain 0-D scalars (or the reverse).  Align
    # only the moment buffers; the scalar ``step`` state must stay scalar.
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor):
            continue
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if not torch.is_tensor(value) or value.shape == parameter.shape:
                continue
            if value.numel() != parameter.numel():
                raise RuntimeError(
                    f"Optimizer state {name} shape {tuple(value.shape)} does not "
                    f"match parameter shape {tuple(parameter.shape)}"
                )
            state[name] = value.reshape(parameter.shape).to(
                device=parameter.device,
                dtype=value.dtype,
            )
    return None
