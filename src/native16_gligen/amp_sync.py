"""Distributed synchronization helpers for CUDA AMP training."""

from __future__ import annotations

import torch
import torch.distributed as dist


def synchronize_grad_scaler_overflow(
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> bool:
    """Make GradScaler's overflow decision identical on every rank.

    ``GradScaler.unscale_(optimizer)`` records non-finite gradients locally.
    FSDP requires every rank to either execute or skip ``optimizer.step()``;
    otherwise Adam's per-parameter ``step`` counters diverge and a later full
    optimizer-state checkpoint fails. This function globally ORs the local
    overflow flags and writes the result back into GradScaler before
    ``scaler.step(optimizer)``.

    This targets the pinned PyTorch 2.1 GradScaler state layout. A loud failure
    is preferable to silently allowing optimizer states to diverge if that
    internal contract changes in a future PyTorch release.
    """

    if not scaler.is_enabled():
        return False

    optimizer_state = scaler._per_optimizer_states.get(id(optimizer))  # noqa: SLF001
    if optimizer_state is None:
        raise RuntimeError(
            "GradScaler overflow synchronization must run after "
            "scaler.unscale_(optimizer)"
        )
    found_inf_per_device = optimizer_state.get("found_inf_per_device")
    if not found_inf_per_device:
        raise RuntimeError("GradScaler did not record found_inf_per_device")

    global_found_inf = torch.zeros((), dtype=torch.float32, device=device)
    for found_inf in found_inf_per_device.values():
        global_found_inf.copy_(
            torch.maximum(
                global_found_inf,
                found_inf.detach().to(device=device, dtype=torch.float32),
            )
        )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_found_inf, op=dist.ReduceOp.MAX)

    for found_inf in found_inf_per_device.values():
        found_inf.copy_(
            global_found_inf.to(device=found_inf.device, dtype=found_inf.dtype)
        )
    return bool(global_found_inf.item())
