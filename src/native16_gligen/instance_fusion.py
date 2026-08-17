from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class InstanceFusionContext:
    boxes: torch.Tensor | None = None
    masks: torch.Tensor | None = None


def _match_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.shape[0] == batch_size:
        return value
    if batch_size % value.shape[0] != 0:
        raise ValueError(
            f"Cannot broadcast instance batch {value.shape[0]} to hidden batch {batch_size}"
        )
    repeats = batch_size // value.shape[0]
    return value.repeat_interleave(repeats, dim=0)


def build_soft_instance_masks(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    side: int,
    feather_power: float = 2.0,
) -> torch.Tensor:
    """Elliptical masks that smoothly become zero at the bbox boundary."""
    if side < 1 or feather_power <= 0:
        raise ValueError("side and feather_power must be positive")
    center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
    half_size = 0.5 * (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0e-6)
    axis = (torch.arange(side, device=boxes.device, dtype=boxes.dtype) + 0.5) / side
    dx = (
        axis.view(1, 1, 1, side) - center[..., 0, None, None]
    ) / half_size[..., 0, None, None]
    dy = (
        axis.view(1, 1, side, 1) - center[..., 1, None, None]
    ) / half_size[..., 1, None, None]
    radius_squared = dx.square() + dy.square()
    masks = (1.0 - radius_squared).clamp(0.0, 1.0).pow(float(feather_power))
    masks = masks * box_mask[..., None, None].to(dtype=masks.dtype)

    # A sub-cell box still receives one differentiable feature cell. This is
    # essential for FLIR pedestrians and distant vehicles at low resolutions.
    empty = (masks.flatten(2).sum(dim=-1) == 0) & box_mask
    flat_index = (
        (center[..., 1] * side).long().clamp(0, side - 1) * side
        + (center[..., 0] * side).long().clamp(0, side - 1)
    )
    nearest = F.one_hot(flat_index, num_classes=side * side).to(masks.dtype)
    nearest = nearest.view(*boxes.shape[:2], side, side)
    return torch.where(empty[..., None, None], nearest, masks)


def compute_scale_weights(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    feature_side: int,
    preferred_extent: float = 3.0,
    log_sigma: float = 1.25,
    minimum: float = 0.05,
) -> torch.Tensor:
    """Route each instance to layers where it occupies a few feature cells."""
    if preferred_extent <= 0 or log_sigma <= 0 or not 0 <= minimum <= 1:
        raise ValueError("Invalid scale-routing parameters")
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0e-8)
    extent = size.prod(dim=-1).sqrt() * float(feature_side)
    distance = torch.log2(extent.clamp_min(1.0e-4) / float(preferred_extent))
    weights = torch.exp(-0.5 * (distance / float(log_sigma)).square())
    weights = float(minimum) + (1.0 - float(minimum)) * weights
    return weights * box_mask.to(dtype=weights.dtype)


class ScaleAwareInstanceFusion(nn.Module):
    """MIGC-style soft instance aggregation with ScaleU-style scale routing."""

    def __init__(
        self,
        query_dim: int,
        object_dim: int,
        rank: int = 48,
        feather_power: float = 2.0,
        preferred_extent: float = 3.0,
        log_sigma: float = 1.25,
        minimum_scale_weight: float = 0.05,
        strength: float = 1.0,
    ) -> None:
        super().__init__()
        rank = min(int(rank), int(query_dim))
        if rank < 1:
            raise ValueError("rank must be positive")
        self.norm = nn.LayerNorm(query_dim)
        self.query = nn.Linear(query_dim, rank, bias=False)
        self.object_affine = nn.Linear(object_dim, 2 * rank)
        self.output = nn.Linear(rank, query_dim)
        self.feather_power = float(feather_power)
        self.preferred_extent = float(preferred_extent)
        self.log_sigma = float(log_sigma)
        self.minimum_scale_weight = float(minimum_scale_weight)
        self.strength = float(strength)
        if self.strength < 0.0:
            raise ValueError("strength must be non-negative")

        # Zero initialization makes installation exactly preserve the existing
        # GLIGEN output before training.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        objects: torch.Tensor,
        context: InstanceFusionContext,
    ) -> torch.Tensor:
        if context.boxes is None or context.masks is None:
            return torch.zeros_like(hidden_states)
        tokens = int(hidden_states.shape[1])
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError(f"Instance fusion expects a square feature map, got {tokens} tokens")

        boxes = _match_batch(context.boxes, hidden_states.shape[0]).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        box_mask = _match_batch(context.masks, hidden_states.shape[0]).to(
            device=hidden_states.device, dtype=torch.bool
        )
        objects = _match_batch(objects, hidden_states.shape[0])
        spatial = build_soft_instance_masks(
            boxes, box_mask, side, feather_power=self.feather_power
        ).flatten(2)
        routing = compute_scale_weights(
            boxes,
            box_mask,
            side,
            preferred_extent=self.preferred_extent,
            log_sigma=self.log_sigma,
            minimum=self.minimum_scale_weight,
        )
        spatial = spatial * routing[..., None]
        denominator = spatial.sum(dim=1).clamp_min(1.0e-6)
        normalized = spatial / denominator[:, None, :]
        coverage = spatial.amax(dim=1).clamp(0.0, 1.0)

        affine = self.object_affine(objects.to(dtype=hidden_states.dtype))
        gamma, beta = affine.chunk(2, dim=-1)
        pixel_gamma = torch.einsum("bmn,bmr->bnr", normalized, gamma)
        pixel_beta = torch.einsum("bmn,bmr->bnr", normalized, beta)
        query = self.query(self.norm(hidden_states))
        fused = F.silu(query * (1.0 + pixel_gamma) + pixel_beta)
        return self.output(fused) * coverage[..., None] * self.strength


def install_instance_fusion(unet: nn.Module, config: Any) -> int:
    """Attach adapters without renaming any existing GLIGEN checkpoint keys."""
    if config is None or not bool(getattr(config, "enabled", False)):
        return 0
    existing = getattr(unet, "_instance_fusion_context", None)
    if existing is not None:
        return sum(
            1 for module in unet.modules() if hasattr(module, "instance_fusion")
        )

    context = InstanceFusionContext()
    object.__setattr__(unet, "_instance_fusion_context", context)
    position_net = getattr(unet, "position_net", None)
    if position_net is None:
        raise ValueError("Instance fusion requires a GLIGEN position_net")
    original_position_forward = position_net.forward.__func__

    def position_forward(module: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        boxes = kwargs.get("boxes")
        masks = kwargs.get("masks")
        if boxes is None or masks is None:
            raise ValueError("GLIGEN position_net did not receive boxes and masks")
        context.boxes = boxes
        context.masks = masks.to(dtype=torch.bool)
        return original_position_forward(module, *args, **kwargs)

    position_net.forward = types.MethodType(position_forward, position_net)

    installed = 0
    target_device = next(unet.parameters()).device
    target_dtype = next(unet.parameters()).dtype
    for module in list(unet.modules()):
        if module is position_net or not (
            hasattr(module, "linear")
            and hasattr(module, "alpha_attn")
            and hasattr(module, "alpha_dense")
            and module.__class__.__name__ == "GatedSelfAttentionDense"
        ):
            continue
        if hasattr(module, "instance_fusion"):
            continue
        adapter = ScaleAwareInstanceFusion(
            query_dim=int(module.linear.out_features),
            object_dim=int(module.linear.in_features),
            rank=int(getattr(config, "rank", 48)),
            feather_power=float(getattr(config, "feather_power", 2.0)),
            preferred_extent=float(getattr(config, "preferred_extent", 3.0)),
            log_sigma=float(getattr(config, "log_sigma", 1.25)),
            minimum_scale_weight=float(getattr(config, "minimum_scale_weight", 0.05)),
            strength=float(getattr(config, "strength", 1.0)),
        ).to(device=target_device, dtype=target_dtype)
        module.add_module("instance_fusion", adapter)
        original_fuser_forward = module.forward.__func__

        def fuser_forward(
            current: nn.Module,
            x: torch.Tensor,
            objs: torch.Tensor,
            _original: Any = original_fuser_forward,
        ) -> torch.Tensor:
            base = _original(current, x, objs)
            return base + current.instance_fusion(x, objs, context)

        module.forward = types.MethodType(fuser_forward, module)
        installed += 1
    if installed == 0:
        raise RuntimeError("No GLIGEN fuser layers were found for instance fusion")
    return installed


def compute_instance_normalized_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    feather_power: float = 2.0,
) -> torch.Tensor:
    """Per-sample soft-core error normalized independently of object area."""
    side_y, side_x = prediction.shape[-2:]
    if side_y != side_x:
        raise ValueError("Instance-normalized loss expects square latents")
    masks = build_soft_instance_masks(
        boxes.to(dtype=prediction.dtype), box_mask, side_y, feather_power=feather_power
    ).amax(dim=1)
    error = (prediction.float() - target.float()).square().mean(dim=1)
    numerator = (error * masks.float()).flatten(1).sum(dim=1)
    denominator = masks.float().flatten(1).sum(dim=1).clamp_min(1.0)
    valid = box_mask.any(dim=1).to(dtype=numerator.dtype)
    return ((numerator / denominator) * valid).sum() / valid.sum().clamp_min(1.0)
