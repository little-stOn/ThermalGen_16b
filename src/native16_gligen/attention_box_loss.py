from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class AttentionBoxLossStats:
    layer_count: int
    valid_boxes: torch.Tensor
    inside_mass: torch.Tensor
    core_density: torch.Tensor
    boundary_density: torch.Tensor
    ring_density: torch.Tensor


class VisualObjectAttentionProcessor:
    """Preserve the native attention output while recording compact visual-object logits."""

    def __init__(self, base_processor: Any, recorder: "GroundingAttentionRecorder") -> None:
        self.base_processor = base_processor
        self.recorder = recorder

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        output = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )
        object_count = self.recorder.object_count
        if object_count <= 0 or encoder_hidden_states is not None or hidden_states.ndim != 3:
            return output
        visual_count = hidden_states.shape[1] - object_count
        side = math.isqrt(visual_count)
        if side * side != visual_count or side < self.recorder.min_resolution:
            return output

        query = attn.to_q(hidden_states[:, :visual_count])
        key = attn.to_k(hidden_states[:, visual_count:])
        batch_size = query.shape[0]
        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch_size, visual_count, attn.heads, head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, object_count, attn.heads, head_dim).permute(0, 2, 1, 3)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        logits = torch.einsum("bhqd,bhkd->bhqk", query, key) * float(attn.scale)
        self.recorder.maps.append(logits.mean(dim=1).permute(0, 2, 1).reshape(batch_size, object_count, side, side))
        return output


class GroundingAttentionRecorder:
    def __init__(self, min_resolution: int = 16) -> None:
        self.min_resolution = int(min_resolution)
        self.object_count = 0
        self.maps: list[torch.Tensor] = []
        self.layer_names: list[str] = []

    def clear(self, object_count: int) -> None:
        self.object_count = int(object_count)
        self.maps.clear()

    def compute_loss(
        self,
        boxes: torch.Tensor,
        box_mask: torch.Tensor,
        small_area_reference: float = 0.01,
        max_small_weight: float = 3.0,
        core_fraction: float = 0.70,
        ring_scale: float = 1.35,
        density_margin: float = 0.35,
        ring_weight: float = 0.50,
        boundary_weight: float = 0.25,
        epsilon: float = 1.0e-6,
    ) -> tuple[torch.Tensor, AttentionBoxLossStats]:
        if not self.maps:
            raise RuntimeError("No grounding attention maps were recorded")
        if not 0.0 < core_fraction <= 1.0:
            raise ValueError("core_fraction must be in (0, 1]")
        if ring_scale < 1.0:
            raise ValueError("ring_scale must be at least 1")
        if ring_weight < 0.0 or boundary_weight < 0.0:
            raise ValueError("attention density weights must be non-negative")
        valid = box_mask.to(dtype=torch.bool)
        valid_count = valid.sum().to(dtype=torch.float32)
        layer_losses: list[torch.Tensor] = []
        layer_inside: list[torch.Tensor] = []
        layer_core_density: list[torch.Tensor] = []
        layer_boundary_density: list[torch.Tensor] = []
        layer_ring_density: list[torch.Tensor] = []
        areas = (
            (boxes[..., 2] - boxes[..., 0]).clamp_min(epsilon)
            * (boxes[..., 3] - boxes[..., 1]).clamp_min(epsilon)
        )
        small_weights = torch.sqrt(float(small_area_reference) / areas)
        small_weights = small_weights.clamp(1.0, float(max_small_weight))

        for logits in self.maps:
            height, width = logits.shape[-2:]
            dtype = logits.dtype
            x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
            y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
            inside_x = (x.view(1, 1, 1, width) >= boxes[..., 0, None, None]) & (
                x.view(1, 1, 1, width) <= boxes[..., 2, None, None]
            )
            inside_y = (y.view(1, 1, height, 1) >= boxes[..., 1, None, None]) & (
                y.view(1, 1, height, 1) <= boxes[..., 3, None, None]
            )
            inside = (inside_x & inside_y).to(dtype=dtype)
            # Extremely small boxes must still own at least their nearest spatial cell.
            empty = inside.flatten(2).sum(dim=-1) == 0
            if empty.any():
                center_x = (((boxes[..., 0] + boxes[..., 2]) * 0.5 * width).long()).clamp(0, width - 1)
                center_y = (((boxes[..., 1] + boxes[..., 3]) * 0.5 * height).long()).clamp(0, height - 1)
                rows, objects = empty.nonzero(as_tuple=True)
                inside[rows, objects, center_y[rows, objects], center_x[rows, objects]] = 1.0

            center_x_value = (boxes[..., 0] + boxes[..., 2]) * 0.5
            center_y_value = (boxes[..., 1] + boxes[..., 3]) * 0.5
            half_width = (boxes[..., 2] - boxes[..., 0]).clamp_min(epsilon) * 0.5
            half_height = (boxes[..., 3] - boxes[..., 1]).clamp_min(epsilon) * 0.5

            core_half_width = half_width * float(core_fraction)
            core_half_height = half_height * float(core_fraction)
            core_x = (
                (x.view(1, 1, 1, width) >= (center_x_value - core_half_width)[..., None, None])
                & (x.view(1, 1, 1, width) <= (center_x_value + core_half_width)[..., None, None])
            )
            core_y = (
                (y.view(1, 1, height, 1) >= (center_y_value - core_half_height)[..., None, None])
                & (y.view(1, 1, height, 1) <= (center_y_value + core_half_height)[..., None, None])
            )
            core = (core_x & core_y).to(dtype=dtype) * inside
            empty_core = core.flatten(2).sum(dim=-1) == 0
            if empty_core.any():
                center_x_index = (center_x_value * width).long().clamp(0, width - 1)
                center_y_index = (center_y_value * height).long().clamp(0, height - 1)
                rows, objects = empty_core.nonzero(as_tuple=True)
                core[rows, objects, center_y_index[rows, objects], center_x_index[rows, objects]] = 1.0
            boundary = (inside - core).clamp_min(0.0)

            ring_half_width = half_width * float(ring_scale)
            ring_half_height = half_height * float(ring_scale)
            expanded_x = (
                (x.view(1, 1, 1, width) >= (center_x_value - ring_half_width)[..., None, None])
                & (x.view(1, 1, 1, width) <= (center_x_value + ring_half_width)[..., None, None])
            )
            expanded_y = (
                (y.view(1, 1, height, 1) >= (center_y_value - ring_half_height)[..., None, None])
                & (y.view(1, 1, height, 1) <= (center_y_value + ring_half_height)[..., None, None])
            )
            expanded = (expanded_x & expanded_y).to(dtype=dtype)
            ring = (expanded - inside).clamp_min(0.0)

            spatial_prob = logits.float().flatten(2).softmax(dim=-1)
            flat_inside = inside.float().flatten(2)
            flat_core = core.float().flatten(2)
            flat_boundary = boundary.float().flatten(2)
            flat_ring = ring.float().flatten(2)
            inside_mass = (spatial_prob * flat_inside).sum(dim=-1).clamp_min(epsilon)
            core_mass = (spatial_prob * flat_core).sum(dim=-1)
            boundary_mass = (spatial_prob * flat_boundary).sum(dim=-1)
            ring_mass = (spatial_prob * flat_ring).sum(dim=-1)

            core_density = core_mass / flat_core.sum(dim=-1).clamp_min(1.0)
            boundary_cells = flat_boundary.sum(dim=-1)
            ring_cells = flat_ring.sum(dim=-1)
            boundary_density = boundary_mass / boundary_cells.clamp_min(1.0)
            ring_density = ring_mass / ring_cells.clamp_min(1.0)
            inside_density = inside_mass / flat_inside.sum(dim=-1).clamp_min(1.0)

            inclusion_loss = -inside_mass.log() * small_weights.float()
            ring_contrast = torch.nn.functional.softplus(
                torch.log(ring_density.clamp_min(epsilon))
                - torch.log(inside_density.clamp_min(epsilon))
                + float(density_margin)
            )
            boundary_contrast = torch.nn.functional.softplus(
                torch.log(boundary_density.clamp_min(epsilon))
                - torch.log(core_density.clamp_min(epsilon))
                + float(density_margin)
            )
            ring_contrast = ring_contrast * (ring_cells > 0).float()
            boundary_contrast = boundary_contrast * (boundary_cells > 0).float()
            weighted = (
                inclusion_loss
                + float(ring_weight) * ring_contrast
                + float(boundary_weight) * boundary_contrast
            )
            denominator = valid_count.clamp_min(1.0)
            layer_losses.append((weighted * valid.float()).sum() / denominator)
            layer_inside.append((inside_mass * valid.float()).sum() / denominator)
            layer_core_density.append((core_density * valid.float()).sum() / denominator)
            layer_boundary_density.append((boundary_density * valid.float()).sum() / denominator)
            layer_ring_density.append((ring_density * valid.float()).sum() / denominator)

        if valid_count.item() == 0:
            loss = sum(attention_map.sum() * 0.0 for attention_map in self.maps)
            mean_inside = loss.detach()
            mean_core_density = loss.detach()
            mean_boundary_density = loss.detach()
            mean_ring_density = loss.detach()
        else:
            loss = torch.stack(layer_losses).mean()
            mean_inside = torch.stack(layer_inside).mean().detach()
            mean_core_density = torch.stack(layer_core_density).mean().detach()
            mean_boundary_density = torch.stack(layer_boundary_density).mean().detach()
            mean_ring_density = torch.stack(layer_ring_density).mean().detach()
        stats = AttentionBoxLossStats(
            layer_count=len(self.maps),
            valid_boxes=valid_count.detach(),
            inside_mass=mean_inside,
            core_density=mean_core_density,
            boundary_density=mean_boundary_density,
            ring_density=mean_ring_density,
        )
        return loss, stats


def install_grounding_attention_recorder(
    unet: nn.Module,
    min_resolution: int = 16,
) -> GroundingAttentionRecorder:
    recorder = GroundingAttentionRecorder(min_resolution=min_resolution)
    for name, module in unet.named_modules():
        if ".fuser" not in name or not hasattr(module, "attn"):
            continue
        attention = module.attn
        attention.set_processor(VisualObjectAttentionProcessor(attention.processor, recorder))
        recorder.layer_names.append(name)
    if not recorder.layer_names:
        raise RuntimeError("No GLIGEN fuser attention modules were found")
    return recorder
