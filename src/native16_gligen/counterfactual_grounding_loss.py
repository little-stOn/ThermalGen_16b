from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CounterfactualGroundingStats:
    true_advantage: torch.Tensor
    shifted_advantage: torch.Tensor
    outside_consistency: torch.Tensor
    negative_iou: torch.Tensor


def _same_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    intersection_min = torch.maximum(first[..., :2], second[..., :2])
    intersection_max = torch.minimum(first[..., 2:], second[..., 2:])
    intersection_size = (intersection_max - intersection_min).clamp_min(0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_size = (first[..., 2:] - first[..., :2]).clamp_min(0.0)
    second_size = (second[..., 2:] - second[..., :2]).clamp_min(0.0)
    first_area = first_size[..., 0] * first_size[..., 1]
    second_area = second_size[..., 0] * second_size[..., 1]
    return intersection / (first_area + second_area - intersection).clamp_min(1.0e-8)


def sample_counterfactual_boxes(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    candidates: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relocate each box while preserving its size and choosing the lowest-IoU proposal."""
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes with shape [B,M,4], got {tuple(boxes.shape)}")
    if candidates < 1:
        raise ValueError("candidates must be positive")
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(1.0e-5, 1.0)
    half_size = size * 0.5
    span = (1.0 - size).clamp_min(0.0)
    random_centers = half_size.unsqueeze(2) + torch.rand(
        *boxes.shape[:2], candidates, 2, device=boxes.device, dtype=boxes.dtype
    ) * span.unsqueeze(2)
    candidate_boxes = torch.cat(
        [random_centers - half_size.unsqueeze(2), random_centers + half_size.unsqueeze(2)],
        dim=-1,
    ).clamp(0.0, 1.0)
    original = boxes.unsqueeze(2).expand_as(candidate_boxes)
    candidate_iou = _same_box_iou(original, candidate_boxes)
    best_index = candidate_iou.argmin(dim=2)
    gather_index = best_index[..., None, None].expand(-1, -1, 1, 4)
    relocated = candidate_boxes.gather(2, gather_index).squeeze(2)
    relocated = torch.where(box_mask[..., None], relocated, boxes)
    best_iou = candidate_iou.gather(2, best_index[..., None]).squeeze(2)
    return relocated, best_iou


def build_soft_object_core(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    height: int,
    width: int,
    core_fraction: float = 0.82,
) -> torch.Tensor:
    """Build an elliptical core that smoothly reaches zero before every box edge."""
    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    half_size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0e-6) * (
        0.5 * float(core_fraction)
    )
    x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
    y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
    dx = (x.view(1, 1, 1, width) - center[..., 0, None, None]) / half_size[
        ..., 0, None, None
    ]
    dy = (y.view(1, 1, height, 1) - center[..., 1, None, None]) / half_size[
        ..., 1, None, None
    ]
    radius_squared = dx.square() + dy.square()
    object_core = (1.0 - radius_squared).clamp(0.0, 1.0).square()
    object_core = object_core * box_mask[..., None, None].to(dtype=object_core.dtype)

    empty = (object_core.flatten(2).sum(dim=-1) == 0) & box_mask
    flat_index = (
        (center[..., 1] * height).long().clamp(0, height - 1) * width
        + (center[..., 0] * width).long().clamp(0, width - 1)
    )
    nearest = F.one_hot(flat_index, num_classes=height * width).to(object_core.dtype)
    nearest = nearest.view(*boxes.shape[:2], height, width)
    object_core = torch.where(empty[..., None, None], nearest, object_core)
    return object_core.amax(dim=1)


def _weighted_spatial_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    numerator = (values * weights).flatten(1).sum(dim=1)
    denominator = weights.flatten(1).sum(dim=1).clamp_min(1.0e-6)
    return numerator / denominator


def compute_counterfactual_grounding_loss(
    prediction: torch.Tensor,
    negative_prediction: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor,
    negative_boxes: torch.Tensor,
    box_mask: torch.Tensor,
    negative_iou: torch.Tensor,
    core_fraction: float = 0.82,
    relative_margin: float = 0.05,
    temperature: float = 0.10,
    outside_weight: float = 0.50,
) -> tuple[torch.Tensor, CounterfactualGroundingStats]:
    if temperature <= 0.0 or relative_margin < 0.0 or outside_weight < 0.0:
        raise ValueError("Invalid counterfactual loss parameters")
    height, width = prediction.shape[-2:]
    true_core = build_soft_object_core(
        boxes, box_mask, height, width, core_fraction=core_fraction
    )
    shifted_core = build_soft_object_core(
        negative_boxes, box_mask, height, width, core_fraction=core_fraction
    )
    correct_error = (prediction.float() - target.float()).square().mean(dim=1)
    negative_error = (negative_prediction.float() - target.float()).square().mean(dim=1)

    correct_true = _weighted_spatial_mean(correct_error, true_core)
    negative_true = _weighted_spatial_mean(negative_error, true_core)
    correct_shifted = _weighted_spatial_mean(correct_error, shifted_core)
    negative_shifted = _weighted_spatial_mean(negative_error, shifted_core)
    true_scale = (0.5 * (correct_true + negative_true)).detach().clamp_min(1.0e-6)
    shifted_scale = (0.5 * (correct_shifted + negative_shifted)).detach().clamp_min(1.0e-6)
    true_relative = (correct_true - negative_true) / true_scale
    shifted_relative = (correct_shifted - negative_shifted) / shifted_scale
    true_rank = F.softplus((true_relative + relative_margin) / temperature) * temperature
    shifted_rank = F.softplus(
        (shifted_relative + relative_margin) / temperature
    ) * temperature

    valid_sample = box_mask.any(dim=1).to(dtype=true_rank.dtype)
    valid_denominator = valid_sample.sum().clamp_min(1.0)
    rank_loss = ((true_rank + shifted_rank) * 0.5 * valid_sample).sum() / valid_denominator

    affected_core = torch.maximum(true_core, shifted_core)
    preserve_region = (1.0 - affected_core).clamp(0.0, 1.0)
    prediction_difference = (
        prediction.float() - negative_prediction.float()
    ).square().mean(dim=1)
    outside_per_sample = _weighted_spatial_mean(prediction_difference, preserve_region)
    outside_consistency = (outside_per_sample * valid_sample).sum() / valid_denominator
    loss = rank_loss + float(outside_weight) * outside_consistency

    valid_box = box_mask.to(dtype=negative_iou.dtype)
    valid_box_denominator = valid_box.sum().clamp_min(1.0)
    mean_negative_iou = (negative_iou * valid_box).sum() / valid_box_denominator
    stats = CounterfactualGroundingStats(
        true_advantage=((-true_relative) * valid_sample).sum().detach() / valid_denominator,
        shifted_advantage=((-shifted_relative) * valid_sample).sum().detach()
        / valid_denominator,
        outside_consistency=outside_consistency.detach(),
        negative_iou=mean_negative_iou.detach(),
    )
    return loss, stats
