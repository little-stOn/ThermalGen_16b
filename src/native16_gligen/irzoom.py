from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class InstanceCrop:
    image: Image.Image
    box_xyxy: torch.Tensor
    crop_xyxy: tuple[int, int, int, int]
    source_side_px: float
    normalized_side_px: float


@dataclass(frozen=True)
class InstanceCropPlan:
    crop_xyxy: tuple[int, int, int, int]
    local_box_xyxy: tuple[float, float, float, float]
    source_side_px: float


def box_side_at_resolution(box_xyxy: torch.Tensor, width: int, height: int, resolution: int = 512) -> float:
    box = box_xyxy.to(torch.float32)
    area_fraction = (
        (box[2] - box[0]).clamp_min(0.0)
        * (box[3] - box[1]).clamp_min(0.0)
        / max(float(width * height), 1.0)
    )
    return float(area_fraction.sqrt() * float(resolution))


def choose_instance_index(
    boxes: torch.Tensor,
    width: int,
    height: int,
    allowed: torch.Tensor,
    small_reference: float = 32.0,
    small_weight: float = 4.0,
    medium_weight: float = 1.0,
    large_weight: float = 0.25,
) -> int:
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or boxes.shape[0] == 0:
        raise ValueError("boxes must be a non-empty [N, 4] tensor")
    if allowed.shape != (boxes.shape[0],):
        raise ValueError("allowed must have one boolean entry per box")
    if not bool(allowed.any()):
        raise ValueError("sample contains no allowed IR-Zoom target")
    weights = []
    for box, keep in zip(boxes, allowed):
        if not bool(keep):
            weights.append(0.0)
            continue
        side = box_side_at_resolution(box, width, height)
        if side <= small_reference:
            weights.append(float(small_weight))
        elif side <= 3.0 * small_reference:
            weights.append(float(medium_weight))
        else:
            weights.append(float(large_weight))
    return int(random.choices(range(len(weights)), weights=weights, k=1)[0])


def _crop_origin(
    center: float,
    side: int,
    limit: int,
    target_min: float,
    target_max: float,
    jitter_fraction: float,
) -> int:
    desired = center - side * 0.5 + random.uniform(-jitter_fraction, jitter_fraction) * side
    low = max(0.0, target_max - side)
    high = min(float(limit - side), target_min)
    if low > high:
        desired = center - side * 0.5
        low, high = 0.0, float(max(0, limit - side))
    return int(round(min(max(desired, low), high)))


def make_scale_normalized_crop(
    image: Image.Image,
    box_xyxy: torch.Tensor,
    output_size: int = 512,
    target_long_side_min: float = 112.0,
    target_long_side_max: float = 160.0,
    min_context_scale: float = 2.5,
    center_jitter: float = 0.12,
) -> InstanceCrop:
    width, height = image.size
    box = box_xyxy.to(torch.float32)
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width, box_height = x2 - x1, y2 - y1
    long_side = max(box_width, box_height)
    if long_side < 1.0:
        raise ValueError("target box is degenerate")
    if not 0.0 < target_long_side_min <= target_long_side_max < output_size:
        raise ValueError("invalid normalized target-side range")
    desired_target_side = random.uniform(target_long_side_min, target_long_side_max)
    crop_side = max(long_side * output_size / desired_target_side, long_side * min_context_scale)
    crop_side = int(math.ceil(min(crop_side, float(min(width, height)))))
    crop_side = max(crop_side, int(math.ceil(long_side)))
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    left = _crop_origin(center_x, crop_side, width, x1, x2, center_jitter)
    top = _crop_origin(center_y, crop_side, height, y1, y2, center_jitter)
    right, bottom = left + crop_side, top + crop_side
    crop = image.crop((left, top, right, bottom)).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    scale = output_size / float(crop_side)
    transformed = torch.tensor(
        [
            (x1 - left) * scale / output_size,
            (y1 - top) * scale / output_size,
            (x2 - left) * scale / output_size,
            (y2 - top) * scale / output_size,
        ],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    return InstanceCrop(
        image=crop,
        box_xyxy=transformed,
        crop_xyxy=(left, top, right, bottom),
        source_side_px=box_side_at_resolution(box, width, height),
        normalized_side_px=long_side * scale,
    )


def plan_scale_normalized_crop(
    box_xyxy: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
    target_long_side: float = 136.0,
    min_context_scale: float = 2.5,
) -> InstanceCropPlan:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    x1, x2 = sorted((x1 * width, x2 * width))
    y1, y2 = sorted((y1 * height, y2 * height))
    long_side = max(x2 - x1, y2 - y1)
    if long_side < 1.0:
        raise ValueError("target box is degenerate")
    side = max(long_side * 512.0 / target_long_side, long_side * min_context_scale)
    side = max(int(math.ceil(long_side)), int(math.ceil(min(side, min(width, height)))))
    left = int(round((x1 + x2 - side) * 0.5))
    top = int(round((y1 + y2 - side) * 0.5))
    left = min(max(0, left), max(0, width - side))
    top = min(max(0, top), max(0, height - side))
    scale = 1.0 / float(side)
    local_box = (
        max(0.0, (x1 - left) * scale),
        max(0.0, (y1 - top) * scale),
        min(1.0, (x2 - left) * scale),
        min(1.0, (y2 - top) * scale),
    )
    source_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    return InstanceCropPlan(
        crop_xyxy=(left, top, left + side, top + side),
        local_box_xyxy=local_box,
        source_side_px=box_side_at_resolution(source_box, width, height),
    )


def predict_x0_from_epsilon(
    noisy_latents: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_cumprod: torch.Tensor | float,
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha_cumprod, device=noisy_latents.device, dtype=noisy_latents.dtype)
    return (noisy_latents - (1.0 - alpha).clamp_min(0.0).sqrt() * epsilon) / alpha.sqrt().clamp_min(1e-6)


def epsilon_from_x0(
    noisy_latents: torch.Tensor,
    x0: torch.Tensor,
    alpha_cumprod: torch.Tensor | float,
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha_cumprod, device=noisy_latents.device, dtype=noisy_latents.dtype)
    return (noisy_latents - alpha.sqrt() * x0) / (1.0 - alpha).clamp_min(0.0).sqrt().clamp_min(1e-6)


def feather_box_mask(
    box_xyxy: torch.Tensor,
    height: int,
    width: int,
    expand_fraction: float = 0.2,
    blur_sigma: float = 1.5,
) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("mask dimensions must be positive")
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    dx, dy = (x2 - x1) * expand_fraction, (y2 - y1) * expand_fraction
    left = max(0, min(width - 1, int(math.floor((x1 - dx) * width))))
    right = max(left + 1, min(width, int(math.ceil((x2 + dx) * width))))
    top = max(0, min(height - 1, int(math.floor((y1 - dy) * height))))
    bottom = max(top + 1, min(height, int(math.ceil((y2 + dy) * height))))
    mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
    mask[:, :, top:bottom, left:right] = 1.0
    if blur_sigma <= 0:
        return mask
    radius = max(1, int(math.ceil(3.0 * blur_sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / blur_sigma).square())
    kernel = kernel / kernel.sum()
    mask = F.pad(mask, (radius, radius, radius, radius), mode="replicate")
    mask = F.conv2d(mask, kernel.view(1, 1, 1, -1))
    mask = F.conv2d(mask, kernel.view(1, 1, -1, 1))
    return mask.clamp(0.0, 1.0)


def irzoom_step_weight(
    step_index: int,
    total_steps: int,
    source_side_px: float,
    max_weight: float,
    active_start: float = 0.2,
    active_end: float = 0.9,
    small_reference: float = 32.0,
) -> float:
    if total_steps < 2 or not 0 <= step_index < total_steps:
        raise ValueError("invalid denoising step")
    progress = step_index / float(total_steps - 1)
    if progress <= active_start or progress >= active_end:
        return 0.0
    phase = (progress - active_start) / (active_end - active_start)
    time_gate = math.sin(math.pi * phase) ** 2
    scale_gate = min(1.0, max(0.0, small_reference / max(source_side_px, 1.0) - 0.25))
    return float(max_weight) * time_gate * scale_gate


def normalized_masked_average(
    base: torch.Tensor,
    proposals: list[torch.Tensor],
    masks: list[torch.Tensor],
    weights: list[float],
) -> torch.Tensor:
    if not (len(proposals) == len(masks) == len(weights)):
        raise ValueError("proposal, mask, and weight counts must match")
    numerator = base.clone()
    denominator = torch.ones_like(base[:, :1])
    for proposal, mask, weight in zip(proposals, masks, weights):
        if proposal.shape != base.shape or mask.shape != base[:, :1].shape:
            raise ValueError("all proposals and masks must match the base tensor")
        alpha = mask.to(device=base.device, dtype=base.dtype) * float(weight)
        numerator = numerator + proposal.to(base) * alpha
        denominator = denominator + alpha
    return numerator / denominator.clamp_min(1e-6)
