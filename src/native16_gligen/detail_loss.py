"""Pixel-space detail losses for the infrared GLIGEN ablation.

The losses operate on the predicted clean image (x0), not on the noisy
latent.  Boxes are normalized xyxy coordinates and the target image uses the
same [-1, 1] range as the training dataset.  The implementation deliberately
keeps the diffusion objective untouched and exposes each term separately so
that ablations can be compared without changing the sampling protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DetailLossWeights:
    """Weights for the auxiliary IR-SRD terms."""

    roi_x0: float = 0.0
    edge: float = 0.0
    highpass: float = 0.0
    statistics: float = 0.0
    context: float = 0.0

    @property
    def any_enabled(self) -> bool:
        return any(value > 0.0 for value in self.__dict__.values())


@dataclass
class DetailLossStats:
    roi_x0: torch.Tensor
    edge: torch.Tensor
    highpass: torch.Tensor
    statistics: torch.Tensor
    context: torch.Tensor
    valid_fraction: torch.Tensor

    @property
    def weighted(self) -> torch.Tensor:
        # This property is only used when a caller has already applied the
        # configured weights.  Keeping the individual tensors makes logging
        # and component-wise ablations straightforward.
        return self.roi_x0 + self.edge + self.highpass + self.statistics + self.context


def prediction_to_x0(
    noisy_latents: torch.Tensor,
    prediction: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: Any,
) -> torch.Tensor:
    """Convert epsilon/v prediction to a float32 clean latent."""

    alphas = scheduler.alphas_cumprod.to(noisy_latents.device, torch.float32)[timesteps]
    alpha = alphas.sqrt().view(-1, 1, 1, 1)
    sigma = (1.0 - alphas).sqrt().view(-1, 1, 1, 1)
    prediction_type = str(scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        return (noisy_latents.float() - sigma * prediction.float()) / alpha.clamp_min(1e-6)
    if prediction_type == "v_prediction":
        return alpha * noisy_latents.float() - sigma * prediction.float()
    raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")


def _coordinate_grid(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width)
    return torch.meshgrid(y, x, indexing="ij")


def build_box_masks(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    height: int,
    width: int,
    ring_scale: float = 1.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-box ROI and context-ring masks at pixel resolution.

    Per-box masks are retained instead of taking a union.  This makes a large
    target unable to dominate the loss merely because it covers more pixels.
    The ring is the dilated rectangle minus the target rectangle and is
    clipped to the image boundaries.
    """

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [B,M,4], got {tuple(boxes.shape)}")
    if box_mask.shape != boxes.shape[:2]:
        raise ValueError("box_mask must match boxes[:2]")
    if height < 2 or width < 2:
        raise ValueError("Mask resolution must be at least 2x2")
    if ring_scale < 1.0:
        raise ValueError("ring_scale must be at least 1")

    batch, max_boxes = boxes.shape[:2]
    y, x = _coordinate_grid(height, width, boxes.device)
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    safe = boxes.float().clamp(0.0, 1.0)
    x1, y1, x2, y2 = [value.view(batch, max_boxes, 1, 1) for value in safe.unbind(-1)]
    valid = box_mask.bool().view(batch, max_boxes, 1, 1) & (x2 > x1) & (y2 > y1)
    # Use cell overlap instead of center sampling so sub-pixel boxes still
    # supervise at least one pixel on the 512x512 decoded image.
    cell_x0 = (torch.arange(width, device=boxes.device, dtype=torch.float32) / float(width))
    cell_x1 = (torch.arange(width, device=boxes.device, dtype=torch.float32) + 1.0) / float(width)
    cell_y0 = (torch.arange(height, device=boxes.device, dtype=torch.float32) / float(height))
    cell_y1 = (torch.arange(height, device=boxes.device, dtype=torch.float32) + 1.0) / float(height)
    cell_x0 = cell_x0.view(1, 1, 1, width)
    cell_x1 = cell_x1.view(1, 1, 1, width)
    cell_y0 = cell_y0.view(1, 1, height, 1)
    cell_y1 = cell_y1.view(1, 1, height, 1)
    roi_x = (torch.minimum(cell_x1, x2) - torch.maximum(cell_x0, x1)).clamp_min(0.0)
    roi_y = (torch.minimum(cell_y1, y2) - torch.maximum(cell_y0, y1)).clamp_min(0.0)
    roi = ((roi_x > 0.0) & (roi_y > 0.0) & valid).float()

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_width = (x2 - x1).clamp_min(1.0 / float(width)) * float(ring_scale) * 0.5
    half_height = (y2 - y1).clamp_min(1.0 / float(height)) * float(ring_scale) * 0.5
    ex1 = (cx - half_width).clamp_min(0.0)
    ey1 = (cy - half_height).clamp_min(0.0)
    ex2 = (cx + half_width).clamp_max(1.0)
    ey2 = (cy + half_height).clamp_max(1.0)
    expanded_x = (torch.minimum(cell_x1, ex2) - torch.maximum(cell_x0, ex1)).clamp_min(0.0)
    expanded_y = (torch.minimum(cell_y1, ey2) - torch.maximum(cell_y0, ey1)).clamp_min(0.0)
    expanded = ((expanded_x > 0.0) & (expanded_y > 0.0) & valid).float()
    ring = (expanded - roi).clamp_min(0.0)
    return roi, ring


def _per_box_values(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Return the spatial mean for every box as a [B,M] tensor."""

    denominator = masks.sum(dim=(-2, -1)).clamp_min(1.0)
    return (values * masks).sum(dim=(-2, -1)) / denominator


def _per_box_mean(values: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean a [B,M,H,W] value over each valid box, then equal-weight boxes."""

    values = _per_box_values(values, masks)
    valid = valid.to(values.dtype)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _per_box_stat(
    values: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = masks.sum(dim=(-2, -1)).clamp_min(1.0)
    mean = (values * masks).sum(dim=(-2, -1)) / denominator
    centered = (values - mean[..., None, None]) * masks
    variance = centered.square().sum(dim=(-2, -1)) / denominator
    per_box_valid = valid.to(values.dtype)
    return mean, (variance + 1e-6).sqrt() * per_box_valid


def _luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image[:, 0]
    coefficients = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image[:, :3] * coefficients).sum(dim=1)


def _gradient(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gray = _luminance(image)
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    magnitude = torch.zeros_like(gray)
    magnitude[..., :, 1:] += dx.abs()
    magnitude[..., :, :-1] += dx.abs()
    magnitude[..., 1:, :] += dy.abs()
    magnitude[..., :-1, :] += dy.abs()
    return gray, dx, dy


def _laplacian(gray: torch.Tensor) -> torch.Tensor:
    kernel = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return F.conv2d(gray.unsqueeze(1), kernel.view(1, 1, 3, 3), padding=1)[:, 0]


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    difference = (pred - target).abs().mean(dim=1)
    return _per_box_mean(difference[:, None], masks, valid).mean()


def compute_detail_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    ring_scale: float = 1.75,
) -> DetailLossStats:
    """Compute unweighted IR-SRD terms for a decoded x0 and its target.

    ``decoded`` and ``target`` are [B,C,H,W] in [-1,1].  The returned values
    are finite scalars and are zero when a batch contains no valid boxes.
    """

    if decoded.shape != target.shape or decoded.ndim != 4:
        raise ValueError("decoded and target must have the same [B,C,H,W] shape")
    if decoded.shape[0] != boxes.shape[0]:
        raise ValueError("Image and box batch dimensions do not match")
    roi, ring = build_box_masks(boxes, box_mask, decoded.shape[-2], decoded.shape[-1], ring_scale)
    valid = (roi.sum(dim=(-2, -1)) > 0) & box_mask.bool()
    valid_fraction = valid.float().sum() / float(valid.numel())
    if not bool(valid.any()):
        zero = decoded.float().sum() * 0.0
        return DetailLossStats(zero, zero, zero, zero, zero, valid_fraction)

    pred = decoded.float().clamp(-1.0, 1.0)
    truth = target.float().clamp(-1.0, 1.0)
    roi_x0 = _masked_l1(pred, truth, roi, valid)

    pred_gray, pred_dx, pred_dy = _gradient(pred)
    truth_gray, truth_dx, truth_dy = _gradient(truth)
    # Align finite-difference fields with the corresponding side of each box.
    roi_x = roi[..., :, 1:]
    roi_y = roi[..., 1:, :]
    edge_x = _per_box_mean((pred_dx[:, None] - truth_dx[:, None]).abs(), roi_x, valid).mean()
    edge_y = _per_box_mean((pred_dy[:, None] - truth_dy[:, None]).abs(), roi_y, valid).mean()
    edge = 0.5 * (edge_x + edge_y)

    pred_lap = _laplacian(pred_gray)
    truth_lap = _laplacian(truth_gray)
    highpass = _per_box_mean(
        (pred_lap[:, None] - truth_lap[:, None]).abs(), roi, valid
    ).mean()

    pred_mean, pred_std = _per_box_stat(pred_gray[:, None], roi, valid)
    truth_mean, truth_std = _per_box_stat(truth_gray[:, None], roi, valid)
    # Mean local gradient energy is computed from the symmetric magnitude map.
    pred_grad_map = torch.zeros_like(pred_gray)
    truth_grad_map = torch.zeros_like(truth_gray)
    pred_grad_map[..., :, 1:] += pred_dx.abs()
    pred_grad_map[..., :, :-1] += pred_dx.abs()
    pred_grad_map[..., 1:, :] += pred_dy.abs()
    pred_grad_map[..., :-1, :] += pred_dy.abs()
    truth_grad_map[..., :, 1:] += truth_dx.abs()
    truth_grad_map[..., :, :-1] += truth_dx.abs()
    truth_grad_map[..., 1:, :] += truth_dy.abs()
    truth_grad_map[..., :-1, :] += truth_dy.abs()
    pred_grad = _per_box_values(pred_grad_map[:, None], roi)
    truth_grad = _per_box_values(truth_grad_map[:, None], roi)

    ring_valid = valid & (ring.sum(dim=(-2, -1)) > 0)
    pred_ring_mean = _per_box_values(pred_gray[:, None], ring)
    truth_ring_mean = _per_box_values(truth_gray[:, None], ring)
    pred_contrast = (pred_mean - pred_ring_mean).abs()
    truth_contrast = (truth_mean - truth_ring_mean).abs()
    stat_terms = [
        F.smooth_l1_loss(
            torch.log(pred_std.clamp_min(1e-4)),
            torch.log(truth_std.detach().clamp_min(1e-4)),
            reduction="none",
        ),
        F.smooth_l1_loss(
            torch.log(pred_grad.clamp_min(1e-4)),
            torch.log(truth_grad.detach().clamp_min(1e-4)),
            reduction="none",
        ),
        F.smooth_l1_loss(
            torch.log(pred_contrast.clamp_min(1e-4)),
            torch.log(truth_contrast.detach().clamp_min(1e-4)),
            reduction="none",
        ),
    ]
    # Contrast is undefined when the box fills the image and has no ring;
    # exclude only that term while retaining ROI statistics for the box.
    contrast_valid = ring_valid.float()
    stat_terms[-1] = stat_terms[-1] * contrast_valid
    box_valid = valid.float()
    statistics = sum(term * box_valid for term in stat_terms).sum() / box_valid.sum().clamp_min(1.0)

    context = _per_box_mean((pred_gray[:, None] - truth_gray[:, None]).abs(), ring, ring_valid).mean()
    # Keep this scalar finite even when a ring is one pixel wide or clipped.
    values = (roi_x0, edge, highpass, statistics, context)
    if not all(torch.isfinite(value).item() for value in values):
        raise FloatingPointError("IR-SRD detail loss produced a non-finite value")
    return DetailLossStats(*values, valid_fraction)


def weighted_detail_loss(stats: DetailLossStats, weights: DetailLossWeights) -> torch.Tensor:
    """Apply configured weights to component-wise statistics."""

    return (
        float(weights.roi_x0) * stats.roi_x0
        + float(weights.edge) * stats.edge
        + float(weights.highpass) * stats.highpass
        + float(weights.statistics) * stats.statistics
        + float(weights.context) * stats.context
    )
