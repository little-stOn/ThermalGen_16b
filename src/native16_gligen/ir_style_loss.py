"""Bounded radiometric style losses for the native16 bridge branch.

The target tensor is [absolute, fixed-window, local-detail] in normalized
[-1, 1]. The decoded bridge output is a single absolute channel. Absolute
statistics are primary; the same prediction is projected into the fixed
window for a weaker auxiliary constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from native16_gligen.detail_loss import build_box_masks


@dataclass
class IRStyleLossStats:
    gray: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    gradient: torch.Tensor
    pixel: torch.Tensor
    quantile: torch.Tensor
    cdf: torch.Tensor
    roi_mean: torch.Tensor
    roi_std: torch.Tensor
    roi_gradient: torch.Tensor
    fixed_mean: torch.Tensor
    fixed_std: torch.Tensor
    fixed_gradient: torch.Tensor
    fixed_roi_mean: torch.Tensor
    fixed_roi_std: torch.Tensor
    fixed_roi_gradient: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return (
            self.gray
            + self.mean
            + self.std
            + self.gradient
            + self.pixel
            + self.quantile
            + self.cdf
            + self.roi_mean
            + self.roi_std
            + self.roi_gradient
            + self.fixed_mean
            + self.fixed_std
            + self.fixed_gradient
            + self.fixed_roi_mean
            + self.fixed_roi_std
            + self.fixed_roi_gradient
        )


def _luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image[:, 0]
    # The output is RGB, so use luminance rather than an arbitrary channel.
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image[:, :3] * weights).sum(dim=1)


def _gradient_energy(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return 0.5 * (dx.abs().mean(dim=(-2, -1)) + dy.abs().mean(dim=(-2, -1)))


def _soft_histogram(values: torch.Tensor, bins: int = 64, sample_limit: int = 4096) -> torch.Tensor:
    """Differentiable per-image histogram on [-1, 1]."""

    flat = values.flatten(1)
    if flat.shape[1] > sample_limit:
        stride = (flat.shape[1] + sample_limit - 1) // sample_limit
        flat = flat[:, ::stride]
    centers = torch.linspace(-1.0, 1.0, bins, device=values.device, dtype=values.dtype)
    width = 1.5 / float(bins)
    weights = torch.exp(-0.5 * ((flat.unsqueeze(-1) - centers) / width) ** 2)
    histogram = weights.mean(dim=1)
    return histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _per_box_mean(values: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denominator = masks.sum(dim=(-2, -1)).clamp_min(1.0)
    means = (values * masks).sum(dim=(-2, -1)) / denominator
    valid_float = valid.to(means.dtype)
    return (means * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)


def _per_box_std(values: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denominator = masks.sum(dim=(-2, -1)).clamp_min(1.0)
    means = (values * masks).sum(dim=(-2, -1)) / denominator
    variance = (((values - means[..., None, None]) ** 2) * masks).sum(dim=(-2, -1)) / denominator
    valid_float = valid.to(values.dtype)
    std = (variance + 1e-6).sqrt()
    return (std * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)


def compute_ir_style_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    storage_min: float = 0.0,
    storage_max: float = 65535.0,
    window_low: float = 20000.0,
    window_high: float = 40000.0,
) -> IRStyleLossStats:
    """Return absolute and fixed-window thermal-statistic terms for decoded x0."""

    if (
        decoded.ndim != 4
        or target.ndim != 4
        or decoded.shape[0] != target.shape[0]
        or decoded.shape[-2:] != target.shape[-2:]
        or decoded.shape[1] not in {1, 3}
        or target.shape[1] < 3
    ):
        raise ValueError("decoded must be [B,1|3,H,W] and target must be [B,3,H,W]")
    if not storage_max > storage_min or not window_high > window_low:
        raise ValueError("Invalid radiometric profile ranges")
    pred = decoded.float().clamp(-1.0, 1.0)
    truth = target.float().clamp(-1.0, 1.0)
    pred_gray = pred[:, 0] if decoded.shape[1] == 1 else _luminance(pred)
    truth_gray = truth[:, 0]
    pred_absolute_unit = pred_gray.add(1.0).mul(0.5)
    raw = pred_absolute_unit * float(storage_max - storage_min) + float(storage_min)
    pred_fixed = (
        (raw - float(window_low)) / float(window_high - window_low)
    ).clamp(0.0, 1.0).mul(2.0).sub(1.0)
    truth_fixed = truth[:, 1]

    channel_gap = (
        0.5 * (
            (pred[:, 0] - pred[:, 1]).abs().mean()
            + (pred[:, 1] - pred[:, 2]).abs().mean()
        )
        if decoded.shape[1] == 3
        else pred.new_zeros(())
    )
    mean_loss = F.smooth_l1_loss(pred_gray.mean(dim=(-2, -1)), truth_gray.mean(dim=(-2, -1)))
    std_loss = F.smooth_l1_loss(
        pred_gray.flatten(1).std(dim=1, unbiased=False),
        truth_gray.flatten(1).std(dim=1, unbiased=False),
    )
    gradient_loss = F.smooth_l1_loss(_gradient_energy(pred_gray), _gradient_energy(truth_gray))
    # Pixel and distribution terms are evaluated per image. The pixel term is
    # intentionally robust (Smooth L1) so it aligns absolute radiometry
    # without turning the denoiser into a blurry regression model.
    pixel_loss = F.smooth_l1_loss(pred_gray, truth_gray)
    quantiles = pred_gray.new_tensor([0.01, 0.10, 0.50, 0.90, 0.99])
    pred_quantiles = torch.quantile(pred_gray.flatten(1), quantiles, dim=1).transpose(0, 1)
    truth_quantiles = torch.quantile(truth_gray.flatten(1), quantiles, dim=1).transpose(0, 1)
    quantile_loss = F.smooth_l1_loss(pred_quantiles, truth_quantiles)
    pred_histogram = _soft_histogram(pred_gray)
    truth_histogram = _soft_histogram(truth_gray)
    cdf_loss = F.smooth_l1_loss(
        pred_histogram.cumsum(dim=-1), truth_histogram.cumsum(dim=-1)
    )

    fixed_mean_loss = F.smooth_l1_loss(
        pred_fixed.mean(dim=(-2, -1)), truth_fixed.mean(dim=(-2, -1))
    )
    fixed_std_loss = F.smooth_l1_loss(
        pred_fixed.flatten(1).std(dim=1, unbiased=False),
        truth_fixed.flatten(1).std(dim=1, unbiased=False),
    )
    fixed_gradient_loss = F.smooth_l1_loss(
        _gradient_energy(pred_fixed), _gradient_energy(truth_fixed)
    )

    roi, _ = build_box_masks(boxes, box_mask, pred_gray.shape[-2], pred_gray.shape[-1])
    valid = (roi.sum(dim=(-2, -1)) > 0) & box_mask.bool()
    if not bool(valid.any()):
        zero = channel_gap * 0.0
        return IRStyleLossStats(
            gray=channel_gap,
            mean=mean_loss,
            std=std_loss,
            gradient=gradient_loss,
            pixel=pixel_loss,
            quantile=quantile_loss,
            cdf=cdf_loss,
            roi_mean=zero,
            roi_std=zero,
            roi_gradient=zero,
            fixed_mean=fixed_mean_loss,
            fixed_std=fixed_std_loss,
            fixed_gradient=fixed_gradient_loss,
            fixed_roi_mean=zero,
            fixed_roi_std=zero,
            fixed_roi_gradient=zero,
        )
    pred_roi_mean = _per_box_mean(pred_gray[:, None], roi, valid)
    truth_roi_mean = _per_box_mean(truth_gray[:, None], roi, valid)
    pred_roi_std = _per_box_std(pred_gray[:, None], roi, valid)
    truth_roi_std = _per_box_std(truth_gray[:, None], roi, valid)
    # Equal-weight boxes so large objects cannot dominate the style constraint.
    pred_grad_map = torch.zeros_like(pred_gray)
    truth_grad_map = torch.zeros_like(truth_gray)
    pred_grad_map[..., :, 1:] += (pred_gray[..., :, 1:] - pred_gray[..., :, :-1]).abs()
    pred_grad_map[..., 1:, :] += (pred_gray[..., 1:, :] - pred_gray[..., :-1, :]).abs()
    truth_grad_map[..., :, 1:] += (truth_gray[..., :, 1:] - truth_gray[..., :, :-1]).abs()
    truth_grad_map[..., 1:, :] += (truth_gray[..., 1:, :] - truth_gray[..., :-1, :]).abs()
    pred_roi_grad = _per_box_mean(pred_grad_map[:, None], roi, valid)
    truth_roi_grad = _per_box_mean(truth_grad_map[:, None], roi, valid)
    roi_mean_loss = F.smooth_l1_loss(pred_roi_mean, truth_roi_mean)
    roi_std_loss = F.smooth_l1_loss(pred_roi_std, truth_roi_std)
    roi_gradient_loss = F.smooth_l1_loss(pred_roi_grad, truth_roi_grad)
    pred_fixed_roi_mean = _per_box_mean(pred_fixed[:, None], roi, valid)
    truth_fixed_roi_mean = _per_box_mean(truth_fixed[:, None], roi, valid)
    pred_fixed_roi_std = _per_box_std(pred_fixed[:, None], roi, valid)
    truth_fixed_roi_std = _per_box_std(truth_fixed[:, None], roi, valid)
    fixed_pred_grad_map = torch.zeros_like(pred_fixed)
    fixed_truth_grad_map = torch.zeros_like(truth_fixed)
    fixed_pred_grad_map[..., :, 1:] += (pred_fixed[..., :, 1:] - pred_fixed[..., :, :-1]).abs()
    fixed_pred_grad_map[..., 1:, :] += (pred_fixed[..., 1:, :] - pred_fixed[..., :-1, :]).abs()
    fixed_truth_grad_map[..., :, 1:] += (truth_fixed[..., :, 1:] - truth_fixed[..., :, :-1]).abs()
    fixed_truth_grad_map[..., 1:, :] += (truth_fixed[..., 1:, :] - truth_fixed[..., :-1, :]).abs()
    fixed_pred_roi_grad = _per_box_mean(fixed_pred_grad_map[:, None], roi, valid)
    fixed_truth_roi_grad = _per_box_mean(fixed_truth_grad_map[:, None], roi, valid)
    fixed_roi_mean_loss = F.smooth_l1_loss(pred_fixed_roi_mean, truth_fixed_roi_mean)
    fixed_roi_std_loss = F.smooth_l1_loss(pred_fixed_roi_std, truth_fixed_roi_std)
    fixed_roi_gradient_loss = F.smooth_l1_loss(fixed_pred_roi_grad, fixed_truth_roi_grad)
    return IRStyleLossStats(
        gray=channel_gap,
        mean=mean_loss,
        std=std_loss,
        gradient=gradient_loss,
        pixel=pixel_loss,
        quantile=quantile_loss,
        cdf=cdf_loss,
        roi_mean=roi_mean_loss,
        roi_std=roi_std_loss,
        roi_gradient=roi_gradient_loss,
        fixed_mean=fixed_mean_loss,
        fixed_std=fixed_std_loss,
        fixed_gradient=fixed_gradient_loss,
        fixed_roi_mean=fixed_roi_mean_loss,
        fixed_roi_std=fixed_roi_std_loss,
        fixed_roi_gradient=fixed_roi_gradient_loss,
    )


def weighted_ir_style_loss(stats: IRStyleLossStats, weights: dict[str, float]) -> torch.Tensor:
    """Apply named weights while keeping each logged component available."""

    return (
        float(weights.get("gray", 0.0)) * stats.gray
        + float(weights.get("mean", 0.0)) * stats.mean
        + float(weights.get("std", 0.0)) * stats.std
        + float(weights.get("gradient", 0.0)) * stats.gradient
        + float(weights.get("pixel", 0.0)) * stats.pixel
        + float(weights.get("quantile", 0.0)) * stats.quantile
        + float(weights.get("cdf", 0.0)) * stats.cdf
        + float(weights.get("roi_mean", 0.0)) * stats.roi_mean
        + float(weights.get("roi_std", 0.0)) * stats.roi_std
        + float(weights.get("roi_gradient", 0.0)) * stats.roi_gradient
        + float(weights.get("fixed_window", 0.0))
        * (
            stats.fixed_mean
            + stats.fixed_std
            + stats.fixed_gradient
            + stats.fixed_roi_mean
            + stats.fixed_roi_std
            + stats.fixed_roi_gradient
        ) / 6.0
    )
