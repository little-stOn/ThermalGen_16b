"""Shared contracts for auxiliary diffusion-training objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossContribution:
    """One auxiliary objective after warmup and diffusion-relative clipping."""

    name: str
    raw: torch.Tensor
    weight: float
    weighted: torch.Tensor
    ratio_to_diffusion: torch.Tensor


@dataclass(frozen=True)
class RadiometricTargets:
    """Native16 targets with explicit channel semantics in VAE pixel space."""

    triplet: torch.Tensor
    absolute: torch.Tensor
    fixed_window: torch.Tensor
    local_detail: torch.Tensor


def radiometric_targets(target: torch.Tensor) -> RadiometricTargets:
    """Validate and label the [absolute, fixed-window, local-detail] target triplet."""

    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError(
            "Native16 radiometric targets must have shape [B,3,H,W] "
            "for [absolute, fixed-window, local-detail]"
        )
    return RadiometricTargets(
        triplet=target,
        absolute=target[:, :1],
        fixed_window=target[:, 1:2],
        local_detail=target[:, 2:3],
    )


def pixel_target_for_decoded(
    decoded: torch.Tensor,
    target: torch.Tensor,
    input_mode: str,
) -> torch.Tensor:
    """Return a target with the decoded channel count without losing target semantics."""

    if decoded.ndim != 4 or target.ndim != 4:
        raise ValueError("decoded and target must both have shape [B,C,H,W]")
    if decoded.shape[0] != target.shape[0] or decoded.shape[-2:] != target.shape[-2:]:
        raise ValueError("decoded and target batch/spatial dimensions must match")
    if decoded.shape[1] == target.shape[1]:
        return target
    if input_mode == "native16_bridge" and decoded.shape[1] == 1 and target.shape[1] == 3:
        return radiometric_targets(target).absolute
    raise ValueError(
        f"cannot align decoded channels={decoded.shape[1]} with "
        f"target channels={target.shape[1]} for input_mode={input_mode}"
    )


def style_target(target: torch.Tensor, input_mode: str) -> torch.Tensor:
    """Return the full radiometric triplet required by IR style statistics."""

    if input_mode != "native16_bridge":
        if target.ndim != 4 or target.shape[1] < 3:
            raise ValueError("IR style loss requires a [B,3,H,W] thermal target")
        return target
    return radiometric_targets(target).triplet


def bounded_contribution(
    name: str,
    raw: torch.Tensor,
    diffusion_loss: torch.Tensor,
    *,
    nominal_weight: float,
    warmup_fraction: float,
    max_ratio: float,
) -> LossContribution:
    """Apply a named objective exactly once and cap its relative contribution."""

    if raw.ndim != 0 or diffusion_loss.ndim != 0:
        raise ValueError("Losses must be scalar tensors")
    if nominal_weight < 0.0 or not 0.0 <= warmup_fraction <= 1.0 or max_ratio <= 0.0:
        raise ValueError("Invalid auxiliary-loss schedule")
    weight = float(nominal_weight) * float(warmup_fraction)
    denominator = diffusion_loss.detach().abs().clamp_min(1.0e-6)
    proposed_ratio = (raw.detach().abs() * weight) / denominator
    if float(proposed_ratio) > max_ratio:
        weight *= max_ratio / float(proposed_ratio)
    weighted = raw * weight
    return LossContribution(
        name=name,
        raw=raw,
        weight=weight,
        weighted=weighted,
        ratio_to_diffusion=weighted.detach().abs() / denominator,
    )


def compose_total_loss(diffusion_loss: torch.Tensor, contributions: list[LossContribution]) -> torch.Tensor:
    """Add every configured auxiliary contribution exactly once."""

    if diffusion_loss.ndim != 0:
        raise ValueError("diffusion_loss must be scalar")
    total = diffusion_loss
    for contribution in contributions:
        total = total + contribution.weighted
    return total
