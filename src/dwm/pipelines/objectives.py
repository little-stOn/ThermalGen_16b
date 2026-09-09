"""Model-agnostic video diffusion objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - config-only environments
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LossResult:
    """Unnormalised loss numerator and the units it represents."""

    numerator: Any
    denominator: Any
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def mean(self) -> Any:
        return self.numerator / self.denominator.clamp_min(1.0)


class VideoObjective(Protocol):
    """Callable objective used by :class:`ThermalVideoPipeline`."""

    def __call__(
        self,
        prediction: Any,
        target: Any,
        valid_mask: Any | None = None,
    ) -> LossResult: ...


class MaskedDiffusionMSE:
    """Mean squared diffusion loss over valid ``[B,T,V]`` units.

    The objective returns a sum/count pair so the pipeline can preserve the
    correct weighting across variable local batch sizes and accumulation groups.
    """

    def __call__(
        self,
        prediction: Any,
        target: Any,
        valid_mask: Any | None = None,
    ) -> LossResult:
        if torch is None:
            raise ImportError("MaskedDiffusionMSE requires PyTorch")
        if not torch.is_tensor(prediction) or not torch.is_tensor(target):
            raise TypeError("prediction and target must be tensors")
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {tuple(prediction.shape)} "
                f"versus {tuple(target.shape)}"
            )
        if prediction.ndim < 4:
            raise ValueError("video diffusion tensors must have [B,T,V,...] dimensions")

        per_unit = (prediction.float() - target.float()).square()
        per_unit = per_unit.mean(dim=tuple(range(3, prediction.ndim)))
        if valid_mask is None:
            valid_mask = torch.ones(
                prediction.shape[:3], device=prediction.device, dtype=torch.bool
            )
        if (
            not torch.is_tensor(valid_mask)
            or tuple(valid_mask.shape) != tuple(prediction.shape[:3])
        ):
            raise ValueError(
                f"valid_mask must have shape {tuple(prediction.shape[:3])}, "
                f"got {getattr(valid_mask, 'shape', None)}"
            )
        mask = valid_mask.to(device=prediction.device, dtype=per_unit.dtype)
        denominator = mask.sum()
        if float(denominator.detach()) <= 0.0:
            raise ValueError("video objective has no valid units")
        numerator = (per_unit * mask).sum()
        mean = numerator.detach() / denominator.detach()
        return LossResult(
            numerator=numerator,
            denominator=denominator,
            metrics={"diffusion_mse": float(mean)},
        )


__all__ = ["LossResult", "MaskedDiffusionMSE", "VideoObjective"]
