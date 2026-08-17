"""Frequency- and box-aware output adapter for Native16 style transfer.

The adapter is deliberately small and identity-initialized.  It changes the
low-frequency background tone while leaving the condition boxes and their
high-frequency edges on a nearly identity path.  This keeps style adaptation
separate from the frozen GLIGEN denoiser.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class FrequencyMaskStyleAdapter(nn.Module):
    """Identity-initialized, mask-aware low-frequency style adapter."""

    def __init__(
        self,
        hidden_channels: int = 32,
        blur_kernel: int = 31,
        max_background_delta: float = 0.08,
        max_roi_delta: float = 0.08,
        background_floor: float = 0.02,
    ) -> None:
        super().__init__()
        if blur_kernel < 3 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd and at least 3")
        self.blur_kernel = int(blur_kernel)
        self.max_background_delta = float(max_background_delta)
        self.max_roi_delta = float(max_roi_delta)
        self.background_floor = float(background_floor)

        # The final convolutions are zero-initialized, so the adapter starts as
        # a pure identity mapping and cannot damage the baseline at step zero.
        self.background_net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        self.roi_net = nn.Sequential(
            nn.Conv2d(3, hidden_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, 3, padding=1),
        )
        nn.init.zeros_(self.background_net[-1].weight)
        nn.init.zeros_(self.background_net[-1].bias)
        nn.init.zeros_(self.roi_net[-1].weight)
        nn.init.zeros_(self.roi_net[-1].bias)

        # At initialization scale=1, shift=0, and the high-frequency bypass is
        # one, making lowpass(x)+highpass(x) exactly equal to x.
        self.log_background_scale = nn.Parameter(torch.zeros(()))
        self.background_shift = nn.Parameter(torch.zeros(()))
        self.high_frequency_gain = nn.Parameter(torch.ones(()))

    def forward(self, values: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != 1:
            raise ValueError(f"values must be [B,1,H,W], got {tuple(values.shape)}")
        if roi_mask.shape != values.shape:
            raise ValueError("roi_mask must have the same shape as values")
        values = values.float().clamp(0.0, 1.0)
        roi_mask = roi_mask.float().clamp(0.0, 1.0)
        background_mask = 1.0 - roi_mask
        padding = self.blur_kernel // 2
        low = F.avg_pool2d(values, self.blur_kernel, stride=1, padding=padding)
        high = values - low

        background_input = torch.cat((low, high, background_mask), dim=1)
        learned_background = torch.tanh(self.background_net(background_input))
        scale = self.log_background_scale.clamp(-3.0, 1.0).exp()
        shift = self.background_shift.clamp(-0.5, 0.5)
        high_gain = self.high_frequency_gain.clamp(0.0, 1.0)
        styled_background = (
            low * scale
            + high * high_gain
            + shift
            + self.max_background_delta * learned_background
        ).clamp(self.background_floor, 1.0)

        roi_input = torch.cat((values, high, roi_mask), dim=1)
        roi_delta = self.max_roi_delta * torch.tanh(self.roi_net(roi_input))
        styled_roi = (values + roi_delta).clamp(0.0, 1.0)
        return (background_mask * styled_background + roi_mask * styled_roi).clamp(
            0.0, 1.0
        )


def adapter_config(adapter: FrequencyMaskStyleAdapter) -> dict[str, float | int]:
    return {
        "hidden_channels": int(adapter.background_net[0].out_channels),
        "blur_kernel": int(adapter.blur_kernel),
        "max_background_delta": float(adapter.max_background_delta),
        "max_roi_delta": float(adapter.max_roi_delta),
        "background_floor": float(adapter.background_floor),
    }


def load_adapter_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[FrequencyMaskStyleAdapter, Mapping[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload.get("adapter_config", {}))
    adapter = FrequencyMaskStyleAdapter(**config)
    adapter.load_state_dict(payload["model"], strict=True)
    adapter.to(device=device, dtype=torch.float32).eval()
    return adapter, payload
