"""Box-aware low-frequency radiometric transport for Native16 outputs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class WindowAwareTransportConfig:
    source_center: float = 0.3354008913
    target_center: float = 0.1103431075
    low_scale: float = 0.60
    background_detail_gain: float = 0.80
    roi_detail_gain: float = 1.00
    blend: float = 0.50
    blur_kernel: int = 31
    mask_soften_kernel: int = 65
    mask_soften_sigma: float = 0.0
    mask_expand: float = 0.25
    roi_protection: float = 0.0
    small_area_threshold: float = 1024.0
    small_protection: float = 1.0
    small_mask_expand: float = 0.75
    small_minimum_side: float = 128.0
    minimum: float = 0.02
    maximum: float = 1.00

    def __post_init__(self) -> None:
        for name in ("blur_kernel", "mask_soften_kernel"):
            value = int(getattr(self, name))
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        for name in ("blend", "roi_protection", "small_protection"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.minimum < self.maximum <= 1.0:
            raise ValueError("minimum/maximum must define a range inside [0, 1]")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def get_transport_config(preset: str) -> WindowAwareTransportConfig:
    if preset != "g065":
        raise ValueError(f"unsupported radiometric transport preset: {preset}")
    return WindowAwareTransportConfig()


def _box_to_xyxy(
    box: Sequence[float], height: int, width: int, box_format: str
) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"expected four box coordinates, got {box}")
    a, b, c, d = (float(value) for value in box)
    if box_format == "xywh":
        x0, y0, x1, y1 = a, b, a + c, b + d
    elif box_format == "xyxy":
        x0, y0, x1, y1 = a, b, c, d
    else:
        raise ValueError(f"unsupported box format: {box_format}")
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    return x0, y0, x1, y1


def build_box_mask(
    boxes: Sequence[Sequence[float]],
    height: int,
    width: int,
    expand: float,
    minimum_side: float,
    box_format: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    for box in boxes:
        x0, y0, x1, y1 = _box_to_xyxy(box, height, width, box_format)
        box_width = max(1.0, x1 - x0)
        box_height = max(1.0, y1 - y0)
        x0 -= expand * box_width
        x1 += expand * box_width
        y0 -= expand * box_height
        y1 += expand * box_height
        if minimum_side > 0.0:
            center_x = 0.5 * (x0 + x1)
            center_y = 0.5 * (y0 + y1)
            half_width = max(0.5 * minimum_side, 0.5 * (x1 - x0))
            half_height = max(0.5 * minimum_side, 0.5 * (y1 - y0))
            x0, x1 = center_x - half_width, center_x + half_width
            y0, y1 = center_y - half_height, center_y + half_height
        ix0 = max(0, int(math.floor(x0)))
        iy0 = max(0, int(math.floor(y0)))
        ix1 = min(width, int(math.ceil(x1)))
        iy1 = min(height, int(math.ceil(y1)))
        if ix1 > ix0 and iy1 > iy0:
            mask[:, :, iy0:iy1, ix0:ix1] = 1.0
    return mask


def _soften_mask(mask: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    padding = kernel_size // 2
    if sigma <= 0.0:
        sigma = max(1.0, kernel_size / 6.0)
    coordinates = torch.arange(
        kernel_size, device=mask.device, dtype=mask.dtype
    ) - float(padding)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    horizontal = kernel.view(1, 1, 1, kernel_size)
    vertical = kernel.view(1, 1, kernel_size, 1)
    output = F.conv2d(
        F.pad(mask, (padding, padding, 0, 0), mode="replicate"), horizontal
    )
    output = F.conv2d(
        F.pad(output, (0, 0, padding, padding), mode="replicate"), vertical
    )
    return output.clamp(0.0, 1.0)


class WindowAwareRadiometricTransport(nn.Module):
    """Apply the validated g065 low-frequency transport while protecting boxes."""

    def __init__(self, config: WindowAwareTransportConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def from_preset(cls, preset: str) -> "WindowAwareRadiometricTransport":
        return cls(get_transport_config(preset))

    def forward(
        self,
        values: torch.Tensor,
        boxes: Sequence[Sequence[float]],
        *,
        box_format: str = "xyxy",
    ) -> torch.Tensor:
        if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] != 1:
            raise ValueError(
                "window-aware transport expects a single BCHW one-channel image, "
                f"got {tuple(values.shape)}"
            )
        config = self.config
        values = values.float()
        height, width = values.shape[-2:]
        roi_mask = build_box_mask(
            boxes,
            height,
            width,
            config.mask_expand,
            0.0,
            box_format,
            values.device,
            values.dtype,
        )
        small_boxes = [
            box
            for box in boxes
            if self._box_area(box, height, width, box_format)
            < config.small_area_threshold
        ]
        small_mask = build_box_mask(
            small_boxes,
            height,
            width,
            config.small_mask_expand,
            config.small_minimum_side,
            box_format,
            values.device,
            values.dtype,
        )
        padding = config.blur_kernel // 2
        low = F.avg_pool2d(
            values,
            config.blur_kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        detail = values - low
        soft_roi = _soften_mask(
            roi_mask, config.mask_soften_kernel, config.mask_soften_sigma
        )
        soft_small = _soften_mask(
            small_mask, config.mask_soften_kernel, config.mask_soften_sigma
        )
        detail_gain = config.background_detail_gain + soft_roi * (
            config.roi_detail_gain - config.background_detail_gain
        )
        transported = (
            config.target_center
            + config.low_scale * (low - config.source_center)
            + detail_gain * detail
        )
        protection = torch.maximum(
            config.roi_protection * soft_roi,
            config.small_protection * soft_small,
        )
        spatial_blend = config.blend * (1.0 - protection)
        return (values + spatial_blend * (transported - values)).clamp(
            config.minimum, config.maximum
        )

    @staticmethod
    def _box_area(
        box: Sequence[float], height: int, width: int, box_format: str
    ) -> float:
        x0, y0, x1, y1 = _box_to_xyxy(box, height, width, box_format)
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)
