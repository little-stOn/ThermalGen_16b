"""Monotonic output-tone calibration for the Native16 bridge branch."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class RadiometricOutputCalibrator(nn.Module):
    """Map decoded bridge values to a calibrated absolute thermal range.

    The tone curve acts on a low-frequency component while retaining a
    separately scaled detail residual. This keeps the calibration monotonic
    and prevents a pure global affine map from erasing all local structure.
    """

    def __init__(self, kernel_size: int = 31) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.kernel_size = int(kernel_size)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.shift = nn.Parameter(torch.zeros(()))
        self.log_detail_scale = nn.Parameter(torch.zeros(()))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    @property
    def detail_scale(self) -> torch.Tensor:
        return self.log_detail_scale.exp()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != 1:
            raise ValueError(f"expected BCHW single-channel values, got {tuple(values.shape)}")
        values = values.float().clamp(1e-5, 1.0 - 1e-5)
        padding = self.kernel_size // 2
        low = torch.nn.functional.avg_pool2d(
            values,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        logits = torch.logit(low)
        mapped_low = torch.sigmoid(self.scale * logits + self.shift)
        detail = values - low
        return (mapped_low + self.detail_scale * detail).clamp(0.0, 1.0)

    def to_json(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "schema_version": 1,
            "kernel_size": self.kernel_size,
            "scale": float(self.scale.detach().cpu()),
            "shift": float(self.shift.detach().cpu()),
            "detail_scale": float(self.detail_scale.detach().cpu()),
        }
        if metadata:
            payload["metadata"] = metadata
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path, device: torch.device | str = "cpu") -> "RadiometricOutputCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        module = cls(kernel_size=int(payload.get("kernel_size", 31))).to(device=device)
        with torch.no_grad():
            module.log_scale.fill_(torch.log(torch.tensor(float(payload["scale"]), device=module.log_scale.device)))
            module.shift.fill_(float(payload["shift"]))
            module.log_detail_scale.fill_(torch.log(torch.tensor(float(payload.get("detail_scale", 1.0)), device=module.log_detail_scale.device)))
        return module
