"""Small trainable bridges around the frozen RGB SD VAE.

The bridge keeps the official four-channel SD latent interface intact while
learning how to present radiometric 16-bit data to the RGB VAE and how to
recover a single-channel uint16 frame from its RGB decoder output.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class RadiometricInputBridge(nn.Module):
    """Map fixed float radiometric features to the official VAE RGB range."""

    def __init__(self, channels: int = 64, blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 3:
            raise ValueError(f"InputBridge expects BCHW with 3 channels, got {tuple(features.shape)}")
        base = features.mul(2.0).sub(1.0)
        residual = self.head(self.blocks(self.stem(base)))
        return (base + 0.10 * residual).clamp(-1.0, 1.0)


class RadiometricOutputBridge(nn.Module):
    """Map the frozen RGB VAE decoder output to one normalized thermal channel."""

    def __init__(self, channels: int = 32, blocks: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, decoded_rgb: torch.Tensor) -> torch.Tensor:
        if decoded_rgb.ndim != 4 or decoded_rgb.shape[1] != 3:
            raise ValueError(
                f"OutputBridge expects BCHW with 3 channels, got {tuple(decoded_rgb.shape)}"
            )
        # The first radiometric channel is the absolute full-range signal;
        # fixed-window and local-detail channels are auxiliary views.
        base = decoded_rgb[:, :1].add(1.0).div(2.0)
        residual = self.head(self.blocks(self.stem(decoded_rgb)))
        return (base + 0.10 * residual).clamp(0.0, 1.0)


class RadiometricLatentBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_bridge = RadiometricInputBridge()
        self.output_bridge = RadiometricOutputBridge()

    def forward(self, features: torch.Tensor, vae: nn.Module, scaling_factor: float):
        vae_input = self.input_bridge(features)
        posterior = vae.encode(vae_input).latent_dist
        latent = posterior.mode() * float(scaling_factor)
        decoded_rgb = vae.decode(
            latent / float(scaling_factor), return_dict=True
        ).sample
        reconstructed = self.output_bridge(decoded_rgb)
        return reconstructed, latent, vae_input


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
