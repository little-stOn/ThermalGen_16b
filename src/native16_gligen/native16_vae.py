"""Construction and validation helpers for the independent one-channel VAE."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL


MANIFEST_NAME = "native16_manifest.json"
LATENT_AFFINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Native16LatentAffine:
    """Reversible affine map in scaled diffusion-latent space."""

    scale: tuple[float, float, float, float]
    shift: tuple[float, float, float, float]
    source_vae: str
    target: str

    @classmethod
    def from_json(cls, path: str | Path) -> "Native16LatentAffine":
        calibration_path = Path(path).expanduser().resolve()
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != LATENT_AFFINE_SCHEMA_VERSION:
            raise ValueError(f"unsupported Native16 latent calibration schema: {calibration_path}")
        if payload.get("branch") != "native16":
            raise ValueError("latent calibration is not marked as native16")
        if payload.get("space") != "scaled_diffusion_latent":
            raise ValueError("latent calibration must be in scaled_diffusion_latent space")
        scale = tuple(float(value) for value in payload.get("scale", ()))
        shift = tuple(float(value) for value in payload.get("shift", ()))
        if len(scale) != 4 or len(shift) != 4:
            raise ValueError("Native16 latent calibration must contain four scale and shift values")
        if any(not math.isfinite(value) or abs(value) < 1e-6 for value in scale):
            raise ValueError("Native16 latent calibration scale contains an invalid value")
        if any(not math.isfinite(value) for value in shift):
            raise ValueError("Native16 latent calibration shift contains an invalid value")
        return cls(
            scale=scale,
            shift=shift,
            source_vae=str(payload.get("source_vae", "")),
            target=str(payload.get("target", "")),
        )

    def _tensors(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim < 2 or int(latents.shape[1]) != 4:
            raise ValueError(f"Native16 latent affine expects BCHW with four channels, got {tuple(latents.shape)}")
        scale = latents.new_tensor(self.scale).view(1, 4, *([1] * (latents.ndim - 2)))
        shift = latents.new_tensor(self.shift).view(1, 4, *([1] * (latents.ndim - 2)))
        return scale, shift

    def apply(self, latents: torch.Tensor) -> torch.Tensor:
        scale, shift = self._tensors(latents)
        return latents * scale + shift

    def inverse(self, latents: torch.Tensor) -> torch.Tensor:
        scale, shift = self._tensors(latents)
        return (latents - shift) / scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_clean_official_parent(model_path: str | Path) -> Path:
    """Require a Diffusers base-model directory with an official VAE layout."""

    path = Path(model_path).expanduser().resolve()
    if not (path / "vae" / "config.json").is_file():
        raise FileNotFoundError(
            "The Native16 base model must be a Diffusers pipeline directory with "
            f"vae/config.json, got: {path}"
        )
    return path


def initialize_native16_vae(model_path: str | Path) -> tuple[AutoencoderKL, dict[str, Any]]:
    """Create a 1->4->1 VAE from an untouched RGB SD VAE initialization."""

    parent = assert_clean_official_parent(model_path)
    rgb = AutoencoderKL.from_pretrained(parent, subfolder="vae")
    config = dict(rgb.config)
    config["in_channels"] = 1
    config["out_channels"] = 1
    native = AutoencoderKL.from_config(config)
    state = rgb.state_dict()

    # Summing reproduces the RGB encoder response for a repeated grayscale input.
    state["encoder.conv_in.weight"] = state["encoder.conv_in.weight"].sum(
        dim=1, keepdim=True
    )
    # Averaging the RGB heads produces one luminance-like thermal reconstruction.
    state["decoder.conv_out.weight"] = state["decoder.conv_out.weight"].mean(
        dim=0, keepdim=True
    )
    state["decoder.conv_out.bias"] = state["decoder.conv_out.bias"].mean().reshape(1)
    missing, unexpected = native.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"native16 initialization mismatch missing={missing} unexpected={unexpected}")
    del rgb

    config_path = parent / "vae" / "config.json"
    manifest = {
        "schema_version": 1,
        "branch": "native16",
        "bit_depth": 16,
        "input_channels": 1,
        "output_channels": 1,
        "latent_channels": int(native.config.latent_channels),
        "parent": "official_sd14_gligen_vae",
        "parent_path": str(parent),
        "parent_config_sha256": _sha256(config_path),
        "initialization": {
            "encoder_conv_in": "sum_rgb_channels",
            "decoder_conv_out": "mean_rgb_channels",
            "all_other_parameters": "copied_from_official_parent",
        },
        "normalization": "fixed_storage_range",
        "legacy_8bit_checkpoint": False,
        "project_checkpoint_loaded": False,
    }
    return native, manifest


def save_manifest(directory: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(directory) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_native16_manifest(directory: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(directory)
    vae_dir = root / "vae" if (root / "vae").is_dir() else root
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() and vae_dir.parent != root:
        manifest_path = vae_dir.parent / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"native16 VAE manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "bit_depth": 16,
        "input_channels": 1,
        "output_channels": 1,
        "legacy_8bit_checkpoint": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"invalid native16 manifest {key}={manifest.get(key)!r}")
    return vae_dir, manifest


def load_native16_vae(directory: str | Path) -> tuple[AutoencoderKL, dict[str, Any]]:
    vae_dir, manifest = read_native16_manifest(directory)
    vae = AutoencoderKL.from_pretrained(vae_dir)
    if int(vae.config.in_channels) != 1 or int(vae.config.out_channels) != 1:
        raise ValueError("native16 VAE config must be one-channel input/output")
    return vae, manifest
