"""Native uint16 thermal loading and fixed radiometric encoding.

The encoder intentionally uses dataset-level constants.  Per-image min/max
normalisation is not allowed because it erases absolute temperature ordering
between frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RadiometricProfile:
    storage_min: float = 0.0
    storage_max: float = 65535.0
    window_low: float = 20000.0
    window_high: float = 40000.0
    local_percentile_low: float = 1.0
    local_percentile_high: float = 99.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    @classmethod
    def from_json(cls, path: str | Path) -> "RadiometricProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        return cls(**{name: payload[name] for name in fields if name in payload})

    @classmethod
    def from_config(cls, value: Any) -> "RadiometricProfile":
        if value is None:
            return cls()
        if isinstance(value, (str, Path)):
            return cls.from_json(value)
        if isinstance(value, dict):
            return cls(**{name: value[name] for name in cls.__dataclass_fields__ if name in value})
        # ConfigNode is dict-like but not necessarily an actual dict subclass
        return cls(**{name: getattr(value, name) for name in cls.__dataclass_fields__ if hasattr(value, name)})

    def validate(self) -> None:
        if not self.storage_max > self.storage_min:
            raise ValueError("thermal16 storage_max must exceed storage_min")
        if not self.window_high > self.window_low:
            raise ValueError("thermal16 window_high must exceed window_low")
        if not 0 <= self.local_percentile_low < self.local_percentile_high <= 100:
            raise ValueError("thermal16 local percentiles are invalid")
        if self.clahe_clip_limit <= 0 or int(self.clahe_grid_size) <= 0:
            raise ValueError("thermal16 CLAHE parameters must be positive")


def read_raw_thermal(path: str | Path) -> np.ndarray:
    """Read native single-channel uint16 TIFF/PNG without AGC."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"thermal16 image not found: {image_path}")
    image: np.ndarray | None = None
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            image = np.asarray(tifffile.imread(image_path))
        except Exception as tif_error:
            try:
                with Image.open(image_path) as source:
                    image = np.asarray(source)
            except Exception:
                raise tif_error
    else:
        try:
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        except ImportError:
            with Image.open(image_path) as source:
                image = np.asarray(source)
    if image is None:
        raise OSError(f"failed to read thermal16 image: {image_path}")
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"expected one thermal channel, got {image.shape}: {image_path}")
    if image.dtype != np.uint16:
        raise TypeError(f"expected uint16 thermal input, got {image.dtype}: {image_path}")
    return image


def _scale(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(
        (raw.astype(np.float32) - np.float32(low)) / np.float32(high - low), 0.0, 1.0
    )


def encode_radiometric_triplet(raw: np.ndarray, profile: RadiometricProfile) -> np.ndarray:
    """Return float32 HxWx3 in [0, 1]: absolute, fixed-window, local-detail."""

    profile.validate()
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise TypeError("raw thermal input must be a single-channel uint16 array")
    absolute = _scale(raw, profile.storage_min, profile.storage_max)
    fixed = _scale(raw, profile.window_low, profile.window_high)
    low, high = np.percentile(
        raw, [profile.local_percentile_low, profile.local_percentile_high]
    )
    local = fixed if high <= low else _scale(raw, float(low), float(high))
    local_u8 = np.rint(local * 255.0).astype(np.uint8)
    try:
        import cv2

        clahe = cv2.createCLAHE(
            clipLimit=float(profile.clahe_clip_limit),
            tileGridSize=(int(profile.clahe_grid_size), int(profile.clahe_grid_size)),
        )
        local_detail = clahe.apply(local_u8).astype(np.float32) / 255.0
    except ImportError:
        local_detail = local.astype(np.float32)
    return np.stack((absolute, fixed, local_detail), axis=-1).astype(np.float32)


def load_thermal16_pil(path: str | Path, profile: RadiometricProfile) -> Image.Image:
    """Encode for the existing PIL crop/flip path without per-image normalisation."""

    encoded = encode_radiometric_triplet(read_raw_thermal(path), profile)
    return Image.fromarray(np.rint(encoded * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def load_thermal16_tensor(path: str | Path, profile: RadiometricProfile):
    """Return CxHxW float32 [0, 1] without an intermediate uint8 quantisation."""

    import torch

    encoded = encode_radiometric_triplet(read_raw_thermal(path), profile)
    return torch.from_numpy(encoded).permute(2, 0, 1).contiguous()


def encode_native16(raw: np.ndarray, profile: RadiometricProfile) -> np.ndarray:
    """Map uint16 values to one float32 channel without per-image AGC."""

    profile.validate()
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise TypeError("raw thermal input must be a single-channel uint16 array")
    return _scale(raw, profile.storage_min, profile.storage_max)[None, ...]


def decode_native16(values: np.ndarray, profile: RadiometricProfile) -> np.ndarray:
    """Invert :func:`encode_native16` and return a native uint16 frame."""

    profile.validate()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected one decoded thermal channel, got {array.shape}")
    scaled = array.clip(0.0, 1.0) * np.float32(
        profile.storage_max - profile.storage_min
    ) + np.float32(profile.storage_min)
    return np.rint(scaled).clip(0, 65535).astype(np.uint16)


def load_native16_tensor(path: str | Path, profile: RadiometricProfile):
    """Return 1xHxW float32 in [0, 1], preserving absolute sensor values."""

    import torch

    return torch.from_numpy(encode_native16(read_raw_thermal(path), profile)).contiguous()


def save_native16_tiff(path: str | Path, values: np.ndarray, profile: RadiometricProfile) -> None:
    """Save a normalized single-channel prediction as lossless uint16 TIFF."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = decode_native16(values, profile)
    try:
        import tifffile

        tifffile.imwrite(output_path, raw, photometric="minisblack", compression="zlib")
    except ImportError:
        Image.fromarray(raw, mode="I;16").save(output_path)
