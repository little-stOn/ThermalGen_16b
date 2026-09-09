"""Public contracts shared by model-agnostic video pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - config-only environments
    torch = None  # type: ignore[assignment]


class PipelineContractError(ValueError):
    """Raised when a loader batch violates the video pipeline contract."""


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("thermal video pipelines require PyTorch")
    return torch


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise PipelineContractError(f"invalid tensor shape: {shape!r}") from error


def _nested_shape(value: Any, levels: int | None = None) -> tuple[int, ...] | None:
    if levels == 0:
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    next_levels = None if levels is None else levels - 1
    child_shape = _nested_shape(value[0], next_levels)
    if child_shape is None:
        return None
    if any(_nested_shape(item, next_levels) != child_shape for item in value[1:]):
        return None
    return (len(value),) + child_shape


def _move_value(value: Any, device: Any) -> Any:
    if torch is not None and torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    return value


def _check_finite(name: str, value: Any) -> None:
    if torch is None or not torch.is_tensor(value) or value.numel() == 0:
        return
    if not bool(torch.isfinite(value).all()):
        raise PipelineContractError(f"{name} contains non-finite values")


def _validate_box_tree(value: Any) -> None:
    if torch is not None and torch.is_tensor(value):
        value_shape = _shape(value)
        if value_shape is None or len(value_shape) != 2 or value_shape[-1] != 4:
            raise PipelineContractError(
                f"each box tensor must have shape [N,4], got {value_shape}"
            )
        _check_finite("boxes", value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_box_tree(item)


def _timestamp_units(value: Any, batch_size: int) -> list[str]:
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) != batch_size:
            raise PipelineContractError(
                f"pts_unit must be a string or [B]={batch_size} sequence, got {value!r}"
            )
        if not all(isinstance(item, str) for item in value):
            raise PipelineContractError(f"pts_unit values must be strings, got {value!r}")
        return list(value)
    raise PipelineContractError(f"pts_unit must be a string or [B] sequence, got {value!r}")


def _timestamp_scale(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return 1.0
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return 1e-3
    raise PipelineContractError(
        f"unsupported pts_unit {unit!r}; expected seconds or milliseconds"
    )


def _normalize_timestamps(
    pts: Any,
    pts_unit: Any,
    batch_size: int,
) -> tuple[Any, str | None]:
    if pts is None:
        if pts_unit is not None:
            _timestamp_units(pts_unit, batch_size)
        return None, None
    if pts_unit is None:
        return pts, "seconds"
    if not torch.is_tensor(pts):
        raise PipelineContractError("pts must be a tensor when pts_unit is provided")
    if not pts.is_floating_point():
        pts = pts.float()
    units = _timestamp_units(pts_unit, batch_size)
    scales = torch.tensor(
        [_timestamp_scale(unit) for unit in units],
        dtype=pts.dtype,
        device=pts.device,
    ).view(batch_size, 1, 1)
    return pts * scales, "seconds"


def _coordinate_space(value: Any, batch_size: int) -> str:
    if value is None:
        return "pixel"
    if isinstance(value, str):
        coordinate_space = value
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) != batch_size or not all(isinstance(item, str) for item in value):
            raise PipelineContractError(
                "box_coordinate_space must be a string or a [B] string sequence"
            )
        unique_values = set(value)
        if len(unique_values) != 1:
            raise PipelineContractError(
                f"mixed box coordinate spaces are not supported: {sorted(unique_values)}"
            )
        coordinate_space = next(iter(unique_values))
    else:
        raise PipelineContractError(
            "box_coordinate_space must be a string or a [B] string sequence"
        )
    if not coordinate_space.strip():
        raise PipelineContractError("box_coordinate_space must not be empty")
    return coordinate_space


@dataclass(frozen=True)
class ConditionBundle:
    """Conditional and optional unconditional model inputs for CFG."""

    conditional: Any
    unconditional: Any | None = None


@dataclass(frozen=True)
class ThermalVideoBatch:
    """Canonical view of the current loader output.

    ``images`` and ``box_condition_images`` retain the loader's
    ``[B,T,V,C,H,W]`` layout.  ``pts`` is normalized to seconds when the
    loader declares ``pts_unit`` as seconds or milliseconds; legacy batches
    without that field are interpreted as seconds.  Box coordinates remain in
    the coordinate space declared by ``box_coordinate_space``; model-specific
    normalization belongs to the injected model adapter because the current
    loader preserves nested per-object annotation trees.
    """

    images: Any | None
    box_condition_images: Any | None
    boxes: Any | None = None
    labels: Any | None = None
    track_ids: Any | None = None
    sample_ids: Any | None = None
    bbox_available: Any | None = None
    condition_valid: Any | None = None
    frame_valid: Any | None = None
    pts: Any | None = None
    pts_unit: str | None = "seconds"
    fps: Any | None = None
    box_image_sizes: Any | None = None
    box_coordinate_space: str = "pixel"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_loader(
        cls,
        batch: Mapping[str, Any] | "ThermalVideoBatch",
        *,
        device: Any | None = None,
        require_images: bool = False,
    ) -> "ThermalVideoBatch":
        _require_torch()
        if isinstance(batch, cls):
            result = batch.to(device) if device is not None else batch
            result.validate(require_images=require_images)
            return result
        if not isinstance(batch, Mapping):
            raise PipelineContractError(f"loader batch must be a mapping, got {type(batch)!r}")

        images = batch.get("vae_images", batch.get("images"))
        conditions = batch.get("box_condition_images", batch.get("bbox_condition_images"))
        if require_images and images is None:
            raise PipelineContractError("training requires batch['vae_images'] or batch['images']")

        leading_shape: tuple[int, ...] | None = None
        for value in (images, conditions):
            value_shape = _shape(value)
            if value_shape is not None:
                leading_shape = value_shape[:3]
                break
        if leading_shape is None:
            for key in (
                "frame_valid",
                "bbox_available",
                "condition_valid",
                "pts",
                "box_image_sizes",
                "boxes",
                "labels",
                "track_ids",
                "sample_ids",
            ):
                value = batch.get(key)
                value_shape = _shape(value) or _nested_shape(value, levels=3)
                if value_shape is not None and len(value_shape) >= 3:
                    leading_shape = value_shape[:3]
                    break
        if leading_shape is None:
            raise PipelineContractError(
                "cannot infer [B,T,V]; provide vae_images, box_condition_images, "
                "or a [B,T,V] metadata field"
            )

        pts = batch.get("pts")
        source_pts_unit = batch.get("pts_unit")
        pts, pts_unit = _normalize_timestamps(pts, source_pts_unit, leading_shape[0])
        known = {
            "vae_images",
            "images",
            "box_condition_images",
            "bbox_condition_images",
            "boxes",
            "labels",
            "track_ids",
            "sample_ids",
            "bbox_available",
            "condition_valid",
            "frame_valid",
            "pts",
            "pts_unit",
            "fps",
            "box_image_sizes",
            "box_coordinate_space",
            "metadata",
        }
        metadata = batch.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise PipelineContractError("batch['metadata'] must be a mapping")
        metadata = dict(metadata)
        if source_pts_unit is not None:
            metadata.setdefault("source_pts_unit", source_pts_unit)
        result = cls(
            images=images,
            box_condition_images=conditions,
            boxes=batch.get("boxes"),
            labels=batch.get("labels"),
            track_ids=batch.get("track_ids"),
            sample_ids=batch.get("sample_ids"),
            bbox_available=batch.get("bbox_available"),
            condition_valid=batch.get("condition_valid"),
            frame_valid=batch.get("frame_valid"),
            pts=pts,
            pts_unit=pts_unit,
            fps=batch.get("fps"),
            box_image_sizes=batch.get("box_image_sizes"),
            box_coordinate_space=_coordinate_space(
                batch.get("box_coordinate_space", "pixel"),
                leading_shape[0],
            ),
            metadata=metadata,
            extra={key: value for key, value in batch.items() if key not in known},
        )
        result.validate(require_images=require_images)
        return result.to(device) if device is not None else result

    @property
    def leading_shape(self) -> tuple[int, int, int]:
        for value in (self.images, self.box_condition_images, self.box_image_sizes):
            value_shape = _shape(value)
            if value_shape is not None and len(value_shape) >= 3:
                return value_shape[:3]  # type: ignore[return-value]
        for value in (
            self.frame_valid,
            self.bbox_available,
            self.condition_valid,
            self.pts,
            self.boxes,
            self.labels,
            self.track_ids,
            self.sample_ids,
        ):
            value_shape = _shape(value) or _nested_shape(value, levels=3)
            if value_shape is not None and len(value_shape) >= 3:
                return value_shape[:3]  # type: ignore[return-value]
        raise PipelineContractError("batch has no inferable [B,T,V] shape")

    @property
    def batch_size(self) -> int:
        return self.leading_shape[0]

    @property
    def sequence_length(self) -> int:
        return self.leading_shape[1]

    @property
    def view_count(self) -> int:
        return self.leading_shape[2]

    def validate(self, *, require_images: bool = False) -> None:
        _require_torch()
        if require_images and self.images is None:
            raise PipelineContractError("images are required for this pipeline operation")
        leading = self.leading_shape
        if min(leading) < 1:
            raise PipelineContractError(f"empty video dimensions: {leading}")

        image_shape = _shape(self.images)
        if self.images is not None:
            if image_shape is None or len(image_shape) != 6:
                raise PipelineContractError(
                    f"images must have shape [B,T,V,C,H,W], got {image_shape}"
                )
            if image_shape[:3] != leading:
                raise PipelineContractError(
                    f"images leading shape differs from {leading}: {image_shape}"
                )
            _check_finite("images", self.images)
            if self.images.numel() and (
                float(self.images.detach().amin()) < 0.0
                or float(self.images.detach().amax()) > 1.0
            ):
                raise PipelineContractError("images must be in [0,1]")

        condition_shape = _shape(self.box_condition_images)
        if self.box_condition_images is not None:
            if condition_shape is None or len(condition_shape) != 6:
                raise PipelineContractError(
                    "box_condition_images must have shape [B,T,V,C,H,W], "
                    f"got {condition_shape}"
                )
            if condition_shape[:3] != leading:
                raise PipelineContractError(
                    "box_condition_images must share [B,T,V] with images: "
                    f"{condition_shape} versus {image_shape}"
                )
            if image_shape is not None and condition_shape[-2:] != image_shape[-2:]:
                raise PipelineContractError(
                    "box_condition_images must share [H,W] with images: "
                    f"{condition_shape} versus {image_shape}"
                )
            _check_finite("box_condition_images", self.box_condition_images)
            if self.box_condition_images.numel() and (
                float(self.box_condition_images.detach().amin()) < 0.0
                or float(self.box_condition_images.detach().amax()) > 1.0
            ):
                raise PipelineContractError("box_condition_images must be in [0,1]")

        for name in ("bbox_available", "condition_valid", "frame_valid"):
            value = getattr(self, name)
            if value is None:
                continue
            value_shape = _shape(value)
            if value_shape != leading:
                raise PipelineContractError(
                    f"{name} must have shape {leading}, got {value_shape}"
                )
            _check_finite(name, value)

        if self.pts is not None:
            if self.pts_unit != "seconds":
                raise PipelineContractError(
                    "pts must be normalized to seconds; provide batch['pts_unit'] "
                    "as seconds or milliseconds"
                )
            value_shape = _shape(self.pts)
            if value_shape != leading:
                raise PipelineContractError(
                    f"pts must have shape {leading}, got {value_shape}"
                )
            _check_finite("pts", self.pts)

        if self.fps is not None:
            fps_shape = _shape(self.fps)
            if fps_shape not in {(), (leading[0],)}:
                raise PipelineContractError(
                    f"fps must be scalar or [B]={leading[0]}, got {fps_shape}"
                )
            _check_finite("fps", self.fps)

        for name in ("boxes", "labels", "track_ids", "sample_ids"):
            value = getattr(self, name)
            if value is None:
                continue
            value_shape = _nested_shape(value, levels=3)
            if value_shape != leading:
                raise PipelineContractError(
                    f"{name} must have nested shape {leading}, got {value_shape}"
                )
            if name == "boxes":
                _validate_box_tree(value)
        if self.box_image_sizes is not None:
            size_shape = _shape(self.box_image_sizes)
            if size_shape is None or size_shape[:3] != leading or size_shape[-1] != 2:
                raise PipelineContractError(
                    f"box_image_sizes must have shape [B,T,V,2], got {size_shape}"
                )
            _check_finite("box_image_sizes", self.box_image_sizes)
            if self.box_image_sizes.numel() and float(self.box_image_sizes.detach().amin()) <= 0.0:
                raise PipelineContractError("box_image_sizes must be positive")
        if not self.box_coordinate_space:
            raise PipelineContractError("box_coordinate_space must not be empty")

    def to(self, device: Any) -> "ThermalVideoBatch":
        _require_torch()
        return replace(
            self,
            images=_move_value(self.images, device),
            box_condition_images=_move_value(self.box_condition_images, device),
            boxes=_move_value(self.boxes, device),
            labels=_move_value(self.labels, device),
            track_ids=_move_value(self.track_ids, device),
            bbox_available=_move_value(self.bbox_available, device),
            condition_valid=_move_value(self.condition_valid, device),
            frame_valid=_move_value(self.frame_valid, device),
            pts=_move_value(self.pts, device),
            fps=_move_value(self.fps, device),
            box_image_sizes=_move_value(self.box_image_sizes, device),
            extra=_move_value(dict(self.extra), device),
        )


@dataclass(frozen=True)
class GenerationResult:
    """Latent and decoded outputs of one jointly generated video batch."""

    latents: Any
    frames: Any | None
    batch: ThermalVideoBatch
    num_steps: int
    guidance_scale: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ThermalVideoModel(Protocol):
    """Interface implemented later by the concrete video model."""

    training: bool

    latent_dtype: Any

    def parameters(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any: ...

    def encode_video(self, batch: ThermalVideoBatch) -> Any: ...

    def decode_video(self, latents: Any) -> Any: ...

    def latent_shape(self, batch: ThermalVideoBatch) -> Sequence[int]: ...

    def sample_timesteps(self, batch: ThermalVideoBatch, generator: Any) -> Any: ...

    def add_noise(self, latents: Any, noise: Any, timesteps: Any) -> Any: ...

    def training_target(self, latents: Any, noise: Any, timesteps: Any) -> Any: ...

    def prepare_conditions(
        self,
        batch: ThermalVideoBatch,
        *,
        training: bool,
        generator: Any,
    ) -> ConditionBundle: ...

    def forward_video(
        self,
        noisy_latents: Any,
        timesteps: Any,
        conditions: Any,
        batch: ThermalVideoBatch,
    ) -> Any: ...

    def inference_timesteps(self, num_steps: int, device: Any) -> Sequence[Any]: ...

    def scheduler_step(
        self,
        model_output: Any,
        timestep: Any,
        latents: Any,
        generator: Any,
    ) -> Any: ...


class VideoWriter(Protocol):
    """Model-specific writer for TIFF/video/metadata output."""

    def write(self, result: GenerationResult, output_dir: str | Any) -> Any: ...


class VideoEvaluator(Protocol):
    """Evaluator callback consuming one generated result and its batch."""

    def __call__(
        self,
        result: GenerationResult,
        batch: ThermalVideoBatch,
    ) -> Mapping[str, float]: ...


__all__ = [
    "ConditionBundle",
    "GenerationResult",
    "PipelineContractError",
    "ThermalVideoBatch",
    "ThermalVideoModel",
    "VideoEvaluator",
    "VideoWriter",
]
