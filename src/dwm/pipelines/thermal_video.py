"""Model-agnostic pipelines for bbox-conditioned thermal video generation.

The pipeline owns the OpenDWM-style lifecycle and consumes the local loader's
``[B,T,V,...]`` batch contract.  A concrete model supplies codec, conditioning,
temporal denoiser and scheduler behavior through ``ThermalVideoModel``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - config-only environments
    torch = None  # type: ignore[assignment]

from .contracts import (
    ConditionBundle,
    GenerationResult,
    PipelineContractError,
    ThermalVideoBatch,
    ThermalVideoModel,
    VideoEvaluator,
    VideoWriter,
)
from .objectives import LossResult, MaskedDiffusionMSE, VideoObjective


class PipelineStateError(RuntimeError):
    """Raised when a pipeline operation would create an invalid state."""


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("thermal video pipelines require PyTorch")
    return torch


def _resolve_device(model: ThermalVideoModel, device: Any | None) -> Any:
    _require_torch()
    if device is not None:
        return torch.device(device)
    candidate = getattr(model, "device", None)
    if candidate is not None:
        return torch.device(candidate)
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise PipelineContractError("model must expose at least one parameter") from error


def _world_size() -> int:
    if (
        torch is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return torch.distributed.get_world_size()
    return 1


def _resolve_latent_dtype(model: ThermalVideoModel) -> Any:
    candidate = getattr(model, "latent_dtype", None)
    if candidate is not None:
        return candidate
    try:
        return next(model.parameters()).dtype
    except StopIteration as error:
        raise PipelineContractError("model must expose a latent_dtype or a parameter") from error


def _rank() -> int:
    if (
        torch is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return torch.distributed.get_rank()
    return 0


def _all_reduce_sum(value: Any) -> Any:
    if (
        torch is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(value)
    return value


def _ensure_video_tensor(name: str, value: Any, leading: tuple[int, int, int]) -> None:
    if torch is None or not torch.is_tensor(value):
        raise PipelineContractError(f"{name} must be a tensor")
    if value.ndim != 6 or tuple(int(dimension) for dimension in value.shape[:3]) != leading:
        raise PipelineContractError(
            f"{name} must have [B,T,V,...] leading shape {leading}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise PipelineContractError(f"{name} contains non-finite values")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


class ThermalVideoPipeline:
    """OpenDWM-style lifecycle around an injected thermal video model.

    The model is intentionally not implemented here.  It must implement the
    ``ThermalVideoModel`` contract, including a genuinely joint video forward;
    flattening frames and sampling them independently is not a valid adapter.
    """

    def __init__(
        self,
        model: ThermalVideoModel,
        *,
        optimizer: Any | None = None,
        lr_scheduler: Any | None = None,
        scaler: Any | None = None,
        objective: VideoObjective | None = None,
        device: Any | None = None,
        accumulation_steps: int = 1,
        autocast_dtype: Any | None = None,
        max_grad_norm: float | None = None,
        seed: int | None = None,
        generator: Any | None = None,
    ) -> None:
        _require_torch()
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")
        self.model = model
        self.device = _resolve_device(model, device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.objective = objective or MaskedDiffusionMSE()
        self.accumulation_steps = int(accumulation_steps)
        self.autocast_dtype = autocast_dtype
        self.max_grad_norm = max_grad_norm
        self.micro_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self._pending_micro_steps = 0
        self._pending_global_count = 0.0
        self._metric_history: list[dict[str, float]] = []

        if generator is None:
            self.generator = torch.Generator(device=self.device)
            if seed is not None:
                self.generator.manual_seed(int(seed))
        else:
            self.generator = generator
            if seed is not None:
                self.generator.manual_seed(int(seed))
        generator_device = torch.device(self.generator.device)
        if generator_device.type != self.device.type:
            raise ValueError(
                f"generator device {generator_device} differs from model device {self.device}"
            )
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    @contextlib.contextmanager
    def _autocast(self):
        if self.autocast_dtype is None:
            yield
            return
        if self.device.type not in {"cuda", "cpu"}:
            yield
            return
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            yield

    def _prepare_batch(
        self,
        batch: Mapping[str, Any] | ThermalVideoBatch,
        *,
        require_images: bool,
    ) -> ThermalVideoBatch:
        return ThermalVideoBatch.from_loader(
            batch,
            device=self.device,
            require_images=require_images,
        )

    def _global_count(self, denominator: Any) -> Any:
        if not torch.is_tensor(denominator) or denominator.ndim != 0:
            raise PipelineContractError("objective denominator must be a scalar tensor")
        count = denominator.detach().clone()
        _all_reduce_sum(count)
        if float(count) <= 0.0 or not math.isfinite(float(count)):
            raise PipelineStateError("global objective denominator is not positive")
        return count

    def _global_mean(self, numerator: Any, denominator: Any) -> float:
        value = numerator.detach().clone()
        _all_reduce_sum(value)
        return float((value / denominator).detach())

    def _validate_model_output(
        self,
        name: str,
        value: Any,
        reference: Any,
        leading: tuple[int, int, int],
    ) -> None:
        _ensure_video_tensor(name, value, leading)
        if tuple(value.shape) != tuple(reference.shape):
            raise PipelineContractError(
                f"{name} shape {tuple(value.shape)} differs from reference {tuple(reference.shape)}"
            )

    def _finish_update(self) -> None:
        if self._pending_micro_steps == 0:
            return
        if self.optimizer is None:
            raise PipelineStateError("cannot update without an optimizer")
        if self._pending_global_count <= 0.0:
            raise PipelineStateError("pending objective count is not positive")
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        normalization = float(self._pending_global_count)
        for parameter in self.model.parameters():
            gradient = getattr(parameter, "grad", None)
            if gradient is not None:
                gradient.div_(normalization)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.max_grad_norm),
            )
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step += 1
        self._pending_micro_steps = 0
        self._pending_global_count = 0.0

    def train_step(
        self,
        batch: Mapping[str, Any] | ThermalVideoBatch,
        global_step: int | None = None,
    ) -> dict[str, float | int | bool]:
        """Consume one loader batch and optionally perform an optimizer update."""

        if self.optimizer is None:
            raise PipelineStateError("train_step requires an optimizer")
        prepared = self._prepare_batch(batch, require_images=True)
        self.model.train()
        leading = prepared.leading_shape
        with self._autocast():
            latents = self.model.encode_video(prepared) # encode to latents
            _ensure_video_tensor("encoded latents", latents, leading)
            noise = torch.randn(
                tuple(latents.shape),
                device=latents.device,
                dtype=latents.dtype,
                generator=self.generator,
            )
            timesteps = self.model.sample_timesteps(prepared, self.generator)
            noisy_latents = self.model.add_noise(latents, noise, timesteps)
            target = self.model.training_target(latents, noise, timesteps)
            self._validate_model_output("target", target, latents, leading)
            conditions = self.model.prepare_conditions(
                prepared,
                training=True,
                generator=self.generator,
            )
            if not isinstance(conditions, ConditionBundle):
                raise PipelineContractError("prepare_conditions must return ConditionBundle")
            prediction = self.model.forward_video(
                noisy_latents,
                timesteps,
                conditions.conditional,
                prepared,
            )
            self._validate_model_output("prediction", prediction, target, leading)
            valid_mask = prepared.frame_valid
            result: LossResult = self.objective(prediction, target, valid_mask)
            if not torch.is_tensor(result.numerator) or result.numerator.ndim != 0:
                raise PipelineContractError("objective numerator must be a scalar tensor")
            global_count = self._global_count(result.denominator)
            backward_value = result.numerator * float(_world_size())
            if self.scaler is None:
                backward_value.backward()
            else:
                self.scaler.scale(backward_value).backward()

        self.micro_step += 1
        self._pending_micro_steps += 1
        self._pending_global_count += float(global_count)
        updated = self._pending_micro_steps >= self.accumulation_steps
        if updated:
            self._finish_update()
        global_loss = self._global_mean(result.numerator, global_count)
        metrics: dict[str, float | int | bool] = {
            **result.metrics,
            "loss": global_loss,
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "updated": updated,
            "valid_units": int(global_count),
        }
        if "diffusion_mse" in metrics:
            metrics["diffusion_mse"] = global_loss
        if global_step is not None:
            metrics["global_step"] = int(global_step)
        self._metric_history.append(
            {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        )
        return metrics

    def flush(self) -> bool:
        """Update a final partial accumulation group using its actual count."""

        had_pending = self._pending_micro_steps > 0
        self._finish_update()
        return had_pending

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def log(self, global_step: int | None = None, *, reset: bool = True) -> dict[str, float]:
        """Return mean numeric metrics collected since the previous log call."""

        if not self._metric_history:
            return {}
        keys = sorted({key for item in self._metric_history for key in item})
        report = {
            key: sum(item[key] for item in self._metric_history if key in item)
            / sum(key in item for item in self._metric_history)
            for key in keys
        }
        if global_step is not None:
            report["global_step"] = float(global_step)
        if reset:
            self._metric_history.clear()
        return report

    def _make_generator(self, seed: int | None) -> Any:
        if seed is None:
            return self.generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def inference_pipeline(
        self,
        batch: Mapping[str, Any] | ThermalVideoBatch,
        *,
        num_steps: int,
        guidance_scale: float = 1.0,
        initial_latents: Any | None = None,
        output_type: str = "tensor",
        seed: int | None = None,
    ) -> GenerationResult:
        """Jointly denoise one ``[B,T,V]`` condition batch."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if output_type not in {"tensor", "latent"}:
            raise ValueError("output_type must be 'tensor' or 'latent'")
        prepared = self._prepare_batch(batch, require_images=False)
        leading = prepared.leading_shape
        shape = tuple(int(dimension) for dimension in self.model.latent_shape(prepared))
        if len(shape) != 6 or shape[:3] != leading or min(shape) < 1:
            raise PipelineContractError(
                f"model.latent_shape must return [B,T,V,C,H,W] for {leading}, got {shape}"
            )
        generator = self._make_generator(seed)
        latent_dtype = _resolve_latent_dtype(self.model)
        if initial_latents is None:
            latents = torch.randn(
                shape,
                device=self.device,
                dtype=latent_dtype,
                generator=generator,
            )
        else:
            if not torch.is_tensor(initial_latents) or tuple(initial_latents.shape) != shape:
                raise PipelineContractError(
                    f"initial_latents must have shape {shape}, got "
                    f"{getattr(initial_latents, 'shape', None)}"
                )
            latents = initial_latents.to(device=self.device, dtype=latent_dtype)
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            conditions = self.model.prepare_conditions(
                prepared,
                training=False,
                generator=generator,
            )
            if not isinstance(conditions, ConditionBundle):
                raise PipelineContractError("prepare_conditions must return ConditionBundle")
            timesteps = tuple(self.model.inference_timesteps(num_steps, self.device))
            if len(timesteps) != num_steps:
                raise PipelineContractError(
                    f"inference_timesteps returned {len(timesteps)} values, expected {num_steps}"
                )

            with torch.no_grad():
                for timestep in timesteps:
                    conditional = self.model.forward_video(
                        latents,
                        timestep,
                        conditions.conditional,
                        prepared,
                    )
                    self._validate_model_output(
                        "conditional prediction",
                        conditional,
                        latents,
                        leading,
                    )
                    if guidance_scale != 1.0:
                        if conditions.unconditional is None:
                            raise PipelineContractError(
                                "guidance_scale != 1 requires unconditional conditions"
                            )
                        unconditional = self.model.forward_video(
                            latents,
                            timestep,
                            conditions.unconditional,
                            prepared,
                        )
                        self._validate_model_output(
                            "unconditional prediction", unconditional, latents, leading
                        )
                        model_output = unconditional + guidance_scale * (
                            conditional - unconditional
                        )
                    else:
                        model_output = conditional
                    latents = self.model.scheduler_step(
                        model_output,
                        timestep,
                        latents,
                        generator,
                    )
                    _ensure_video_tensor("updated latents", latents, leading)
                    if tuple(latents.shape) != shape:
                        raise PipelineContractError(
                            f"scheduler changed latent shape from {shape} to {tuple(latents.shape)}"
                        )
                frames = None if output_type == "latent" else self.model.decode_video(latents)
                if frames is not None:
                    _ensure_video_tensor("decoded frames", frames, leading)
        finally:
            if was_training:
                self.model.train()

        metadata = {
            "shape": shape,
            "num_steps": int(num_steps),
            "guidance_scale": float(guidance_scale),
            "box_coordinate_space": prepared.box_coordinate_space,
        }
        return GenerationResult(
            latents=latents,
            frames=frames,
            batch=prepared,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            metadata=metadata,
        )

    def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
        """Alias for :meth:`inference_pipeline`."""

        return self.inference_pipeline(*args, **kwargs)

    def preview_pipeline(
        self,
        batch: Mapping[str, Any] | ThermalVideoBatch,
        output_dir: str | Path | None = None,
        *,
        num_steps: int,
        guidance_scale: float = 1.0,
        writer: VideoWriter | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate a preview and optionally delegate task-specific output writing."""

        if output_dir is not None and writer is None:
            raise ValueError("output_dir requires a model-specific VideoWriter")
        result = self.inference_pipeline(
            batch,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if writer is not None:
            if output_dir is None:
                raise ValueError("VideoWriter requires output_dir")
            writer.write(result, output_dir)
        return result

    def evaluate_pipeline(
        self,
        dataloader: Any,
        evaluator: VideoEvaluator,
        *,
        num_steps: int,
        guidance_scale: float = 1.0,
        max_batches: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate validation clips and aggregate evaluator-returned scalars."""

        if evaluator is None:
            raise ValueError("evaluate_pipeline requires an evaluator")
        reports: list[Mapping[str, float]] = []
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            prepared = self._prepare_batch(raw_batch, require_images=False)
            result = self.inference_pipeline(
                prepared,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=None if seed is None else int(seed) + batch_index,
            )
            report = evaluator(result, prepared)
            if not isinstance(report, Mapping):
                raise TypeError("evaluator must return a mapping of scalar metrics")
            if any(not isinstance(value, (int, float)) for value in report.values()):
                raise TypeError("evaluator metrics must be numeric")
            reports.append(report)
        if not reports:
            raise PipelineStateError("evaluation dataloader produced no batches")
        keys = sorted({key for report in reports for key in report})
        metrics = {
            key: sum(float(report[key]) for report in reports if key in report)
            / sum(key in report for report in reports)
            for key in keys
        }
        return {"batches": len(reports), "metrics": metrics, "reports": reports}

    @staticmethod
    def _rank_checkpoint_path(path: Path) -> Path:
        if _world_size() == 1:
            return path
        suffix = path.suffix
        stem = path.name[: -len(suffix)] if suffix else path.name
        return path.with_name(f"{stem}.rank{_rank()}{suffix}")

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save a rank-local, update-boundary-complete pipeline checkpoint."""

        if self._pending_micro_steps:
            raise PipelineStateError(
                "save_checkpoint requires an optimizer boundary; call flush() first"
            )
        checkpoint = Path(path)
        target = self._rank_checkpoint_path(checkpoint)
        rng: dict[str, Any] = {
            "pipeline_generator": self.generator.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        payload: dict[str, Any] = {
            "format_version": 1,
            "model": dict(self.model.state_dict()),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "pipeline": {
                "micro_step": self.micro_step,
                "optimizer_step": self.optimizer_step,
                "epoch": self.epoch,
                "accumulation_steps": self.accumulation_steps,
                "pending_micro_steps": 0,
            },
            "rng": rng,
            "metadata": dict(metadata or {}),
        }
        _atomic_torch_save(payload, target)
        if _world_size() > 1:
            torch.distributed.barrier()
            if _rank() == 0:
                index_path = checkpoint.with_name(checkpoint.name + ".index.json")
                suffix = checkpoint.suffix
                stem = checkpoint.name[: -len(suffix)] if suffix else checkpoint.name
                index = {
                    "format_version": 1,
                    "world_size": _world_size(),
                    "files": [
                        str(checkpoint.with_name(f"{stem}.rank{rank}{suffix}"))
                        for rank in range(_world_size())
                    ],
                }
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
            torch.distributed.barrier()
        return target

    def load_checkpoint(self, path: str | Path) -> Mapping[str, Any]:
        """Restore model, optimizer, counters and random state from a checkpoint."""

        checkpoint = Path(path)
        target = self._rank_checkpoint_path(checkpoint)
        if not target.is_file() and checkpoint.is_file():
            target = checkpoint
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
            raise PipelineStateError(f"unsupported pipeline checkpoint: {target}")
        self.model.load_state_dict(payload["model"], strict=True)
        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
            _move_optimizer_state(self.optimizer, self.device)
        if self.lr_scheduler is not None and payload.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(payload["lr_scheduler"])
        if self.scaler is not None and payload.get("scaler") is not None:
            self.scaler.load_state_dict(payload["scaler"])
        state = payload.get("pipeline")
        if not isinstance(state, Mapping) or int(state.get("pending_micro_steps", 0)) != 0:
            raise PipelineStateError("checkpoint contains an unfinished accumulation group")
        self.micro_step = int(state.get("micro_step", 0))
        self.optimizer_step = int(state.get("optimizer_step", 0))
        self.epoch = int(state.get("epoch", 0))
        if int(state.get("accumulation_steps", self.accumulation_steps)) != self.accumulation_steps:
            raise PipelineStateError("checkpoint accumulation_steps differs from the pipeline")
        rng = payload.get("rng", {})
        if isinstance(rng, Mapping):
            if rng.get("pipeline_generator") is not None:
                self.generator.set_state(rng["pipeline_generator"])
            if rng.get("torch_cpu") is not None:
                torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        return payload.get("metadata", {})


__all__ = ["PipelineStateError", "ThermalVideoPipeline"]
