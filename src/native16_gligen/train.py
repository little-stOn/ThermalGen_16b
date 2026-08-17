from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from native16_gligen.amp_sync import synchronize_grad_scaler_overflow
from native16_gligen.attention_box_loss import install_grounding_attention_recorder
from native16_gligen.counterfactual_grounding_loss import (
    compute_counterfactual_grounding_loss,
    sample_counterfactual_boxes,
)
from native16_gligen.detail_loss import (
    DetailLossWeights,
    compute_detail_losses,
    prediction_to_x0,
    weighted_detail_loss,
)
from native16_gligen.ir_style_loss import compute_ir_style_losses, weighted_ir_style_loss
from native16_gligen.instance_fusion import compute_instance_normalized_loss
from native16_gligen.checkpoint import (
    collect_trainable_parameter_names,
    load_checkpoint_file,
    load_model_weights,
    restore_optimizer_state,
    save_checkpoint,
)
from native16_gligen.config import build_common_parser, load_config, save_config
from native16_gligen.model_stack import (
    ModelStack,
    load_checkpoint_layer_into,
    load_model_stack,
    load_stack_into,
    verify_base_model,
)
from native16_gligen.objectives import (
    bounded_contribution,
    pixel_target_for_decoded,
    style_target,
)
from native16_gligen.data import (
    DistributedWeightedSampler,
    build_dataset,
    build_dataset_sample_weights,
    classify_layout_bucket,
    collate_batch,
)
from native16_gligen.distributed import (
    cleanup_distributed,
    initialize_distributed,
    reduce_mean,
    validate_v100_precision,
    wrap_fsdp,
)
from native16_gligen.model import GLIGENDenoisingCore, build_system, configure_trainable, resolve_dtype
from native16_gligen.native16_vae import (
    Native16LatentAffine,
    assert_clean_official_parent,
    read_native16_manifest,
)


def update_latest_checkpoint(source: Path, latest: Path) -> None:
    """Atomically point latest at a same-filesystem checkpoint without duplicating it."""

    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_suffix(latest.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(latest)


def validate_native16_isolation(cfg: Any, output_dir: Path) -> dict[str, Any] | None:
    input_mode = str(getattr(cfg.data, "input_mode", "rgb8")).lower()
    if input_mode not in {"native16", "native16_bridge"}:
        return None
    parent = assert_clean_official_parent(str(cfg.model.pretrained_model))
    if input_mode == "native16_bridge":
        bridge_value = getattr(cfg.model, "radiometric_bridge_checkpoint", None)
        if bridge_value in (None, "", "null", "false", False):
            raise ValueError("native16_bridge requires model.radiometric_bridge_checkpoint")
        bridge_path = Path(str(bridge_value)).expanduser().resolve()
        payload = torch.load(bridge_path, map_location="cpu", weights_only=False)
        bridge_manifest = payload.get("manifest", {})
        if bridge_manifest.get("legacy_8bit_checkpoint") is not False:
            raise ValueError("native16 bridge checkpoint is not marked independent")
        return {
            "schema_version": 1,
            "branch": "native16_bridge",
            "bit_depth": 16,
            "pixel_channels": 3,
            "output_channels": 1,
            "official_gligen_parent": str(parent),
            "bridge_checkpoint": str(bridge_path),
            "bridge_training_step": payload.get("step"),
            "legacy_8bit_checkpoint": False,
        }
    vae_path = Path(str(cfg.model.vae_pretrained_model)).resolve()
    _, vae_manifest = read_native16_manifest(vae_path)
    for key in ("base_checkpoint", "init_checkpoint"):
        value = getattr(cfg.train, key, None)
        if value is not None and str(value).lower() not in {"", "none", "false"}:
            raise ValueError(f"native16 pilot refuses train.{key}; initialize only from official GLIGEN")
    resume = getattr(cfg.train, "resume", None)
    if resume is not None and str(resume).lower() not in {"", "none", "false"}:
        existing = output_dir / "native16_run_manifest.json"
        if not existing.is_file():
            raise FileNotFoundError("native16 resume requires its original run manifest")
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("vae_path") != str(vae_path):
            raise ValueError("native16 resume VAE differs from the original run")
    calibration_path = getattr(cfg.model, "latent_calibration_path", None)
    calibration_manifest: dict[str, Any] = {"enabled": False}
    if calibration_path not in (None, "", "null", "false", False):
        calibration_file = Path(str(calibration_path)).expanduser().resolve()
        calibration = Native16LatentAffine.from_json(calibration_file)
        calibration_manifest = {
            "enabled": True,
            "path": str(calibration_file),
            "target": calibration.target,
            "scale": list(calibration.scale),
            "shift": list(calibration.shift),
            "source_vae": calibration.source_vae,
        }
    return {
        "schema_version": 1,
        "branch": "native16",
        "bit_depth": 16,
        "pixel_channels": 1,
        "official_gligen_parent": str(parent),
        "vae_path": str(vae_path),
        "vae_parent": vae_manifest.get("parent"),
        "vae_training_step": vae_manifest.get("training_step"),
        "latent_calibration": calibration_manifest,
        "legacy_8bit_checkpoint": False,
    }


def seed_everything(seed: int, rank: int) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return None


def move_batch_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    tensor_keys = [
        "pixel_values",
        "boxes",
        "labels",
        "box_mask",
        "reference_pixel_values",
        "reference_present",
    ]
    for key in tensor_keys:
        moved[key] = batch[key].to(device=device, non_blocking=True)
    if moved["pixel_values"].device != device:
        raise RuntimeError("Failed to move training batch to the selected device")
    return moved


def apply_condition_dropout(batch: dict[str, Any], cfg: Any, device: torch.device) -> list[str]:
    batch_size = batch["boxes"].shape[0]
    box_drop = torch.rand(batch_size, device=device) < float(cfg.box_dropout)
    all_drop = torch.rand(batch_size, device=device) < float(cfg.cfg_dropout)
    batch["box_mask"] = batch["box_mask"] & (~box_drop[:, None]) & (~all_drop[:, None])
    prompts = []
    for index, prompt in enumerate(batch["prompt"]):
        text_drop = random.random() < float(cfg.text_dropout)
        prompts.append("" if text_drop or bool(all_drop[index].item()) else prompt)
    if len(prompts) != batch_size:
        raise RuntimeError("Prompt dropout changed the batch size")
    return prompts


def perturb_condition_boxes(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Apply small box jitter only to the condition path.

    Pixel, diffusion-region, and attention targets keep the original boxes.
    This prevents the denoiser from using an exact rectangle edge as a visual
    shortcut while preserving the requested object center and scale.
    """

    probability = float(getattr(cfg, "box_jitter_probability", 0.0))
    center_std = float(getattr(cfg, "box_jitter_center_std", 0.0))
    scale_std = float(getattr(cfg, "box_jitter_scale_std", 0.0))
    if probability <= 0.0 or (center_std <= 0.0 and scale_std <= 0.0):
        return boxes
    active = box_mask.bool() & (torch.rand_like(boxes[..., 0]) < probability)
    safe = boxes.float().clamp(0.0, 1.0)
    x1, y1, x2, y2 = safe.unbind(dim=-1)
    width = (x2 - x1).clamp_min(1.0e-4)
    height = (y2 - y1).clamp_min(1.0e-4)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    shift_x = torch.randn_like(center_x).clamp(-2.0, 2.0) * center_std * width
    shift_y = torch.randn_like(center_y).clamp(-2.0, 2.0) * center_std * height
    scale_x = torch.exp(torch.randn_like(width).clamp(-2.0, 2.0) * scale_std)
    scale_y = torch.exp(torch.randn_like(height).clamp(-2.0, 2.0) * scale_std)
    new_center_x = center_x + shift_x
    new_center_y = center_y + shift_y
    new_width = width * scale_x
    new_height = height * scale_y
    jittered = torch.stack(
        (
            new_center_x - 0.5 * new_width,
            new_center_y - 0.5 * new_height,
            new_center_x + 0.5 * new_width,
            new_center_y + 0.5 * new_height,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    # Clipping at the image boundary may collapse a box. Keep the original in
    # that rare case instead of passing invalid geometry to GLIGEN.
    valid_jitter = (jittered[..., 2] - jittered[..., 0] > 1.0e-4) & (
        jittered[..., 3] - jittered[..., 1] > 1.0e-4
    )
    use_jitter = active & valid_jitter
    return torch.where(use_jitter[..., None], jittered.to(boxes.dtype), boxes)


def resolve_resume_path(output_dir: Path, resume_value: Any) -> Path | None:
    if resume_value is None or str(resume_value).lower() in {"", "none", "false"}:
        return None
    if str(resume_value).lower() == "latest":
        candidate = output_dir / "checkpoints" / "latest.pt"
    else:
        candidate = Path(str(resume_value))
    if not candidate.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return candidate


def resolve_init_checkpoint(init_value: Any) -> Path | None:
    if init_value is None or str(init_value).lower() in {"", "none", "false"}:
        return None
    candidate = Path(str(init_value))
    if not candidate.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {candidate}")
    return candidate


def print_parameter_summary(stats: dict[str, int], model_type: str, mode: str) -> None:
    million = 1_000_000.0
    trainable = stats["trainable"] / million
    frozen = stats["frozen"] / million
    total = stats["total"] / million
    print(f"model={model_type} mode={mode}")
    print(f"trainable={trainable:.2f}M frozen={frozen:.2f}M total={total:.2f}M")
    if mode == "full_unet":
        print("warning: full_unet consumes substantially more memory than grounding_only")
    return None


def parameter_group_name(parameter_name: str) -> str:
    if "position_net" in parameter_name or ".fuser." in parameter_name:
        return "grounding"
    if ".attn2." in parameter_name:
        return "cross_attention"
    return "base_unet"


def build_optimizer_parameter_groups(
    core: torch.nn.Module, cfg: Any
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    configured = getattr(cfg.train, "learning_rate_groups", None)
    if configured is None:
        parameters = [parameter for parameter in core.parameters() if parameter.requires_grad]
        return ([{"params": parameters, "lr": float(cfg.train.learning_rate), "name": "trainable"}], {
            "trainable": {"parameters": float(sum(parameter.numel() for parameter in parameters)), "lr": float(cfg.train.learning_rate)}
        })
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "base_unet": [],
        "cross_attention": [],
        "grounding": [],
    }
    for name, parameter in core.named_parameters():
        if parameter.requires_grad:
            grouped[parameter_group_name(name)].append(parameter)
    optimizer_groups: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}
    for name in ("base_unet", "cross_attention", "grounding"):
        parameters = grouped[name]
        if not parameters:
            continue
        learning_rate = float(configured[name])
        optimizer_groups.append({"params": parameters, "lr": learning_rate, "name": name})
        summary[name] = {
            "parameters": float(sum(parameter.numel() for parameter in parameters)),
            "lr": learning_rate,
        }
    if not optimizer_groups:
        raise RuntimeError("Optimizer parameter grouping selected no parameters")
    return optimizer_groups, summary


def compute_snr(noise_scheduler: Any, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha = alphas_cumprod[timesteps]
    return alpha / (1.0 - alpha).clamp_min(1e-8)


def compute_diffusion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Any,
    prediction_type: str,
    min_snr_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_sample = F.mse_loss(prediction.float(), target.float(), reduction="none")
    per_sample = per_sample.mean(dim=tuple(range(1, per_sample.ndim)))
    unweighted = per_sample.mean()
    if min_snr_gamma <= 0:
        weights = torch.ones_like(per_sample)
    else:
        snr = compute_snr(noise_scheduler, timesteps)
        clipped = torch.minimum(snr, torch.full_like(snr, float(min_snr_gamma)))
        if prediction_type == "epsilon":
            weights = clipped / snr.clamp_min(1e-8)
        elif prediction_type == "v_prediction":
            weights = clipped / (snr + 1.0)
        else:
            raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")
    weighted = (per_sample * weights).mean()
    return weighted, unweighted, weights.mean()


def build_bbox_weight_map(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    height: int,
    width: int,
    box_loss_weight: float,
) -> torch.Tensor:
    """Build a latent-cell occupancy map without losing small boxes."""
    if box_loss_weight < 1.0:
        raise ValueError("box_loss_weight must be at least 1")
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes with shape [B,M,4], got {tuple(boxes.shape)}")
    device = boxes.device
    dtype = boxes.dtype
    x0 = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width) / width
    x1 = torch.arange(width, device=device, dtype=dtype).add(1).view(1, 1, 1, width) / width
    y0 = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1) / height
    y1 = torch.arange(height, device=device, dtype=dtype).add(1).view(1, 1, height, 1) / height
    box_x0 = boxes[:, :, 0].view(-1, boxes.shape[1], 1, 1)
    box_y0 = boxes[:, :, 1].view(-1, boxes.shape[1], 1, 1)
    box_x1 = boxes[:, :, 2].view(-1, boxes.shape[1], 1, 1)
    box_y1 = boxes[:, :, 3].view(-1, boxes.shape[1], 1, 1)
    overlap_x = (torch.minimum(x1, box_x1) - torch.maximum(x0, box_x0)).clamp_min(0.0)
    overlap_y = (torch.minimum(y1, box_y1) - torch.maximum(y0, box_y0)).clamp_min(0.0)
    occupancy = overlap_x * overlap_y * float(height * width)
    occupancy = occupancy.clamp(0.0, 1.0)
    occupancy = occupancy * box_mask.to(dtype=dtype).view(-1, boxes.shape[1], 1, 1)
    union_occupancy = occupancy.amax(dim=1)
    return 1.0 + (float(box_loss_weight) - 1.0) * union_occupancy


def compute_bbox_weighted_diffusion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Any,
    prediction_type: str,
    min_snr_gamma: float,
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    box_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    element_loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
    region_map = build_bbox_weight_map(
        boxes,
        box_mask,
        int(element_loss.shape[-2]),
        int(element_loss.shape[-1]),
        float(box_loss_weight),
    )
    weighted_element_loss = element_loss * region_map.unsqueeze(1)
    per_sample = weighted_element_loss.mean(dim=tuple(range(1, weighted_element_loss.ndim)))
    unweighted_per_sample = element_loss.mean(dim=tuple(range(1, element_loss.ndim)))
    if min_snr_gamma <= 0:
        snr_weights = torch.ones_like(per_sample)
    else:
        snr = compute_snr(noise_scheduler, timesteps)
        clipped = torch.minimum(snr, torch.full_like(snr, float(min_snr_gamma)))
        if prediction_type == "epsilon":
            snr_weights = clipped / snr.clamp_min(1e-8)
        elif prediction_type == "v_prediction":
            snr_weights = clipped / (snr + 1.0)
        else:
            raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")
    return (
        (per_sample * snr_weights).mean(),
        unweighted_per_sample.mean(),
        snr_weights.mean(),
        (region_map > 1.0).to(torch.float32).mean(),
    )


def build_training_sampler(dataset: Any, cfg: Any, context: Any) -> tuple[Any, dict[str, float]]:
    sampling = getattr(cfg.data, "sampling", None)
    strategy = str(getattr(sampling, "strategy", "uniform")) if sampling is not None else "uniform"
    if strategy == "weighted":
        configured_weights = {str(name): float(value) for name, value in dict(sampling.dataset_weights).items()}
        layout_weights = {
            str(name): float(value)
            for name, value in dict(getattr(sampling, "layout_weights", {})).items()
        }
        sample_weights = build_dataset_sample_weights(
            dataset,
            configured_weights,
            layout_weights=layout_weights,
            resolution=int(cfg.model.width),
            prefer_small=bool(
                getattr(getattr(cfg.data, "instance_crop", None), "enabled", False)
            ),
        )
        sampler = DistributedWeightedSampler(
            sample_weights,
            samples_per_epoch=int(sampling.samples_per_epoch),
            num_replicas=int(context.world_size),
            rank=int(context.rank),
            seed=int(cfg.train.seed),
            replacement=bool(getattr(sampling, "replacement", True)),
        )
        probability = sample_weights / sample_weights.sum()
        expected: dict[str, float] = {}
        for index, record in enumerate(dataset.records):
            mass = float(probability[index])
            dataset_name = str(record.get("dataset", "unknown"))
            bucket = classify_layout_bucket(
                record,
                resolution=int(cfg.model.width),
                prefer_small=bool(
                    getattr(getattr(cfg.data, "instance_crop", None), "enabled", False)
                ),
            )
            expected[f"dataset:{dataset_name}"] = expected.get(f"dataset:{dataset_name}", 0.0) + mass
            expected[f"layout:{bucket}"] = expected.get(f"layout:{bucket}", 0.0) + mass
        return sampler, expected
    if strategy != "uniform":
        raise ValueError(f"Unsupported data.sampling.strategy: {strategy}")
    if context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            drop_last=bool(cfg.train.drop_last),
        )
        return sampler, {}
    return None, {}


def select_pixel_supervision_target(
    decoded: torch.Tensor,
    target: torch.Tensor,
    input_mode: str,
) -> torch.Tensor:
    """Align pixel-space supervision with the model's decoded output domain."""

    return pixel_target_for_decoded(decoded, target, input_mode)


def train() -> None:
    parser = build_common_parser("Fine-tune Diffusers GLIGEN grounding modules on SD1.4")
    args = parser.parse_args()
    cfg = load_config(args.cfg, args.set)
    context = initialize_distributed(int(cfg.distributed.timeout_minutes))
    validate_v100_precision(str(cfg.train.precision), context.device)
    seed_everything(int(cfg.train.seed), context.rank)
    cfg.data.height = int(cfg.model.height)
    cfg.data.width = int(cfg.model.width)
    cfg.data.ref_size = int(getattr(cfg.condition, "reference_size", 224))
    cfg.distributed.precision = str(cfg.train.precision)
    output_dir = Path(cfg.output.dir)
    native16_manifest = validate_native16_isolation(cfg, output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "resolved_config.yaml")
    dataset = build_dataset(cfg.data)
    if context.is_main:
        with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as handle:
            json.dump(dataset.class_to_idx, handle, indent=2, ensure_ascii=False)
        print(f"dataset={cfg.data.type} samples={len(dataset)} classes={len(dataset.class_to_idx)}")
    if args.dry_run:
        sample = dataset[0]
        if context.is_main:
            print({key: tuple(value.shape) if torch.is_tensor(value) else value for key, value in sample.items()})
        cleanup_distributed(context)
        return None
    sampler, expected_dataset_mix = build_training_sampler(dataset, cfg, context)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=context.device.type == "cuda",
        persistent_workers=int(cfg.train.num_workers) > 0,
        drop_last=bool(cfg.train.drop_last),
        collate_fn=collate_batch,
    )
    system = build_system(cfg, context.device)
    radiometric_profile = getattr(dataset, "thermal16_profile", None)
    style_profile_kwargs = {
        "storage_min": float(getattr(radiometric_profile, "storage_min", 0.0)),
        "storage_max": float(getattr(radiometric_profile, "storage_max", 65535.0)),
        "window_low": float(getattr(radiometric_profile, "window_low", 20000.0)),
        "window_high": float(getattr(radiometric_profile, "window_high", 40000.0)),
    }
    stats = configure_trainable(system.core, str(cfg.train.mode))
    detail_weights = DetailLossWeights(
        roi_x0=float(getattr(cfg.train, "detail_loss_roi_x0", 0.0)),
        edge=float(getattr(cfg.train, "detail_loss_edge", 0.0)),
        highpass=float(getattr(cfg.train, "detail_loss_highpass", 0.0)),
        statistics=float(getattr(cfg.train, "detail_loss_statistics", 0.0)),
        context=float(getattr(cfg.train, "detail_loss_context", 0.0)),
    )
    detail_max_timestep = int(getattr(cfg.train, "detail_loss_max_timestep", 300))
    detail_interval = int(getattr(cfg.train, "detail_loss_decode_interval", 4))
    detail_warmup_steps = int(getattr(cfg.train, "detail_loss_warmup_steps", 100))
    detail_max_ratio = float(getattr(cfg.train, "detail_loss_max_ratio", 0.15))
    if detail_weights.any_enabled and detail_interval < 1:
        raise ValueError("train.detail_loss_decode_interval must be positive when detail loss is enabled")
    if detail_weights.any_enabled and detail_max_timestep < 0:
        raise ValueError("train.detail_loss_max_timestep must be non-negative")
    if detail_warmup_steps < 0:
        raise ValueError("train.detail_loss_warmup_steps must be non-negative")
    if detail_max_ratio <= 0.0:
        raise ValueError("train.detail_loss_max_ratio must be positive")
    style_weights = {
        "gray": float(getattr(cfg.train, "ir_style_loss_gray", 0.0)),
        "mean": float(getattr(cfg.train, "ir_style_loss_mean", 0.0)),
        "std": float(getattr(cfg.train, "ir_style_loss_std", 0.0)),
        "gradient": float(getattr(cfg.train, "ir_style_loss_gradient", 0.0)),
        "pixel": float(getattr(cfg.train, "ir_style_loss_pixel", 0.0)),
        "quantile": float(getattr(cfg.train, "ir_style_loss_quantile", 0.0)),
        "cdf": float(getattr(cfg.train, "ir_style_loss_cdf", 0.0)),
        "roi_mean": float(getattr(cfg.train, "ir_style_loss_roi_mean", 0.0)),
        "roi_std": float(getattr(cfg.train, "ir_style_loss_roi_std", 0.0)),
        "roi_gradient": float(getattr(cfg.train, "ir_style_loss_roi_gradient", 0.0)),
        "fixed_window": float(getattr(cfg.train, "ir_style_loss_fixed_window", 0.0)),
    }
    if any(value < 0.0 for value in style_weights.values()):
        raise ValueError("IR style loss weights must be non-negative")
    style_enabled = any(value > 0.0 for value in style_weights.values())
    style_max_timestep = int(getattr(cfg.train, "ir_style_loss_max_timestep", 300))
    style_interval = int(getattr(cfg.train, "ir_style_loss_decode_interval", 4))
    style_warmup_steps = int(getattr(cfg.train, "ir_style_loss_warmup_steps", 500))
    style_max_ratio = float(getattr(cfg.train, "ir_style_loss_max_ratio", 0.10))
    if style_enabled and style_interval < 1:
        raise ValueError("train.ir_style_loss_decode_interval must be positive")
    if style_max_timestep < 0 or style_warmup_steps < 0 or style_max_ratio <= 0.0:
        raise ValueError("Invalid IR style loss schedule")
    teacher_weight = float(getattr(cfg.train, "teacher_distill_weight", 0.0))
    teacher_interval = int(getattr(cfg.train, "teacher_distill_interval", 2))
    teacher_warmup_steps = int(getattr(cfg.train, "teacher_distill_warmup_steps", 500))
    teacher_max_ratio = float(getattr(cfg.train, "teacher_distill_max_ratio", 0.20))
    if teacher_weight < 0.0 or teacher_interval < 1 or teacher_warmup_steps < 0 or teacher_max_ratio <= 0.0:
        raise ValueError("Invalid teacher distillation schedule")
    attention_box_loss_weight = float(getattr(cfg.train, "attention_box_loss_weight", 0.0))
    attention_box_loss_warmup_steps = int(
        getattr(cfg.train, "attention_box_loss_warmup_steps", 0)
    )
    attention_box_loss_max_ratio = float(
        getattr(cfg.train, "attention_box_loss_max_ratio", 0.05)
    )
    if attention_box_loss_weight < 0.0 or attention_box_loss_warmup_steps < 0:
        raise ValueError("Invalid attention box loss schedule")
    if attention_box_loss_max_ratio <= 0.0:
        raise ValueError("train.attention_box_loss_max_ratio must be positive")
    counterfactual_weight = float(getattr(cfg.train, "counterfactual_loss_weight", 0.0))
    counterfactual_warmup_steps = int(
        getattr(cfg.train, "counterfactual_loss_warmup_steps", 0)
    )
    counterfactual_max_ratio = float(
        getattr(cfg.train, "counterfactual_loss_max_ratio", 0.12)
    )
    counterfactual_candidates = int(
        getattr(cfg.train, "counterfactual_negative_candidates", 8)
    )
    if (
        counterfactual_weight < 0.0
        or counterfactual_warmup_steps < 0
        or counterfactual_max_ratio <= 0.0
        or counterfactual_candidates < 1
    ):
        raise ValueError("Invalid counterfactual grounding loss schedule")
    if counterfactual_weight > 0.0 and attention_box_loss_weight > 0.0:
        raise ValueError(
            "counterfactual grounding and attention box losses cannot record the same fuser pass"
        )
    instance_core_weight = float(getattr(cfg.train, "instance_core_loss_weight", 0.0))
    instance_core_warmup_steps = int(
        getattr(cfg.train, "instance_core_loss_warmup_steps", 50)
    )
    instance_core_max_ratio = float(
        getattr(cfg.train, "instance_core_loss_max_ratio", 0.20)
    )
    instance_core_feather_power = float(
        getattr(cfg.train, "instance_core_feather_power", 2.0)
    )
    if (
        instance_core_weight < 0.0
        or instance_core_warmup_steps < 0
        or instance_core_max_ratio <= 0.0
        or instance_core_feather_power <= 0.0
    ):
        raise ValueError("Invalid instance core loss schedule")
    attention_recorder = None
    if attention_box_loss_weight > 0.0:
        if bool(cfg.train.gradient_checkpointing):
            raise ValueError(
                "attention_box_loss requires train.gradient_checkpointing=false "
                "so recorded attention logits retain their gradient graph"
            )
        attention_recorder = install_grounding_attention_recorder(
            system.core.unet,
            min_resolution=int(getattr(cfg.train, "attention_box_min_resolution", 16)),
        )
    if bool(cfg.train.gradient_checkpointing):
        system.core.unet.enable_gradient_checkpointing()
    if bool(cfg.train.attention_slicing):
        system.core.unet.set_attention_slice("auto")
    if bool(cfg.train.xformers):
        try:
            system.core.unet.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            raise RuntimeError("xFormers was requested but could not be enabled") from exc
    resume_path = resolve_resume_path(output_dir, cfg.train.resume)
    stack_value = getattr(cfg.model, "stack_manifest", None)
    parent_stack: ModelStack | None = None
    stack_report = None
    resume_payload = load_checkpoint_file(resume_path) if resume_path is not None else None
    base_path = resolve_init_checkpoint(getattr(cfg.train, "base_checkpoint", None))
    init_path = resolve_init_checkpoint(getattr(cfg.train, "init_checkpoint", None))
    if stack_value not in (None, "", "null", "false", False):
        if base_path is not None or init_path is not None:
            raise ValueError(
                "model.stack_manifest cannot be combined with train.base_checkpoint or "
                "train.init_checkpoint"
            )
        parent_stack = load_model_stack(str(stack_value))
        verify_base_model(parent_stack, str(cfg.model.pretrained_model))
        stack_report = load_stack_into(system.core, parent_stack)
        if resume_path is not None:
            load_checkpoint_layer_into(
                system.core,
                resume_path,
                parent_stack=parent_stack,
            )
    else:
        if resume_path is not None and (base_path is not None or init_path is not None):
            raise ValueError(
                "A resumed run must not also set train.base_checkpoint or train.init_checkpoint"
            )
        base_payload = load_checkpoint_file(base_path) if base_path is not None else None
        init_payload = load_checkpoint_file(init_path) if init_path is not None else None
        if base_payload is not None:
            load_model_weights(system.core, base_payload)
        if init_payload is not None:
            load_model_weights(system.core, init_payload)
        if resume_payload is not None:
            load_model_weights(system.core, resume_payload)
    if context.is_main and native16_manifest is not None:
        native16_manifest["model_stack"] = parent_stack.metadata() if parent_stack is not None else None
        native16_manifest["project_checkpoint_loaded"] = parent_stack is not None
        (output_dir / "native16_run_manifest.json").write_text(
            json.dumps(native16_manifest, indent=2) + "\n", encoding="utf-8"
        )
    checkpoint_model_stack = (
        {
            "parent_fingerprint": parent_stack.fingerprint,
            "stack": parent_stack.metadata(),
        }
        if parent_stack is not None
        else None
    )
    teacher_core: GLIGENDenoisingCore | None = None
    if teacher_weight > 0.0:
        # Keep a frozen copy of the initialized IR teacher.  It is deliberately
        # created before FSDP wraps the student, so only the student participates
        # in optimizer/state-dict handling.
        teacher_core = copy.deepcopy(system.core)
        teacher_core.requires_grad_(False)
        teacher_core.eval()
        if context.device.type == "cuda":
            teacher_core.to(dtype=torch.float16)
    trainable_names = collect_trainable_parameter_names(system.core)
    optimizer_groups, optimizer_group_summary = build_optimizer_parameter_groups(system.core, cfg)
    trainable_parameters = [parameter for group in optimizer_groups for parameter in group["params"]]
    core = wrap_fsdp(system.core, cfg.distributed, context)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        weight_decay=float(cfg.train.weight_decay),
        eps=float(cfg.train.adam_epsilon),
    )
    accumulation = int(cfg.train.gradient_accumulation_steps)
    if len(loader) < 1:
        raise ValueError("DataLoader has zero batches. Reduce batch_size or disable drop_last")
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    configured_max_steps = int(cfg.train.max_steps)
    total_steps = configured_max_steps if configured_max_steps > 0 else int(cfg.train.epochs) * updates_per_epoch
    training_epochs = math.ceil(total_steps / updates_per_epoch) if configured_max_steps > 0 else int(cfg.train.epochs)
    warmup_steps = int(cfg.train.warmup_steps)
    scheduler_total_steps = int(getattr(cfg.train, "scheduler_total_steps", total_steps))
    if scheduler_total_steps < total_steps:
        raise ValueError("train.scheduler_total_steps must be at least train.max_steps")
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(scheduler_total_steps, 1),
    )
    precision_dtype = resolve_dtype(str(cfg.train.precision), context.device)
    scaler_enabled = context.device.type == "cuda" and precision_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload else 0
    start_epoch = int(resume_payload.get("epoch", 0)) if resume_payload else 0
    if resume_payload is not None:
        restore_optimizer_state(core, optimizer, resume_payload, context)
        if resume_payload.get("scheduler") is not None:
            lr_scheduler.load_state_dict(resume_payload["scheduler"])
        if scaler_enabled and resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
    writer = SummaryWriter(output_dir / "tensorboard") if context.is_main else None
    if context.is_main:
        print_parameter_summary(stats, str(cfg.model.type), str(cfg.train.mode))
        print(f"optimizer_groups={json.dumps(optimizer_group_summary, sort_keys=True)}")
        if expected_dataset_mix:
            print(f"expected_sampling_mix={json.dumps(expected_dataset_mix, sort_keys=True)}")
        if parent_stack is not None and stack_report is not None:
            print(
                "initialized_model_stack="
                f"{parent_stack.name} fingerprint={parent_stack.fingerprint} "
                f"layers={[(layer.identifier, layer.key_count) for layer in stack_report.layers]}"
            )
        if base_path is not None:
            print(f"initialized_base_model_from={base_path}")
        if init_path is not None:
            print(f"initialized_model_only_from={init_path}")
        if attention_recorder is not None:
            print(
                f"attention_box_loss_weight={attention_box_loss_weight} "
                f"warmup_steps={attention_box_loss_warmup_steps} "
                f"max_ratio={attention_box_loss_max_ratio} "
                f"recorded_fuser_layers={len(attention_recorder.layer_names)}"
            )
        if detail_weights.any_enabled:
            print(
                "detail_loss="
                f"{json.dumps(detail_weights.__dict__, sort_keys=True)} "
                f"max_timestep={detail_max_timestep} interval={detail_interval} "
                f"warmup_steps={detail_warmup_steps} max_ratio={detail_max_ratio}"
            )
        if style_enabled:
            print(
                "ir_style_loss="
                f"{json.dumps(style_weights, sort_keys=True)} "
                f"max_timestep={style_max_timestep} interval={style_interval} "
                f"warmup_steps={style_warmup_steps} max_ratio={style_max_ratio} "
                f"profile={json.dumps(style_profile_kwargs, sort_keys=True)}"
            )
        if teacher_core is not None:
            print(
                f"teacher_distill_weight={teacher_weight} interval={teacher_interval} "
                f"warmup_steps={teacher_warmup_steps} max_ratio={teacher_max_ratio}"
            )
        if counterfactual_weight > 0.0:
            print(
                f"counterfactual_loss_weight={counterfactual_weight} "
                f"warmup_steps={counterfactual_warmup_steps} "
                f"max_ratio={counterfactual_max_ratio} candidates={counterfactual_candidates}"
            )
        print(f"world_size={context.world_size} total_steps={total_steps} precision={cfg.train.precision}")
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, initial=global_step, disable=not context.is_main, dynamic_ncols=True)
    core.train()
    training_start = time.time()
    observed_datasets: Counter[str] = Counter()
    overflow_skips = 0
    micro_step = 0
    detail_active_micro_steps = 0
    detail_decode_count = 0
    style_active_micro_steps = 0
    style_decode_count = 0
    epoch = start_epoch
    while global_step < total_steps:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if global_step >= total_steps:
                break
            micro_step += 1
            batch = move_batch_tensors(batch, context.device)
            observed_datasets.update(str(name) for name in batch["dataset"])
            target_boxes = batch["boxes"].clone()
            target_box_mask = batch["box_mask"].clone()
            prompts = apply_condition_dropout(batch, cfg.condition, context.device)
            batch["boxes"] = perturb_condition_boxes(
                batch["boxes"], batch["box_mask"], cfg.condition
            )
            latents = system.encode_images(batch["pixel_values"])
            text = system.encode_text(prompts)
            phrase_embeddings = system.encode_phrases(batch["phrases"], batch["box_mask"])
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                int(system.noise_scheduler.config.num_train_timesteps),
                (latents.shape[0],),
                device=context.device,
                dtype=torch.long,
            )
            noisy_latents = system.noise_scheduler.add_noise(latents, noise, timesteps)
            prediction_type = str(system.noise_scheduler.config.prediction_type)
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = system.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")
            is_last_batch = batch_index + 1 == len(loader)
            should_step = (batch_index + 1) % accumulation == 0 or is_last_batch
            sync_context = contextlib.nullcontext()
            if isinstance(core, FullyShardedDataParallel) and not should_step:
                sync_context = core.no_sync()
            autocast_enabled = context.device.type == "cuda" and precision_dtype != torch.float32
            with sync_context:
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=precision_dtype,
                    enabled=autocast_enabled,
                ):
                    if attention_recorder is not None:
                        attention_recorder.clear(int(batch["boxes"].shape[1]))
                    prediction = core(
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        text_hidden_states=text.hidden_states,
                        boxes=batch["boxes"],
                        phrase_embeddings=phrase_embeddings,
                        box_mask=batch["box_mask"],
                    )
                    (
                        loss,
                        unweighted_loss,
                        mean_snr_weight,
                        bbox_latent_fraction,
                    ) = compute_bbox_weighted_diffusion_loss(
                        prediction,
                        target,
                        timesteps,
                        system.noise_scheduler,
                        prediction_type,
                        float(getattr(cfg.train, "min_snr_gamma", 0.0)),
                        boxes=target_boxes,
                        box_mask=target_box_mask,
                        box_loss_weight=float(getattr(cfg.train, "bbox_loss_weight", 1.0)),
                    )
                    diffusion_loss = loss
                    instance_core_loss = loss.new_zeros(())
                    instance_core_loss_ratio = loss.new_zeros(())
                    current_instance_core_weight = 0.0
                    if instance_core_weight > 0.0:
                        instance_core_loss = compute_instance_normalized_loss(
                            prediction=prediction,
                            target=target,
                            boxes=target_boxes,
                            box_mask=target_box_mask,
                            feather_power=instance_core_feather_power,
                        )
                        warmup_fraction = (
                            min(
                                1.0,
                                float(global_step + 1)
                                / float(instance_core_warmup_steps),
                            )
                            if instance_core_warmup_steps > 0
                            else 1.0
                        )
                        proposed_weight = instance_core_weight * warmup_fraction
                        ratio = float(
                            (proposed_weight * instance_core_loss.detach()).abs()
                            / diffusion_loss.detach().abs().clamp_min(1.0e-6)
                        )
                        if ratio > instance_core_max_ratio:
                            proposed_weight *= instance_core_max_ratio / ratio
                        current_instance_core_weight = proposed_weight
                        loss = loss + current_instance_core_weight * instance_core_loss
                        instance_core_loss_ratio = (
                            current_instance_core_weight
                            * instance_core_loss.detach().abs()
                            / diffusion_loss.detach().abs().clamp_min(1.0e-6)
                        )
                    detail_loss = loss.new_zeros(())
                    detail_roi_x0 = loss.new_zeros(())
                    detail_edge = loss.new_zeros(())
                    detail_highpass = loss.new_zeros(())
                    detail_statistics = loss.new_zeros(())
                    detail_context = loss.new_zeros(())
                    current_detail_weight = 0.0
                    detail_loss_ratio = loss.new_zeros(())
                    detail_active = False
                    if detail_weights.any_enabled and micro_step % detail_interval == 0:
                        selected = timesteps <= detail_max_timestep
                        if bool(selected.any().item()):
                            selected_indices = selected.nonzero(as_tuple=False).flatten()
                            x0_latents = prediction_to_x0(
                                noisy_latents[selected_indices],
                                prediction[selected_indices],
                                timesteps[selected_indices],
                                system.noise_scheduler,
                            )
                            decoded_x0 = system.decode_latents_with_grad(x0_latents)
                            pixel_target = select_pixel_supervision_target(
                                decoded_x0,
                                batch["pixel_values"][selected_indices].float(),
                                str(getattr(cfg.data, "input_mode", "rgb8")).lower(),
                            )
                            detail_stats = compute_detail_losses(
                                decoded=decoded_x0,
                                target=pixel_target,
                                boxes=target_boxes[selected_indices],
                                box_mask=target_box_mask[selected_indices],
                            )
                            detail_loss = weighted_detail_loss(detail_stats, detail_weights)
                            if detail_warmup_steps > 0:
                                warmup_fraction = min(
                                    1.0,
                                    float(global_step + 1) / float(detail_warmup_steps),
                                )
                            else:
                                warmup_fraction = 1.0
                            # Component coefficients are applied by weighted_detail_loss.
                            # The shared contribution helper only warms up and caps their sum.
                            detail_contribution = bounded_contribution(
                                "detail",
                                detail_loss,
                                diffusion_loss,
                                nominal_weight=1.0,
                                warmup_fraction=warmup_fraction,
                                max_ratio=detail_max_ratio,
                            )
                            current_detail_weight = detail_contribution.weight
                            loss = loss + detail_contribution.weighted
                            detail_loss_ratio = detail_contribution.ratio_to_diffusion
                            detail_roi_x0 = detail_stats.roi_x0.detach()
                            detail_edge = detail_stats.edge.detach()
                            detail_highpass = detail_stats.highpass.detach()
                            detail_statistics = detail_stats.statistics.detach()
                            detail_context = detail_stats.context.detach()
                            detail_active = True
                            detail_active_micro_steps += 1
                            detail_decode_count += int(selected_indices.numel())
                    style_loss = loss.new_zeros(())
                    style_gray = loss.new_zeros(())
                    style_mean = loss.new_zeros(())
                    style_std = loss.new_zeros(())
                    style_gradient = loss.new_zeros(())
                    style_roi_mean = loss.new_zeros(())
                    style_roi_std = loss.new_zeros(())
                    style_roi_gradient = loss.new_zeros(())
                    style_fixed_mean = loss.new_zeros(())
                    style_fixed_std = loss.new_zeros(())
                    style_fixed_gradient = loss.new_zeros(())
                    style_fixed_roi_mean = loss.new_zeros(())
                    style_fixed_roi_std = loss.new_zeros(())
                    style_fixed_roi_gradient = loss.new_zeros(())
                    current_style_weight = 0.0
                    style_loss_ratio = loss.new_zeros(())
                    style_active = False
                    if style_enabled and micro_step % style_interval == 0:
                        selected = timesteps <= style_max_timestep
                        if bool(selected.any().item()):
                            selected_indices = selected.nonzero(as_tuple=False).flatten()
                            x0_latents = prediction_to_x0(
                                noisy_latents[selected_indices],
                                prediction[selected_indices],
                                timesteps[selected_indices],
                                system.noise_scheduler,
                            )
                            decoded_x0 = system.decode_latents_with_grad(x0_latents)
                            full_radiometric_target = style_target(
                                batch["pixel_values"][selected_indices].float(),
                                str(getattr(cfg.data, "input_mode", "rgb8")).lower(),
                            )
                            style_stats = compute_ir_style_losses(
                                decoded=decoded_x0,
                                target=full_radiometric_target,
                                boxes=target_boxes[selected_indices],
                                box_mask=target_box_mask[selected_indices],
                                **style_profile_kwargs,
                            )
                            style_loss = weighted_ir_style_loss(style_stats, style_weights)
                            warmup_fraction = (
                                min(1.0, float(global_step + 1) / float(style_warmup_steps))
                                if style_warmup_steps > 0
                                else 1.0
                            )
                            # weighted_ir_style_loss already applies every named
                            # component coefficient. Applying max(style_weights)
                            # here used to square all configured coefficients.
                            style_contribution = bounded_contribution(
                                "ir_style",
                                style_loss,
                                diffusion_loss,
                                nominal_weight=1.0,
                                warmup_fraction=warmup_fraction,
                                max_ratio=style_max_ratio,
                            )
                            current_style_weight = style_contribution.weight
                            loss = loss + style_contribution.weighted
                            style_loss_ratio = style_contribution.ratio_to_diffusion
                            style_gray = style_stats.gray.detach()
                            style_mean = style_stats.mean.detach()
                            style_std = style_stats.std.detach()
                            style_gradient = style_stats.gradient.detach()
                            style_roi_mean = style_stats.roi_mean.detach()
                            style_roi_std = style_stats.roi_std.detach()
                            style_roi_gradient = style_stats.roi_gradient.detach()
                            style_fixed_mean = style_stats.fixed_mean.detach()
                            style_fixed_std = style_stats.fixed_std.detach()
                            style_fixed_gradient = style_stats.fixed_gradient.detach()
                            style_fixed_roi_mean = style_stats.fixed_roi_mean.detach()
                            style_fixed_roi_std = style_stats.fixed_roi_std.detach()
                            style_fixed_roi_gradient = style_stats.fixed_roi_gradient.detach()
                            style_active = True
                            style_active_micro_steps += 1
                            style_decode_count += int(selected_indices.numel())
                    teacher_loss = loss.new_zeros(())
                    teacher_loss_ratio = loss.new_zeros(())
                    current_teacher_weight = 0.0
                    if teacher_core is not None and micro_step % teacher_interval == 0:
                        with torch.no_grad():
                            teacher_prediction = teacher_core(
                                noisy_latents=noisy_latents,
                                timesteps=timesteps,
                                text_hidden_states=text.hidden_states,
                                boxes=batch["boxes"],
                                phrase_embeddings=phrase_embeddings,
                                box_mask=batch["box_mask"],
                            )
                        teacher_loss = F.mse_loss(
                            prediction.float(), teacher_prediction.float().detach()
                        )
                        warmup_fraction = (
                            min(1.0, float(global_step + 1) / float(teacher_warmup_steps))
                            if teacher_warmup_steps > 0
                            else 1.0
                        )
                        proposed_weight = teacher_weight * warmup_fraction
                        ratio = float(
                            (proposed_weight * teacher_loss.detach()).abs()
                            / diffusion_loss.detach().abs().clamp_min(1e-6)
                        )
                        if ratio > teacher_max_ratio:
                            proposed_weight *= teacher_max_ratio / ratio
                        current_teacher_weight = proposed_weight
                        loss = loss + current_teacher_weight * teacher_loss
                        teacher_loss_ratio = (
                            current_teacher_weight * teacher_loss.detach().abs()
                            / diffusion_loss.detach().abs().clamp_min(1e-6)
                        )
                    attention_loss = loss.new_zeros(())
                    attention_inside_mass = loss.new_zeros(())
                    attention_core_density = loss.new_zeros(())
                    attention_boundary_density = loss.new_zeros(())
                    attention_ring_density = loss.new_zeros(())
                    attention_loss_ratio = loss.new_zeros(())
                    current_attention_weight = 0.0
                    if attention_recorder is not None:
                        attention_loss, attention_stats = attention_recorder.compute_loss(
                            boxes=target_boxes,
                            box_mask=batch["box_mask"],
                            small_area_reference=float(
                                getattr(cfg.train, "attention_box_small_area_reference", 0.01)
                            ),
                            max_small_weight=float(
                                getattr(cfg.train, "attention_box_max_small_weight", 3.0)
                            ),
                            core_fraction=float(
                                getattr(cfg.train, "attention_box_core_fraction", 0.70)
                            ),
                            ring_scale=float(
                                getattr(cfg.train, "attention_box_ring_scale", 1.35)
                            ),
                            density_margin=float(
                                getattr(cfg.train, "attention_box_density_margin", 0.35)
                            ),
                            ring_weight=float(
                                getattr(cfg.train, "attention_box_ring_weight", 0.50)
                            ),
                            boundary_weight=float(
                                getattr(cfg.train, "attention_box_boundary_weight", 0.25)
                            ),
                        )
                        attention_inside_mass = attention_stats.inside_mass
                        attention_core_density = attention_stats.core_density
                        attention_boundary_density = attention_stats.boundary_density
                        attention_ring_density = attention_stats.ring_density
                        if attention_box_loss_warmup_steps > 0:
                            warmup_fraction = min(
                                1.0,
                                float(global_step + 1) / float(attention_box_loss_warmup_steps),
                            )
                        else:
                            warmup_fraction = 1.0
                        proposed_weight = attention_box_loss_weight * warmup_fraction
                        ratio = float(
                            (proposed_weight * attention_loss.detach()).abs()
                            / diffusion_loss.detach().abs().clamp_min(1e-6)
                        )
                        if ratio > attention_box_loss_max_ratio:
                            proposed_weight *= attention_box_loss_max_ratio / ratio
                        current_attention_weight = proposed_weight
                        loss = loss + current_attention_weight * attention_loss
                        attention_loss_ratio = (
                            current_attention_weight * attention_loss.detach().abs()
                            / diffusion_loss.detach().abs().clamp_min(1e-6)
                        )
                    counterfactual_loss = loss.new_zeros(())
                    counterfactual_loss_ratio = loss.new_zeros(())
                    counterfactual_true_advantage = loss.new_zeros(())
                    counterfactual_shifted_advantage = loss.new_zeros(())
                    counterfactual_outside_consistency = loss.new_zeros(())
                    counterfactual_negative_iou = loss.new_zeros(())
                    current_counterfactual_weight = 0.0
                    if counterfactual_weight > 0.0:
                        negative_boxes, negative_iou = sample_counterfactual_boxes(
                            batch["boxes"],
                            batch["box_mask"],
                            candidates=counterfactual_candidates,
                        )
                        negative_prediction = core(
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            text_hidden_states=text.hidden_states,
                            boxes=negative_boxes,
                            phrase_embeddings=phrase_embeddings,
                            box_mask=batch["box_mask"],
                        )
                        counterfactual_loss, counterfactual_stats = (
                            compute_counterfactual_grounding_loss(
                                prediction=prediction,
                                negative_prediction=negative_prediction,
                                target=target,
                                boxes=batch["boxes"],
                                negative_boxes=negative_boxes,
                                box_mask=batch["box_mask"],
                                negative_iou=negative_iou,
                                core_fraction=float(
                                    getattr(cfg.train, "counterfactual_core_fraction", 0.82)
                                ),
                                relative_margin=float(
                                    getattr(cfg.train, "counterfactual_relative_margin", 0.05)
                                ),
                                temperature=float(
                                    getattr(cfg.train, "counterfactual_temperature", 0.10)
                                ),
                                outside_weight=float(
                                    getattr(cfg.train, "counterfactual_outside_weight", 0.50)
                                ),
                            )
                        )
                        warmup_fraction = (
                            min(
                                1.0,
                                float(global_step + 1)
                                / float(counterfactual_warmup_steps),
                            )
                            if counterfactual_warmup_steps > 0
                            else 1.0
                        )
                        proposed_weight = counterfactual_weight * warmup_fraction
                        ratio = float(
                            (proposed_weight * counterfactual_loss.detach()).abs()
                            / diffusion_loss.detach().abs().clamp_min(1e-6)
                        )
                        if ratio > counterfactual_max_ratio:
                            proposed_weight *= counterfactual_max_ratio / ratio
                        current_counterfactual_weight = proposed_weight
                        loss = loss + current_counterfactual_weight * counterfactual_loss
                        counterfactual_loss_ratio = (
                            current_counterfactual_weight
                            * counterfactual_loss.detach().abs()
                            / diffusion_loss.detach().abs().clamp_min(1e-6)
                        )
                        counterfactual_true_advantage = (
                            counterfactual_stats.true_advantage
                        )
                        counterfactual_shifted_advantage = (
                            counterfactual_stats.shifted_advantage
                        )
                        counterfactual_outside_consistency = (
                            counterfactual_stats.outside_consistency
                        )
                        counterfactual_negative_iou = counterfactual_stats.negative_iou
                    scaled_loss = loss / float(accumulation)
                scaler.scale(scaled_loss).backward()
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            overflow = synchronize_grad_scaler_overflow(
                scaler, optimizer, context.device
            )
            if not overflow and float(cfg.train.max_grad_norm) > 0:
                if isinstance(core, FullyShardedDataParallel):
                    core.clip_grad_norm_(float(cfg.train.max_grad_norm))
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, float(cfg.train.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if overflow:
                overflow_skips += 1
                if context.is_main:
                    current_scale = scaler.get_scale()
                    print(
                        f"amp_overflow_skip={overflow_skips} "
                        f"global_step={global_step} next_scale={current_scale:.1f}"
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "train/amp_overflow_skips", overflow_skips, global_step
                        )
                        writer.add_scalar(
                            "train/amp_loss_scale", current_scale, global_step
                        )
                continue
            lr_scheduler.step()
            global_step += 1
            reduced_loss = reduce_mean(loss, context)
            reduced_diffusion_loss = reduce_mean(diffusion_loss, context)
            reduced_instance_core_loss = reduce_mean(instance_core_loss, context)
            reduced_instance_core_ratio = reduce_mean(instance_core_loss_ratio, context)
            reduced_detail_loss = reduce_mean(detail_loss, context)
            reduced_detail_roi_x0 = reduce_mean(detail_roi_x0, context)
            reduced_detail_edge = reduce_mean(detail_edge, context)
            reduced_detail_highpass = reduce_mean(detail_highpass, context)
            reduced_detail_statistics = reduce_mean(detail_statistics, context)
            reduced_detail_context = reduce_mean(detail_context, context)
            reduced_detail_ratio = reduce_mean(detail_loss_ratio, context)
            reduced_style_loss = reduce_mean(style_loss, context)
            reduced_style_gray = reduce_mean(style_gray, context)
            reduced_style_mean = reduce_mean(style_mean, context)
            reduced_style_std = reduce_mean(style_std, context)
            reduced_style_gradient = reduce_mean(style_gradient, context)
            reduced_style_roi_mean = reduce_mean(style_roi_mean, context)
            reduced_style_roi_std = reduce_mean(style_roi_std, context)
            reduced_style_roi_gradient = reduce_mean(style_roi_gradient, context)
            reduced_style_fixed_mean = reduce_mean(style_fixed_mean, context)
            reduced_style_fixed_std = reduce_mean(style_fixed_std, context)
            reduced_style_fixed_gradient = reduce_mean(style_fixed_gradient, context)
            reduced_style_fixed_roi_mean = reduce_mean(style_fixed_roi_mean, context)
            reduced_style_fixed_roi_std = reduce_mean(style_fixed_roi_std, context)
            reduced_style_fixed_roi_gradient = reduce_mean(style_fixed_roi_gradient, context)
            reduced_style_ratio = reduce_mean(style_loss_ratio, context)
            reduced_teacher_loss = reduce_mean(teacher_loss, context)
            reduced_teacher_ratio = reduce_mean(teacher_loss_ratio, context)
            reduced_attention_loss = reduce_mean(attention_loss, context)
            reduced_attention_inside_mass = reduce_mean(attention_inside_mass, context)
            reduced_attention_core_density = reduce_mean(attention_core_density, context)
            reduced_attention_boundary_density = reduce_mean(
                attention_boundary_density, context
            )
            reduced_attention_ring_density = reduce_mean(attention_ring_density, context)
            reduced_attention_ratio = reduce_mean(attention_loss_ratio, context)
            reduced_counterfactual_loss = reduce_mean(counterfactual_loss, context)
            reduced_counterfactual_ratio = reduce_mean(counterfactual_loss_ratio, context)
            reduced_counterfactual_true_advantage = reduce_mean(
                counterfactual_true_advantage, context
            )
            reduced_counterfactual_shifted_advantage = reduce_mean(
                counterfactual_shifted_advantage, context
            )
            reduced_counterfactual_outside_consistency = reduce_mean(
                counterfactual_outside_consistency, context
            )
            reduced_counterfactual_negative_iou = reduce_mean(
                counterfactual_negative_iou, context
            )
            reduced_unweighted_loss = reduce_mean(unweighted_loss, context)
            reduced_snr_weight = reduce_mean(mean_snr_weight, context)
            reduced_bbox_fraction = reduce_mean(bbox_latent_fraction, context)
            if context.is_main:
                progress.update(1)
                current_lrs = lr_scheduler.get_last_lr()
                progress.set_postfix(loss=f"{reduced_loss.item():.4f}", lr=f"{current_lrs[0]:.2e}")
                if writer is not None:
                    writer.add_scalar("train/loss", reduced_loss.item(), global_step)
                    writer.add_scalar("train/diffusion_loss", reduced_diffusion_loss.item(), global_step)
                    writer.add_scalar(
                        "train/instance_core_loss",
                        reduced_instance_core_loss.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/instance_core_weight",
                        current_instance_core_weight,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/instance_core_loss_ratio",
                        reduced_instance_core_ratio.item(),
                        global_step,
                    )
                    writer.add_scalar("train/detail_loss", reduced_detail_loss.item(), global_step)
                    writer.add_scalar("train/detail_roi_x0", reduced_detail_roi_x0.item(), global_step)
                    writer.add_scalar("train/detail_edge", reduced_detail_edge.item(), global_step)
                    writer.add_scalar("train/detail_highpass", reduced_detail_highpass.item(), global_step)
                    writer.add_scalar("train/detail_statistics", reduced_detail_statistics.item(), global_step)
                    writer.add_scalar("train/detail_context", reduced_detail_context.item(), global_step)
                    writer.add_scalar("train/detail_weight", current_detail_weight, global_step)
                    writer.add_scalar("train/detail_loss_ratio", reduced_detail_ratio.item(), global_step)
                    writer.add_scalar(
                        "train/detail_active_micro_fraction",
                        detail_active_micro_steps / float(max(micro_step, 1)),
                        global_step,
                    )
                    writer.add_scalar("train/detail_decode_count", detail_decode_count, global_step)
                    writer.add_scalar("train/ir_style_loss", reduced_style_loss.item(), global_step)
                    writer.add_scalar("train/ir_style_gray", reduced_style_gray.item(), global_step)
                    writer.add_scalar("train/ir_style_mean", reduced_style_mean.item(), global_step)
                    writer.add_scalar("train/ir_style_std", reduced_style_std.item(), global_step)
                    writer.add_scalar("train/ir_style_gradient", reduced_style_gradient.item(), global_step)
                    writer.add_scalar("train/ir_style_roi_mean", reduced_style_roi_mean.item(), global_step)
                    writer.add_scalar("train/ir_style_roi_std", reduced_style_roi_std.item(), global_step)
                    writer.add_scalar("train/ir_style_roi_gradient", reduced_style_roi_gradient.item(), global_step)
                    writer.add_scalar("train/ir_style_fixed_mean", reduced_style_fixed_mean.item(), global_step)
                    writer.add_scalar("train/ir_style_fixed_std", reduced_style_fixed_std.item(), global_step)
                    writer.add_scalar("train/ir_style_fixed_gradient", reduced_style_fixed_gradient.item(), global_step)
                    writer.add_scalar("train/ir_style_fixed_roi_mean", reduced_style_fixed_roi_mean.item(), global_step)
                    writer.add_scalar("train/ir_style_fixed_roi_std", reduced_style_fixed_roi_std.item(), global_step)
                    writer.add_scalar("train/ir_style_fixed_roi_gradient", reduced_style_fixed_roi_gradient.item(), global_step)
                    writer.add_scalar("train/ir_style_weight", current_style_weight, global_step)
                    writer.add_scalar("train/ir_style_loss_ratio", reduced_style_ratio.item(), global_step)
                    writer.add_scalar(
                        "train/ir_style_active_micro_fraction",
                        style_active_micro_steps / float(max(micro_step, 1)),
                        global_step,
                    )
                    writer.add_scalar("train/ir_style_decode_count", style_decode_count, global_step)
                    writer.add_scalar("train/teacher_distill_loss", reduced_teacher_loss.item(), global_step)
                    writer.add_scalar("train/teacher_distill_weight", current_teacher_weight, global_step)
                    writer.add_scalar("train/teacher_distill_loss_ratio", reduced_teacher_ratio.item(), global_step)
                    writer.add_scalar("train/attention_box_loss", reduced_attention_loss.item(), global_step)
                    writer.add_scalar(
                        "train/attention_box_inside_mass",
                        reduced_attention_inside_mass.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/attention_box_core_density",
                        reduced_attention_core_density.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/attention_box_boundary_density",
                        reduced_attention_boundary_density.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/attention_box_ring_density",
                        reduced_attention_ring_density.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/attention_box_loss_ratio",
                        reduced_attention_ratio.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/attention_box_weight", current_attention_weight, global_step
                    )
                    writer.add_scalar(
                        "train/counterfactual_loss",
                        reduced_counterfactual_loss.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/counterfactual_loss_ratio",
                        reduced_counterfactual_ratio.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/counterfactual_weight",
                        current_counterfactual_weight,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/counterfactual_true_advantage",
                        reduced_counterfactual_true_advantage.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/counterfactual_shifted_advantage",
                        reduced_counterfactual_shifted_advantage.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/counterfactual_outside_consistency",
                        reduced_counterfactual_outside_consistency.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/counterfactual_negative_iou",
                        reduced_counterfactual_negative_iou.item(),
                        global_step,
                    )
                    writer.add_scalar("train/loss_unweighted", reduced_unweighted_loss.item(), global_step)
                    writer.add_scalar("train/min_snr_mean_weight", reduced_snr_weight.item(), global_step)
                    writer.add_scalar("train/bbox_latent_fraction", reduced_bbox_fraction.item(), global_step)
                    for group, learning_rate in zip(optimizer.param_groups, current_lrs):
                        writer.add_scalar(f"train/lr_{group.get('name', 'group')}", learning_rate, global_step)
                    observed_total = max(sum(observed_datasets.values()), 1)
                    for dataset_name in sorted(observed_datasets):
                        writer.add_scalar(
                            f"data/observed_fraction_{dataset_name}",
                            observed_datasets[dataset_name] / observed_total,
                            global_step,
                        )
            if (
                global_step % int(cfg.output.save_every_steps) == 0
                and global_step < total_steps
            ):
                checkpoint_dir = output_dir / "checkpoints"
                step_path = checkpoint_dir / f"step_{global_step:08d}.pt"
                save_checkpoint(
                    path=step_path,
                    module=core,
                    optimizer=optimizer,
                    scaler=scaler if scaler_enabled else None,
                    scheduler=lr_scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    trainable_names=trainable_names,
                    class_to_idx=dataset.class_to_idx,
                    config_dict=dict(cfg),
                    context=context,
                    model_stack=checkpoint_model_stack,
                )
                if context.is_main:
                    update_latest_checkpoint(step_path, checkpoint_dir / "latest.pt")
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
        epoch += 1
    final_path = output_dir / "checkpoints" / "final.pt"
    save_checkpoint(
        path=final_path,
        module=core,
        optimizer=optimizer,
        scaler=scaler if scaler_enabled else None,
        scheduler=lr_scheduler,
        global_step=global_step,
        epoch=max(start_epoch, epoch if "epoch" in locals() else 0),
        trainable_names=trainable_names,
        class_to_idx=dataset.class_to_idx,
        config_dict=dict(cfg),
        context=context,
        model_stack=checkpoint_model_stack,
    )
    if context.is_main:
        update_latest_checkpoint(final_path, output_dir / "checkpoints" / "latest.pt")
        elapsed = time.time() - training_start
        print(f"finished step={global_step} elapsed_seconds={elapsed:.1f}")
        progress.close()
        if writer is not None:
            writer.close()
    cleanup_distributed(context)
    return None


if __name__ == "__main__":
    train()
