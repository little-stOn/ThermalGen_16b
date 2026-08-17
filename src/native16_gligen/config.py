from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import yaml

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from native16_gligen.config_schema import RecipeValidationError, resolve_stack_manifest, validate_active_recipe


class ConfigNode(dict):
    """Dictionary with recursive attribute access."""

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        message = f"Configuration key '{key}' does not exist"
        raise AttributeError(message)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
        return None

    def clone(self) -> "ConfigNode":
        raw = copy.deepcopy(dict(self))
        node = to_config_node(raw)
        if not isinstance(node, ConfigNode):
            raise TypeError("Cloned configuration is not a mapping")
        return node

    def validate(self) -> None:
        required = ["model", "data", "condition", "train", "distributed", "output"]
        missing = [key for key in required if key not in self]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing top-level configuration keys {joined}")
        if self.model.type != "gligen_sd14":
            raise ValueError("model.type must be gligen_sd14")
        if self.train.mode not in {
            "grounding_only",
            "grounding_adapter",
            "grounding_crossattn",
            "instance_fusion",
            "full_unet",
            "style_unet",
        }:
            raise ValueError(
                "train.mode must be grounding_only, grounding_adapter, "
                "grounding_crossattn, instance_fusion, full_unet, or style_unet"
            )
        if self.train.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("train.precision must be fp32, fp16, or bf16")
        if int(self.data.max_boxes) < 1:
            raise ValueError("data.max_boxes must be positive")
        input_mode = str(getattr(self.data, "input_mode", "rgb8")).lower()
        if input_mode not in {"rgb8", "thermal16", "native16", "native16_bridge"}:
            raise ValueError("data.input_mode must be rgb8, thermal16, native16, or native16_bridge")
        if input_mode in {"thermal16", "native16", "native16_bridge"}:
            thermal_cfg = getattr(self.data, "thermal16", None)
            if thermal_cfg is None:
                raise ValueError("thermal16 input mode requires data.thermal16")
            field = str(getattr(thermal_cfg, "field", "thermal16"))
            if not field:
                raise ValueError("data.thermal16.field must be non-empty")
        if input_mode == "native16":
            if int(getattr(self.model, "pixel_channels", 1)) != 1:
                raise ValueError("native16 mode requires model.pixel_channels=1")
            vae_path = str(getattr(self.model, "vae_pretrained_model", ""))
            if not vae_path:
                raise ValueError("native16 mode requires an independent model.vae_pretrained_model")
        if input_mode == "native16_bridge":
            if int(getattr(self.model, "pixel_channels", 3)) != 3:
                raise ValueError("native16_bridge requires model.pixel_channels=3")
            bridge_path = str(getattr(self.model, "radiometric_bridge_checkpoint", ""))
            if not bridge_path:
                raise ValueError("native16_bridge requires model.radiometric_bridge_checkpoint")
        if int(self.model.height) % 8 or int(self.model.width) % 8:
            raise ValueError("model.height and model.width must be divisible by 8")
        if int(self.train.batch_size) < 1 or int(self.train.gradient_accumulation_steps) < 1:
            raise ValueError("train.batch_size and gradient_accumulation_steps must be positive")
        if int(self.train.epochs) < 1:
            raise ValueError("train.epochs must be positive")
        scheduler_total_steps = int(getattr(self.train, "scheduler_total_steps", 0))
        if scheduler_total_steps < 0:
            raise ValueError("train.scheduler_total_steps must be non-negative")
        if scheduler_total_steps > 0 and int(self.train.max_steps) > 0 and scheduler_total_steps < int(self.train.max_steps):
            raise ValueError("train.scheduler_total_steps must be at least train.max_steps")
        if int(self.output.save_every_steps) < 1:
            raise ValueError("output.save_every_steps must be positive")
        min_snr_gamma = float(getattr(self.train, "min_snr_gamma", 0.0))
        if min_snr_gamma < 0:
            raise ValueError("train.min_snr_gamma must be non-negative")
        bbox_loss_weight = float(getattr(self.train, "bbox_loss_weight", 1.0))
        if bbox_loss_weight < 1.0:
            raise ValueError("train.bbox_loss_weight must be at least 1")
        detail_names = (
            "detail_loss_roi_x0",
            "detail_loss_edge",
            "detail_loss_highpass",
            "detail_loss_statistics",
            "detail_loss_context",
        )
        if any(float(getattr(self.train, name, 0.0)) < 0.0 for name in detail_names):
            raise ValueError("detail loss weights must be non-negative")
        detail_enabled = any(float(getattr(self.train, name, 0.0)) > 0.0 for name in detail_names)
        detail_interval = int(getattr(self.train, "detail_loss_decode_interval", 4))
        detail_timestep = int(getattr(self.train, "detail_loss_max_timestep", 300))
        detail_warmup = int(getattr(self.train, "detail_loss_warmup_steps", 100))
        detail_ratio = float(getattr(self.train, "detail_loss_max_ratio", 0.15))
        if detail_enabled and detail_interval < 1:
            raise ValueError("train.detail_loss_decode_interval must be positive")
        if detail_timestep < 0:
            raise ValueError("train.detail_loss_max_timestep must be non-negative")
        if detail_warmup < 0:
            raise ValueError("train.detail_loss_warmup_steps must be non-negative")
        if detail_ratio <= 0.0:
            raise ValueError("train.detail_loss_max_ratio must be positive")
        ir_style_names = (
            "ir_style_loss_gray",
            "ir_style_loss_mean",
            "ir_style_loss_std",
            "ir_style_loss_gradient",
            "ir_style_loss_pixel",
            "ir_style_loss_quantile",
            "ir_style_loss_cdf",
            "ir_style_loss_roi_mean",
            "ir_style_loss_roi_std",
            "ir_style_loss_roi_gradient",
            "ir_style_loss_fixed_window",
        )
        if any(float(getattr(self.train, name, 0.0)) < 0.0 for name in ir_style_names):
            raise ValueError("IR style loss weights must be non-negative")
        style_enabled = any(float(getattr(self.train, name, 0.0)) > 0.0 for name in ir_style_names)
        style_interval = int(getattr(self.train, "ir_style_loss_decode_interval", 4))
        style_timestep = int(getattr(self.train, "ir_style_loss_max_timestep", 300))
        style_warmup = int(getattr(self.train, "ir_style_loss_warmup_steps", 500))
        style_ratio = float(getattr(self.train, "ir_style_loss_max_ratio", 0.10))
        if style_enabled and style_interval < 1:
            raise ValueError("train.ir_style_loss_decode_interval must be positive")
        if style_timestep < 0 or style_warmup < 0 or style_ratio <= 0.0:
            raise ValueError("Invalid IR style loss schedule")
        teacher_weight = float(getattr(self.train, "teacher_distill_weight", 0.0))
        teacher_interval = int(getattr(self.train, "teacher_distill_interval", 2))
        teacher_warmup = int(getattr(self.train, "teacher_distill_warmup_steps", 500))
        teacher_ratio = float(getattr(self.train, "teacher_distill_max_ratio", 0.20))
        if teacher_weight < 0.0 or teacher_interval < 1 or teacher_warmup < 0 or teacher_ratio <= 0.0:
            raise ValueError("Invalid teacher distillation schedule")
        attention_weight = float(getattr(self.train, "attention_box_loss_weight", 0.0))
        attention_warmup = int(getattr(self.train, "attention_box_loss_warmup_steps", 0))
        attention_ratio = float(getattr(self.train, "attention_box_loss_max_ratio", 0.05))
        core_fraction = float(getattr(self.train, "attention_box_core_fraction", 0.70))
        ring_scale = float(getattr(self.train, "attention_box_ring_scale", 1.35))
        ring_weight = float(getattr(self.train, "attention_box_ring_weight", 0.50))
        boundary_weight = float(getattr(self.train, "attention_box_boundary_weight", 0.25))
        if attention_weight < 0.0 or attention_warmup < 0 or attention_ratio <= 0.0:
            raise ValueError("Invalid attention box loss schedule")
        if not 0.0 < core_fraction <= 1.0 or ring_scale < 1.0:
            raise ValueError("Invalid attention box geometry")
        if ring_weight < 0.0 or boundary_weight < 0.0:
            raise ValueError("Attention box density weights must be non-negative")
        learning_rate_groups = getattr(self.train, "learning_rate_groups", None)
        if learning_rate_groups is not None:
            required_groups = {"base_unet", "cross_attention", "grounding"}
            missing_groups = required_groups - set(learning_rate_groups)
            if missing_groups:
                raise ValueError(f"train.learning_rate_groups is missing {sorted(missing_groups)}")
            if any(float(learning_rate_groups[name]) <= 0 for name in required_groups):
                raise ValueError("All grouped learning rates must be positive")
        sampling = getattr(self.data, "sampling", None)
        if sampling is not None and str(getattr(sampling, "strategy", "uniform")) == "weighted":
            weights = getattr(sampling, "dataset_weights", None)
            if not isinstance(weights, dict) or not weights:
                raise ValueError("Weighted sampling requires data.sampling.dataset_weights")
            if any(float(value) <= 0 for value in weights.values()):
                raise ValueError("Every data.sampling.dataset_weights value must be positive")
            if int(getattr(sampling, "samples_per_epoch", 0)) < 1:
                raise ValueError("Weighted sampling requires a positive samples_per_epoch")
        visible = float(getattr(self.data, "min_visible_fraction", 0.3))
        focus = float(getattr(self.data, "bbox_focus_probability", 0.7))
        if not 0.0 <= visible <= 1.0:
            raise ValueError("data.min_visible_fraction must be in [0, 1]")
        if not 0.0 <= focus <= 1.0:
            raise ValueError("data.bbox_focus_probability must be in [0, 1]")
        jitter_probability = float(getattr(self.condition, "box_jitter_probability", 0.0))
        jitter_center = float(getattr(self.condition, "box_jitter_center_std", 0.0))
        jitter_scale = float(getattr(self.condition, "box_jitter_scale_std", 0.0))
        if not 0.0 <= jitter_probability <= 1.0:
            raise ValueError("condition.box_jitter_probability must be in [0, 1]")
        if jitter_center < 0.0 or jitter_scale < 0.0:
            raise ValueError("Condition box jitter magnitudes must be non-negative")
        try:
            validate_active_recipe(self)
        except RecipeValidationError as exc:
            raise ValueError(str(exc)) from exc


def to_config_node(value: Any) -> Any:
    if isinstance(value, dict):
        converted = ConfigNode()
        for key, item in value.items():
            converted[key] = to_config_node(item)
        return converted
    if isinstance(value, list):
        return [to_config_node(item) for item in value]
    return value


_UNRESOLVED_ENV = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def expand_environment_variables(value: Any) -> Any:
    """Expand recipe paths deterministically and reject missing variables."""

    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if _UNRESOLVED_ENV.search(expanded):
            raise ValueError(f"Unresolved environment variable in configuration value: {value}")
        return expanded
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment_variables(item) for key, item in value.items()}
    return value


def parse_override_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: ConfigNode, overrides: list[str]) -> ConfigNode:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax, got {override}")
        dotted_key, raw_value = override.split("=", 1)
        value = parse_override_value(raw_value)
        path = dotted_key.split(".")
        cursor: ConfigNode = cfg
        for part in path[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = ConfigNode()
            cursor = cursor[part]
        cursor[path[-1]] = to_config_node(value)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> ConfigNode:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    cfg = to_config_node(expand_environment_variables(raw))
    cfg = apply_overrides(cfg, overrides or [])
    resolve_stack_manifest(cfg, config_path)
    cfg.validate()
    return cfg


def save_config(cfg: ConfigNode, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plain = json.loads(json.dumps(cfg))
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(plain, handle, sort_keys=False, allow_unicode=True)
    if not output_path.is_file():
        raise RuntimeError(f"Failed to save configuration to {output_path}")
    return None


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", required=True, help="Path to a YAML configuration")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Configuration overrides such as train.batch_size=2 model.height=512",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    return parser
