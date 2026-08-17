"""Validation shared by versioned, runnable recipe configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping


class RecipeValidationError(ValueError):
    """Raised when an active recipe contains incompatible runtime contracts."""


def resolve_stack_manifest(config: MutableMapping[str, Any], config_path: Path) -> None:
    model = config.get("model")
    if not isinstance(model, MutableMapping):
        return
    raw = model.get("stack_manifest")
    if raw in (None, "", False):
        return
    stack_path = Path(str(raw)).expanduser()
    if not stack_path.is_absolute():
        stack_path = (config_path.parent / stack_path).resolve()
    if not stack_path.is_file():
        raise RecipeValidationError(f"model.stack_manifest does not exist: {stack_path}")
    model["stack_manifest"] = str(stack_path)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise RecipeValidationError(f"{name} must be a mapping")
    return value


def validate_active_recipe(config: Mapping[str, Any]) -> None:
    """Reject the configuration combinations that previously created bad runs."""

    model = _section(config, "model")
    data = _section(config, "data")
    train = _section(config, "train")
    stack = model.get("stack_manifest")
    legacy_layers = [
        name
        for name in ("base_checkpoint", "init_checkpoint")
        if train.get(name) not in (None, "", "none", "null", False)
    ]
    if stack not in (None, "", False) and legacy_layers:
        joined = ", ".join(legacy_layers)
        raise RecipeValidationError(
            "model.stack_manifest replaces train.base_checkpoint/train.init_checkpoint; "
            f"remove {joined}"
        )
    if str(data.get("input_mode", "rgb8")).lower() == "native16_bridge":
        if int(model.get("pixel_channels", 3)) != 3:
            raise RecipeValidationError("native16_bridge requires model.pixel_channels=3")
        if not model.get("radiometric_bridge_checkpoint"):
            raise RecipeValidationError("native16_bridge requires a radiometric bridge checkpoint")
    sampling = data.get("sampling")
    if isinstance(sampling, Mapping):
        layout_weights = sampling.get("layout_weights")
        if isinstance(layout_weights, Mapping) and "overlap" in layout_weights:
            raise RecipeValidationError(
                "data.sampling.layout_weights.overlap is unsupported; use default, small, and crowded"
            )
