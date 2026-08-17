from __future__ import annotations

from pathlib import Path

import pytest

from native16_gligen.config_schema import RecipeValidationError, resolve_stack_manifest, validate_active_recipe


def base_recipe() -> dict[str, object]:
    return {
        "model": {
            "stack_manifest": "stack.yaml",
            "pixel_channels": 3,
            "radiometric_bridge_checkpoint": "bridge.pt",
        },
        "data": {"input_mode": "native16_bridge", "sampling": {"layout_weights": {"small": 1.0}}},
        "train": {"base_checkpoint": None, "init_checkpoint": None},
    }


def test_stack_replaces_legacy_checkpoint_flags(tmp_path: Path) -> None:
    config = base_recipe()
    assert isinstance(config["train"], dict)
    config["train"]["init_checkpoint"] = "legacy.pt"  # type: ignore[index]

    with pytest.raises(RecipeValidationError, match="replaces"):
        validate_active_recipe(config)


def test_overlap_bucket_is_rejected() -> None:
    config = base_recipe()
    assert isinstance(config["data"], dict)
    config["data"]["sampling"] = {"layout_weights": {"overlap": 1.0}}  # type: ignore[index]

    with pytest.raises(RecipeValidationError, match="overlap"):
        validate_active_recipe(config)


def test_stack_path_resolves_relative_to_recipe(tmp_path: Path) -> None:
    stack = tmp_path / "stack.yaml"
    stack.write_text("schema_version: 1\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    config = base_recipe()

    resolve_stack_manifest(config, recipe)

    assert config["model"]["stack_manifest"] == str(stack)  # type: ignore[index]
