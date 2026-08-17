from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from native16_gligen.model_stack import (
    ModelStackError,
    load_checkpoint_layer_into,
    load_model_stack,
    load_stack_into,
    verify_base_model,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint(path: Path, state: dict[str, torch.Tensor], **extra: object) -> None:
    torch.save({"model": state, **extra}, path)


def write_stack(path: Path, layers: list[dict[str, object]]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "unit-stack",
                "base_model": {"id": "official-gligen-sd14", "source": "."},
                "layers": layers,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def layer(identifier: str, path: Path, required: str, **extra: object) -> dict[str, object]:
    return {
        "id": identifier,
        "uri": path.name,
        "sha256": digest(path),
        "legacy_audited": True,
        "key_contract": {"required": [required], "expected_key_count": 1},
        **extra,
    }


def test_stack_loads_ordered_non_overlapping_layers(tmp_path: Path) -> None:
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    write_checkpoint(left, {"left": torch.tensor([3.0])})
    write_checkpoint(right, {"right": torch.tensor([7.0])})
    manifest = tmp_path / "stack.yaml"
    write_stack(manifest, [layer("style", left, "left"), layer("grounding", right, "right")])

    module = nn.Module()
    module.register_parameter("left", nn.Parameter(torch.zeros(1)))
    module.register_parameter("right", nn.Parameter(torch.zeros(1)))
    stack = load_model_stack(manifest)
    report = load_stack_into(module, stack)

    assert [entry.identifier for entry in report.layers] == ["style", "grounding"]
    assert module.left.detach().tolist() == [3.0]
    assert module.right.detach().tolist() == [7.0]


def test_stack_rejects_undeclared_parameter_override(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    write_checkpoint(first, {"left": torch.tensor([1.0])})
    write_checkpoint(second, {"left": torch.tensor([2.0])})
    manifest = tmp_path / "stack.yaml"
    write_stack(manifest, [layer("first", first, "left"), layer("second", second, "left")])
    module = nn.Module()
    module.register_parameter("left", nn.Parameter(torch.zeros(1)))

    with pytest.raises(ModelStackError, match="overwrites undeclared"):
        load_stack_into(module, load_model_stack(manifest))


def test_new_delta_requires_exact_parent_fingerprint(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    write_checkpoint(parent, {"left": torch.tensor([1.0])})
    manifest = tmp_path / "stack.yaml"
    write_stack(manifest, [layer("parent", parent, "left")])
    stack = load_model_stack(manifest)
    module = nn.Module()
    module.register_parameter("left", nn.Parameter(torch.zeros(1)))
    module.register_parameter("right", nn.Parameter(torch.zeros(1)))
    load_stack_into(module, stack)

    child = tmp_path / "child.pt"
    write_checkpoint(
        child,
        {"right": torch.tensor([9.0])},
        model_stack={"parent_fingerprint": stack.fingerprint},
    )
    fingerprint = load_checkpoint_layer_into(module, child, parent_stack=stack)
    assert len(fingerprint) == 64
    assert module.right.detach().tolist() == [9.0]

    invalid = tmp_path / "invalid.pt"
    write_checkpoint(
        invalid,
        {"right": torch.tensor([4.0])},
        model_stack={"parent_fingerprint": "0" * 64},
    )
    with pytest.raises(ModelStackError, match="expects parent"):
        load_checkpoint_layer_into(module, invalid, parent_stack=stack)


def test_stack_rejects_base_model_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    write_checkpoint(checkpoint, {"left": torch.tensor([1.0])})
    manifest = tmp_path / "stack.yaml"
    write_stack(manifest, [layer("parent", checkpoint, "left")])
    stack = load_model_stack(manifest)

    verify_base_model(stack, tmp_path)
    with pytest.raises(ModelStackError, match="does not match"):
        verify_base_model(stack, tmp_path / "other-base")
