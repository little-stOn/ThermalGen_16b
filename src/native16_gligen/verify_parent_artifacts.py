"""Verify immutable Native16 parent artifacts before training or inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from native16_gligen.config import load_config
from native16_gligen.model_stack import (
    ModelStackError,
    load_model_stack,
    sha256_file,
    verify_base_model,
)
from native16_gligen.native16_vae import assert_clean_official_parent


def _require_file(value: object, name: str) -> Path:
    if value in (None, "", "null", "false", False):
        raise ModelStackError(f"Missing required {name} in the active recipe")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True, help="Native16 recipe YAML")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Configuration override key=value",
    )
    args = parser.parse_args()

    cfg = load_config(args.cfg, args.set)
    stack = load_model_stack(cfg.model.stack_manifest)
    verify_base_model(stack, cfg.model.pretrained_model)
    assert_clean_official_parent(cfg.model.pretrained_model)

    for layer in stack.layers:
        actual = sha256_file(layer.resolved_path)
        if actual != layer.sha256:
            raise ModelStackError(
                f"SHA-256 mismatch for {layer.identifier}: expected {layer.sha256}, got {actual}"
            )
        print(f"verified_layer={layer.identifier} sha256={actual}")

    for name in ("radiometric_bridge_checkpoint", "style_adapter_checkpoint"):
        required_path = _require_file(getattr(cfg.model, name, None), name)
        print(f"present_{name}={required_path}")
    print(
        f"verified_parent_stack={stack.name} fingerprint={stack.fingerprint}"
    )


if __name__ == "__main__":
    main()
