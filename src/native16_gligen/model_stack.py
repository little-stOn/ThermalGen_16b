"""Strict, ordered composition of delta checkpoints.

A model stack is the only supported way to compose training, inference, and
assessment models.  Each layer has a pinned digest and a key contract, so a
partially restored GLIGEN model fails before it can generate misleading data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import torch
from torch import nn
import yaml


class ModelStackError(RuntimeError):
    """Raised when a checkpoint stack is incomplete, ambiguous, or tampered with."""


_UNRESOLVED_ENV = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class KeyContract:
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    expected_key_count: int | None = None


@dataclass(frozen=True)
class LayerSpec:
    identifier: str
    uri: str
    sha256: str
    contract: KeyContract
    allow_overrides: tuple[str, ...] = ()
    parent_fingerprint: str | None = None
    legacy_audited: bool = False
    role: str | None = None
    resolved_path: Path = field(compare=False, repr=False, default=Path("."))


@dataclass(frozen=True)
class ModelStack:
    schema_version: int
    name: str
    manifest_path: Path
    base_model: Mapping[str, Any]
    layers: tuple[LayerSpec, ...]
    fingerprint: str

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "manifest": str(self.manifest_path),
            "fingerprint": self.fingerprint,
            "base_model": dict(self.base_model),
            "layers": [
                {
                    "id": layer.identifier,
                    "uri": layer.uri,
                    "sha256": layer.sha256,
                    "role": layer.role,
                    "legacy_audited": layer.legacy_audited,
                }
                for layer in self.layers
            ],
        }


@dataclass(frozen=True)
class LayerLoadReport:
    identifier: str
    path: str
    key_count: int
    loaded_keys: tuple[str, ...]
    overridden_keys: tuple[str, ...]


@dataclass(frozen=True)
class StackLoadReport:
    fingerprint: str
    layers: tuple[LayerLoadReport, ...]


def sha256_file(path: str | Path) -> str:
    """Return the digest of a regular file without materialising it in memory."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Artifact not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_uri(uri: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(uri))
    if _UNRESOLVED_ENV.search(expanded):
        raise ModelStackError(f"Unresolved environment variable in artifact URI: {uri}")
    return expanded


def _resolve_uri(uri: str, manifest_path: Path) -> Path:
    path = Path(_expand_uri(uri))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()



def verify_base_model(stack: ModelStack, source: str | Path) -> None:
    """Require the configured base model to equal the stack's pinned identity."""

    expected = stack.base_model.get("source")
    if not isinstance(expected, str) or not expected:
        raise ModelStackError("Model stack base_model.source must be a non-empty string")
    expected_path = _resolve_uri(expected, stack.manifest_path)
    actual_path = Path(source).expanduser().resolve()
    if actual_path != expected_path:
        raise ModelStackError(
            f"Configured base model {actual_path} does not match stack base {expected_path}"
        )

def _as_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ModelStackError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def _canonical_stack_payload(
    schema_version: int,
    name: str,
    base_model: Mapping[str, Any],
    layers: Iterable[LayerSpec],
) -> bytes:
    payload = {
        "schema_version": schema_version,
        "name": name,
        "base_model": dict(base_model),
        "layers": [
            {
                "id": layer.identifier,
                "sha256": layer.sha256,
                "role": layer.role,
                "contract": asdict(layer.contract),
                "allow_overrides": list(layer.allow_overrides),
                "parent_fingerprint": layer.parent_fingerprint,
                "legacy_audited": layer.legacy_audited,
            }
            for layer in layers
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint_for_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def stack_prefix_fingerprint(stack: ModelStack, layer_count: int) -> str:
    """Fingerprint a stack prefix, including the immutable base-model identity."""

    if not 0 <= layer_count <= len(stack.layers):
        raise ValueError(f"Invalid model stack prefix length: {layer_count}")
    return _fingerprint_for_payload(
        _canonical_stack_payload(
            stack.schema_version,
            stack.name,
            stack.base_model,
            stack.layers[:layer_count],
        )
    )


def load_model_stack(path: str | Path) -> ModelStack:
    """Load and fully validate a versioned model stack manifest."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model stack manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ModelStackError("Model stack manifest root must be a mapping")
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ModelStackError(f"Unsupported model stack schema_version: {schema_version!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ModelStackError("Model stack requires a non-empty name")
    base_model = raw.get("base_model")
    if not isinstance(base_model, dict) or not base_model:
        raise ModelStackError("Model stack requires a base_model mapping")
    raw_layers = raw.get("layers")
    if not isinstance(base_model.get("source"), str) or not base_model["source"]:
        raise ModelStackError("Model stack base_model.source must be a non-empty string")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ModelStackError("Model stack requires a non-empty layers list")

    layers: list[LayerSpec] = []
    identifiers: set[str] = set()
    for index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, dict):
            raise ModelStackError(f"layers[{index}] must be a mapping")
        identifier = raw_layer.get("id")
        uri = raw_layer.get("uri")
        checksum = raw_layer.get("sha256")
        if not isinstance(identifier, str) or not identifier:
            raise ModelStackError(f"layers[{index}].id must be a non-empty string")
        if identifier in identifiers:
            raise ModelStackError(f"Duplicate layer id: {identifier}")
        identifiers.add(identifier)
        if not isinstance(uri, str) or not uri:
            raise ModelStackError(f"layers[{index}].uri must be a non-empty string")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ModelStackError(f"layers[{index}].sha256 must be a lower-case SHA256 digest")
        raw_contract = raw_layer.get("key_contract", {})
        if not isinstance(raw_contract, dict):
            raise ModelStackError(f"layers[{index}].key_contract must be a mapping")
        expected_key_count = raw_contract.get("expected_key_count")
        if expected_key_count is not None and (
            not isinstance(expected_key_count, int) or expected_key_count < 1
        ):
            raise ModelStackError(f"layers[{index}].key_contract.expected_key_count must be positive")
        contract = KeyContract(
            required=_as_tuple(raw_contract.get("required"), f"layers[{index}].key_contract.required"),
            forbidden=_as_tuple(raw_contract.get("forbidden"), f"layers[{index}].key_contract.forbidden"),
            expected_key_count=expected_key_count,
        )
        layer = LayerSpec(
            identifier=identifier,
            uri=uri,
            sha256=checksum,
            contract=contract,
            allow_overrides=_as_tuple(raw_layer.get("allow_overrides"), f"layers[{index}].allow_overrides"),
            parent_fingerprint=raw_layer.get("parent_fingerprint"),
            legacy_audited=bool(raw_layer.get("legacy_audited", False)),
            role=raw_layer.get("role"),
            resolved_path=_resolve_uri(uri, manifest_path),
        )
        if layer.parent_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", layer.parent_fingerprint
        ):
            raise ModelStackError(f"layers[{index}].parent_fingerprint must be a SHA256 digest")
        if not layer.resolved_path.is_file():
            raise FileNotFoundError(f"Layer artifact not found: {layer.resolved_path}")
        actual_checksum = sha256_file(layer.resolved_path)
        if actual_checksum != layer.sha256:
            raise ModelStackError(
                f"SHA256 mismatch for {layer.identifier}: expected {layer.sha256}, got {actual_checksum}"
            )
        layers.append(layer)

    fingerprint = _fingerprint_for_payload(
        _canonical_stack_payload(schema_version, name, base_model, layers)
    )
    return ModelStack(
        schema_version=schema_version,
        name=name,
        manifest_path=manifest_path,
        base_model=base_model,
        layers=tuple(layers),
        fingerprint=fingerprint,
    )


def _load_payload(path: Path, *, model_only: bool = True) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", mmap=model_only)
    except (RuntimeError, TypeError):
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ModelStackError(f"Checkpoint is not a mapping: {path}")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ModelStackError(f"Checkpoint has no model state mapping: {path}")
    if not state:
        raise ModelStackError(f"Checkpoint model state is empty: {path}")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise ModelStackError(f"Checkpoint model state is not a string-to-tensor mapping: {path}")
    return payload


def _matching_keys(keys: Iterable[str], patterns: Iterable[str]) -> set[str]:
    return {key for key in keys for pattern in patterns if fnmatchcase(key, pattern)}


def _validate_key_contract(state: Mapping[str, torch.Tensor], contract: KeyContract, source: Path) -> None:
    keys = tuple(state)
    if contract.expected_key_count is not None and len(keys) != contract.expected_key_count:
        raise ModelStackError(
            f"Checkpoint {source} has {len(keys)} tensors; expected {contract.expected_key_count}"
        )
    for pattern in contract.required:
        if not any(fnmatchcase(key, pattern) for key in keys):
            raise ModelStackError(f"Checkpoint {source} does not satisfy required key pattern {pattern!r}")
    forbidden = _matching_keys(keys, contract.forbidden)
    if forbidden:
        rendered = ", ".join(sorted(forbidden)[:5])
        raise ModelStackError(f"Checkpoint {source} contains forbidden keys: {rendered}")


def _transform_state(
    state: Mapping[str, torch.Tensor],
    *,
    state_prefix: str,
    source: Path,
) -> dict[str, torch.Tensor]:
    if not state_prefix:
        return dict(state)
    transformed: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not name.startswith(state_prefix):
            raise ModelStackError(f"Checkpoint {source} key does not start with {state_prefix!r}: {name}")
        transformed[name.removeprefix(state_prefix)] = tensor
    return transformed


def _verify_parent(payload: Mapping[str, Any], expected_parent: str, source: Path, *, allow_legacy: bool) -> None:
    metadata = payload.get("model_stack")
    if metadata is None:
        if allow_legacy:
            return
        raise ModelStackError(
            f"Checkpoint {source} has no model_stack ancestry. Migrate it or list it as legacy_audited."
        )
    if not isinstance(metadata, Mapping):
        raise ModelStackError(f"Checkpoint {source} has invalid model_stack metadata")
    parent = metadata.get("parent_fingerprint")
    if parent != expected_parent:
        raise ModelStackError(
            f"Checkpoint {source} expects parent {parent!r}; resolved stack is {expected_parent}"
        )


def _load_state(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    source: Path,
    loaded_before: set[str],
    allow_overrides: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target_keys = set(module.state_dict())
    unexpected = sorted(set(state) - target_keys)
    if unexpected:
        raise ModelStackError(
            f"Checkpoint {source} has keys absent from target module: {', '.join(unexpected[:5])}"
        )
    overridden = sorted(set(state) & loaded_before)
    allowed = tuple(allow_overrides)
    invalid_overrides = [
        name for name in overridden if not any(fnmatchcase(name, pattern) for pattern in allowed)
    ]
    if invalid_overrides:
        raise ModelStackError(
            f"Checkpoint {source} overwrites undeclared layers: {', '.join(invalid_overrides[:5])}"
        )
    incompatible = module.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise ModelStackError(
            f"Checkpoint {source} produced unexpected keys: {', '.join(incompatible.unexpected_keys[:5])}"
        )
    return tuple(sorted(state)), tuple(overridden)


def load_stack_into(
    module: nn.Module,
    stack: ModelStack,
    *,
    state_prefix: str = "",
) -> StackLoadReport:
    """Load a pinned stack into a module and reject ambiguous parameter overrides."""

    loaded_before: set[str] = set()
    reports: list[LayerLoadReport] = []
    for index, layer in enumerate(stack.layers):
        running_fingerprint = stack_prefix_fingerprint(stack, index)
        payload = _load_payload(layer.resolved_path)
        if layer.parent_fingerprint is not None and layer.parent_fingerprint != running_fingerprint:
            raise ModelStackError(
                f"Layer {layer.identifier} declares parent {layer.parent_fingerprint}; "
                f"previous stack fingerprint is {running_fingerprint}"
            )
        _verify_parent(
            payload,
            running_fingerprint,
            layer.resolved_path,
            allow_legacy=layer.legacy_audited,
        )
        raw_state = payload["model"]
        _validate_key_contract(raw_state, layer.contract, layer.resolved_path)
        state = _transform_state(raw_state, state_prefix=state_prefix, source=layer.resolved_path)
        loaded, overridden = _load_state(
            module,
            state,
            source=layer.resolved_path,
            loaded_before=loaded_before,
            allow_overrides=tuple(
                pattern.removeprefix(state_prefix) for pattern in layer.allow_overrides
            ),
        )
        loaded_before.update(loaded)
        reports.append(
            LayerLoadReport(
                identifier=layer.identifier,
                path=str(layer.resolved_path),
                key_count=len(loaded),
                loaded_keys=loaded,
                overridden_keys=overridden,
            )
        )
    if stack_prefix_fingerprint(stack, len(stack.layers)) != stack.fingerprint:
        raise ModelStackError("Resolved model stack fingerprint is internally inconsistent")
    return StackLoadReport(fingerprint=stack.fingerprint, layers=tuple(reports))


def load_checkpoint_layer_into(
    module: nn.Module,
    checkpoint: str | Path,
    *,
    parent_stack: ModelStack,
    state_prefix: str = "",
) -> str:
    """Load one newly trained delta after a validated parent stack.

    New deltas must name the exact parent stack fingerprint written during
    training.  Legacy deltas cannot enter a stable inference/evaluation run.
    """

    source = Path(checkpoint).expanduser().resolve()
    payload = _load_payload(source)
    _verify_parent(payload, parent_stack.fingerprint, source, allow_legacy=False)
    state = _transform_state(payload["model"], state_prefix=state_prefix, source=source)
    _load_state(
        module,
        state,
        source=source,
        loaded_before=set(),
        allow_overrides=tuple(state),
    )
    return hashlib.sha256(
        (parent_stack.fingerprint + ":" + sha256_file(source)).encode("utf-8")
    ).hexdigest()
