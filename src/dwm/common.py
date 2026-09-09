"""Configuration construction and fork-friendly shared state utilities.

This module mirrors the small, data-driven object factory used by OpenDWM. A
JSON/YAML mapping with ``_class_name`` describes a Python class or callable;
nested mappings and lists are instantiated recursively. Dataset-specific file
parsing stays in ``dwm.datasets`` while this module owns configuration and
resource construction.
"""

from __future__ import annotations

import bisect
import copy
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any, MutableMapping, Sequence

import numpy as np
import yaml


class PartialReadableRawIO(io.RawIOBase):
    """Bounded seekable view over a shared file object."""

    def __init__(
        self,
        base_io_object: io.RawIOBase,
        start: int,
        end: int,
        close_with_this_object: bool = False,
    ) -> None:
        super().__init__()
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range {start}:{end}")
        self.base_io_object = base_io_object
        self.start = int(start)
        self.end = int(end)
        self.position = self.start
        self.close_with_this_object = close_with_this_object
        self.base_io_object.seek(self.start)

    def close(self) -> None:
        if self.close_with_this_object:
            self.base_io_object.close()
        super().close()

    @property
    def closed(self) -> bool:
        return bool(self.base_io_object.closed) if self.close_with_this_object else False

    def readable(self) -> bool:
        return bool(self.base_io_object.readable())

    def seekable(self) -> bool:
        return bool(self.base_io_object.seekable())

    def writable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        remaining = self.end - self.position
        count = remaining if size is None or size < 0 else min(int(size), remaining)
        data = self.base_io_object.read(count)
        self.position += len(data)
        return data

    def readall(self) -> bytes:
        return self.read(-1)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = self.start + int(offset)
        elif whence == os.SEEK_CUR:
            target = self.position + int(offset)
        elif whence == os.SEEK_END:
            target = self.end + int(offset)
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        self.position = max(self.start, min(self.end, target))
        self.base_io_object.seek(self.position)
        return self.tell()

    def tell(self) -> int:
        return self.position - self.start


class ReadonlyDictIndices:
    """Read-only key-to-index lookup that is safe to share with workers."""

    def __init__(self, base_dict_keys: Sequence[Any]) -> None:
        sorted_table = sorted(enumerate(base_dict_keys), key=lambda item: item[1])
        self.sorted_keys = [item[1] for item in sorted_table]
        self.key_indices = np.asarray([item[0] for item in sorted_table], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sorted_keys)

    def __contains__(self, key: Any) -> bool:
        index = bisect.bisect_left(self.sorted_keys, key)
        return index < len(self.sorted_keys) and self.sorted_keys[index] == key

    def __getitem__(self, key: Any) -> int:
        index = bisect.bisect_left(self.sorted_keys, key)
        if index >= len(self.sorted_keys) or self.sorted_keys[index] != key:
            raise KeyError(f"{key!r} not found")
        return int(self.key_indices[index])

    def get_all_indices(self, key: Any) -> list[int]:
        start = bisect.bisect_left(self.sorted_keys, key)
        end = bisect.bisect_right(self.sorted_keys, key)
        return [int(value) for value in self.key_indices[start:end]]


class SerializedReadonlyList:
    """Pickle-backed list that avoids copy-on-write memory divergence."""

    def __init__(self, items: Sequence[Any]) -> None:
        serialized_items = [pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL) for item in items]
        self.offsets = np.cumsum([len(item) for item in serialized_items], dtype=np.int64)
        self.blob = b"".join(serialized_items)

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = 0 if index == 0 else int(self.offsets[index - 1])
        end = int(self.offsets[index])
        return pickle.loads(self.blob[start:end])


class SerializedReadonlyDict:
    """Pickle-backed dictionary with deterministic sorted-key lookup."""

    def __init__(self, base_dict: MutableMapping[Any, Any]) -> None:
        self.indices = ReadonlyDictIndices(list(base_dict.keys()))
        self.values = SerializedReadonlyList(list(base_dict.values()))

    def __len__(self) -> int:
        return len(self.indices)

    def __contains__(self, key: Any) -> bool:
        return key in self.indices

    def __getitem__(self, key: Any) -> Any:
        return self.values[self.indices[key]]

    def keys(self) -> list[Any]:
        return list(self.indices.sorted_keys)


class ConfigError(ValueError):
    """Raised for malformed or unresolved object configurations."""


_ENV_PATTERN = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def _parse_override_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def expand_environment_variables(value: Any) -> Any:
    """Expand environment variables recursively and reject unresolved paths."""

    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if _ENV_PATTERN.search(expanded):
            raise ConfigError(f"unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment_variables(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_environment_variables(item) for key, item in value.items()}
    return value


def apply_overrides(config: dict[str, Any], overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides without mutating the input mapping."""

    result = copy.deepcopy(config)
    for override in overrides or ():
        if "=" not in override:
            raise ConfigError(f"override must use key=value syntax: {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        parts = [part for part in dotted_key.split(".") if part]
        if not parts:
            raise ConfigError(f"override has an empty key: {override!r}")
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            current = cursor.get(part)
            if current is None:
                current = cursor[part] = {}
            if not isinstance(current, dict):
                raise ConfigError(f"override path crosses a non-mapping key: {dotted_key!r}")
            cursor = current
        cursor[parts[-1]] = _parse_override_value(raw_value)
    return expand_environment_variables(result)


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Load a JSON/YAML object config with environment expansion."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle) if config_path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    return apply_overrides(expand_environment_variables(raw), overrides)


def get_class(class_name: str) -> Any:
    """Resolve a dotted module/class or callable name."""

    if not isinstance(class_name, str) or not class_name.strip():
        raise ConfigError(f"class_name must be a non-empty string, got {class_name!r}")
    value = class_name.strip()
    if "." not in value:
        if value in globals():
            return globals()[value]
        raise ConfigError(f"cannot resolve unqualified class name {value!r}")
    module_name, attribute_name = value.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise ConfigError(f"module {module_name!r} has no attribute {attribute_name!r}") from error


def create_instance(class_name: str, **kwargs: Any) -> Any:
    """Construct a class or call a configured callable."""

    target = get_class(class_name)
    if not callable(target):
        raise ConfigError(f"configured target is not callable: {class_name!r}")
    return target(**kwargs)


def instantiate_config(config: Any, level: int = 0) -> Any:
    """Recursively instantiate nested values, retaining plain dictionaries."""

    if isinstance(config, dict):
        if "_class_name" in config:
            arguments = {
                key: instantiate_config(value, level + 1)
                for key, value in config.items()
                if key != "_class_name"
            }
            class_name = str(config["_class_name"])
            if class_name == "get_class":
                return get_class(**arguments)
            return create_instance(class_name, **arguments)
        return {
            key: instantiate_config(value, level + 1)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [instantiate_config(item, level + 1) for item in config]
    if isinstance(config, tuple):
        return tuple(instantiate_config(item, level + 1) for item in config)
    return config


def create_instance_from_config(config: Any, level: int = 0, **kwargs: Any) -> Any:
    """Instantiate a config tree; top-level kwargs override constructor args."""

    if isinstance(config, dict) and "_class_name" in config:
        arguments = {
            key: instantiate_config(value, level + 1)
            for key, value in config.items()
            if key != "_class_name"
        }
        if level == 0:
            arguments.update(kwargs)
        class_name = str(config["_class_name"])
        if class_name == "get_class":
            return get_class(**arguments)
        return create_instance(class_name, **arguments)
    return instantiate_config(config, level)


def initialize_global_state(config: MutableMapping[str, Any]) -> dict[str, Any]:
    """Instantiate and register the optional top-level ``global_state`` mapping."""

    state_config = config.get("global_state", {})
    if state_config is None:
        return global_state
    if not isinstance(state_config, dict):
        raise ConfigError("global_state must be a mapping")
    for key, value in state_config.items():
        global_state[str(key)] = create_instance_from_config(value)
    return global_state


def load_object_from_config(
    path: str | Path,
    key: str,
    overrides: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load a config file, initialize global state, and instantiate ``key``."""

    config = load_config(path, overrides)
    initialize_global_state(config)
    if key not in config:
        raise ConfigError(f"configuration key {key!r} is missing")
    return create_instance_from_config(config[key], **kwargs), config


def load_dataloader_from_config(
    path: str | Path,
    dataset_key: str = "dataset",
    dataloader_key: str = "dataloader",
    overrides: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build a configured dataset and inject it into a DataLoader config.

    When the top-level configuration contains ``mix_config``, the loader uses
    OpenDWM-compatible ``VariableVideoBatchSampler`` batches. The sampler is
    resolved after distributed initialization, so the same helper works for
    single-process inspection and DDP training.
    """

    config = load_config(path, overrides)
    initialize_global_state(config)
    if dataset_key not in config:
        raise ConfigError(f"configuration key {dataset_key!r} is missing")
    dataset = create_instance_from_config(config[dataset_key])
    loader_config = config.get(dataloader_key, {})
    if loader_config is None:
        loader_config = {}
    if not isinstance(loader_config, dict):
        raise ConfigError(f"{dataloader_key!r} must be a mapping")
    loader_config = copy.deepcopy(loader_config)
    loader_config.setdefault("_class_name", "torch.utils.data.DataLoader")
    loader_config.update(kwargs)
    loader_config["dataset"] = dataset

    mix_config = config.get("mix_config")
    if mix_config is not None:
        if loader_config.get("_class_name") != "torch.utils.data.DataLoader":
            raise ConfigError("mix_config requires torch.utils.data.DataLoader")
        if loader_config.get("batch_sampler") is not None:
            raise ConfigError("mix_config cannot be combined with dataloader.batch_sampler")
        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            else:
                num_replicas, rank = 1, 0
            from dwm.utils.sampler import VariableVideoBatchSampler

            training_sampler = VariableVideoBatchSampler(
                dataset,
                mix_config,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=bool(config.get("data_shuffle", loader_config.get("shuffle", True))),
                seed=int(config.get("generator_seed", 0)),
                drop_last=bool(config.get("drop_last", loader_config.get("drop_last", False))),
            )
        except (ImportError, TypeError, ValueError) as error:
            raise ConfigError(f"invalid mix_config: {error}") from error
        for key in ("batch_size", "shuffle", "sampler", "drop_last", "batch_sampler"):
            loader_config.pop(key, None)
        loader_config["batch_sampler"] = training_sampler

    return create_instance_from_config(loader_config), config


def load_pipeline_from_config(
    path: str | Path,
    pipeline_key: str = "pipeline",
    overrides: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate a pipeline and inject runtime model/optimizer dependencies."""

    config = load_config(path, overrides)
    initialize_global_state(config)
    if pipeline_key not in config:
        raise ConfigError(f"configuration key {pipeline_key!r} is missing")
    pipeline_config = config[pipeline_key]
    if not isinstance(pipeline_config, dict) or "_class_name" not in pipeline_config:
        raise ConfigError(f"{pipeline_key!r} must be an object configuration")
    return create_instance_from_config(pipeline_config, **kwargs), config


TASK_CONFIG_NAMES = {
    "bbox": "multi_bbox.yaml",
    "style": "multi_style.yaml",
    "joint": "multi_joint.yaml",
}


def load_task_dataloader(
    task: str,
    config_dir: str | Path = "configs/datasets",
    overrides: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load the multi-dataset DataLoader selected by training task."""

    task_name = str(task).lower()
    try:
        config_name = TASK_CONFIG_NAMES[task_name]
    except KeyError as error:
        raise ConfigError(
            f"task must be one of {sorted(TASK_CONFIG_NAMES)}, got {task!r}"
        ) from error
    config_dir_path = Path(config_dir).expanduser()
    if not config_dir_path.is_absolute() and not config_dir_path.is_dir():
        config_dir_path = Path(__file__).resolve().parents[2] / config_dir_path
    return load_dataloader_from_config(
        config_dir_path / config_name,
        overrides=overrides,
        **kwargs,
    )
def get_state(key: str) -> Any:
    try:
        return global_state[key]
    except KeyError as error:
        raise KeyError(f"global state key {key!r} is not initialized") from error


global_state: dict[str, Any] = {}

# 文件讲解：
# 1. 本文件是配置对象工厂，不负责解析任何具体数据集。load_config 读取
#    YAML/JSON，递归展开环境变量，并支持 dotted key=value 覆盖。
# 2. 配置字典只要包含 _class_name，就会由 instantiate_config 递归解析；
#    普通字典和列表保持容器结构，嵌套对象则先构造后注入父对象。
# 3. initialize_global_state 用于注册共享路径或资源。配置中的
#    dwm.common.get_state 可以在构造 dataset 时取回这些共享值，避免重复
#    传递长路径或重复打开同一个资源。
# 4. load_object_from_config 适合只验证 dataset；load_dataloader_from_config
#    先构造 dataset，再把同一个实例注入 DataLoader，避免 dataset 被构造两次。
# 5. 调试时优先运行 scripts/prepare/load_dataset_config.py；它会打印构造时间、
#    样本耗时、bbox 数量、字段形状，并可用 --batches 测量吞吐。
