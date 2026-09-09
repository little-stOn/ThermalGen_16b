"""Resolution-mixed video batch sampling.

The configuration format follows OpenDWM::

    {
        "256-448": [0.6, [[19, 2, 1]]],
        "176-304": [0.3, [[19, 4, 1]]],
    }

The resolution key is ``height-width``.  Each option is ``[T, B, weight]``;
``T`` is the temporal length and ``B`` is the per-rank batch size.  A sampler
item is encoded as ``dataset_index-T-height-width`` so ``DatasetAdapter`` can
apply the corresponding temporal crop and resize lazily.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, TypeVar

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for config-only use
    torch = None  # type: ignore[assignment]


_SamplerBase = torch.utils.data.Sampler if torch is not None else object
_IndexT = TypeVar("_IndexT")


@dataclass(frozen=True)
class _BucketOption:
    """One weighted ``(T, B)`` choice for a resolution."""

    height: int
    width: int
    frames: int
    batch_size: int
    weight: float

    @property
    def bucket_id(self) -> str:
        return f"{self.height}-{self.width}-{self.frames}-{self.batch_size}"


@dataclass(frozen=True)
class _ResolutionGroup:
    """All weighted temporal/batch choices for one resolution."""

    height: int
    width: int
    weight: float
    options: tuple[_BucketOption, ...]

    @property
    def key(self) -> str:
        return f"{self.height}-{self.width}"


class VariableVideoBatchSampler(_SamplerBase):
    """Sample variable-resolution video batches with optional DDP sharding.

    The implementation preserves OpenDWM's two-stage weighted choice and
    rank-local ``B`` semantics.  It fixes three failure modes in the reference
    implementation: bucket assignment is seeded, ``__len__`` is stable, and
    both bucket samples and bucket accesses are padded without creating short
    or empty batches.

    A global access order is built once per epoch.  At each global step every
    rank consumes one bucket access; ranks may therefore receive different
    resolutions and per-rank batch sizes, matching OpenDWM's official sampler.
    Every rank still receives exactly one complete batch per step.
    """

    def __init__(
        self,
        dataset: Any,
        bucket_config: Mapping[Any, Any],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ) -> None:
        if torch is None:
            raise ImportError("VariableVideoBatchSampler requires PyTorch")
        if not hasattr(dataset, "__len__"):
            raise TypeError("dataset must implement __len__")
        if len(dataset) < 1:
            raise ValueError("dataset must not be empty")
        if num_replicas is None or rank is None:
            num_replicas, rank = self._distributed_defaults(num_replicas, rank)
        if not isinstance(num_replicas, int) or isinstance(num_replicas, bool) or num_replicas < 1:
            raise ValueError("num_replicas must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(num_bucket_build_workers, int) or num_bucket_build_workers < 1:
            raise ValueError("num_bucket_build_workers must be a positive integer")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)
        self.verbose = bool(verbose)
        self.num_bucket_build_workers = num_bucket_build_workers
        self.epoch = 0
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch: int | None = None
        self._resume_micro_batch_access_index = 0
        self._plan_cache: tuple[tuple[tuple[str, ...], ...], ...] | None = None
        self._groups = self._parse_bucket_config(bucket_config)

    @staticmethod
    def _distributed_defaults(
        num_replicas: int | None,
        rank: int | None,
    ) -> tuple[int, int]:
        if (num_replicas is None) != (rank is None):
            raise ValueError("num_replicas and rank must be provided together")
        if num_replicas is not None and rank is not None:
            return num_replicas, rank
        if (
            torch is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    @staticmethod
    def _positive_integer(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer, got {value!r}")
        return int(value)

    @staticmethod
    def _finite_weight(value: Any, field: str, *, allow_zero: bool) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a finite number, got {value!r}")
        result = float(value)
        if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
            requirement = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{field} must be finite and {requirement}, got {value!r}")
        return result

    @classmethod
    def _parse_resolution(cls, value: Any) -> tuple[int, int]:
        if isinstance(value, str):
            parts = value.split("-")
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            parts = list(value)
        else:
            raise ValueError(f"resolution key must be 'height-width', got {value!r}")
        if len(parts) != 2:
            raise ValueError(f"resolution key must contain height and width, got {value!r}")
        normalized_parts: list[int] = []
        for part in parts:
            if isinstance(part, str) and part.isdigit():
                part = int(part)
            normalized_parts.append(cls._positive_integer(part, "resolution"))

        return normalized_parts[0], normalized_parts[1]

    @classmethod
    def _parse_bucket_config(
        cls,
        bucket_config: Mapping[Any, Any],
    ) -> tuple[_ResolutionGroup, ...]:
        if not isinstance(bucket_config, Mapping) or not bucket_config:
            raise ValueError("mix_config must be a non-empty mapping")
        groups: list[_ResolutionGroup] = []
        for resolution_key, raw_value in bucket_config.items():
            height, width = cls._parse_resolution(resolution_key)
            if (
                not isinstance(raw_value, Sequence)
                or isinstance(raw_value, (str, bytes))
                or len(raw_value) != 2
            ):
                raise ValueError(
                    f"mix_config[{resolution_key!r}] must be [resolution_weight, options]"
                )
            resolution_weight = cls._finite_weight(
                raw_value[0], f"mix_config[{resolution_key!r}].weight", allow_zero=True
            )
            raw_options = raw_value[1]
            if (
                not isinstance(raw_options, Sequence)
                or isinstance(raw_options, (str, bytes))
                or not raw_options
            ):
                raise ValueError(
                    f"mix_config[{resolution_key!r}] options must be non-empty"
                )
            options: list[_BucketOption] = []
            for option_index, raw_option in enumerate(raw_options):
                if (
                    not isinstance(raw_option, Sequence)
                    or isinstance(raw_option, (str, bytes))
                    or len(raw_option) != 3
                ):
                    raise ValueError(
                        f"mix_config[{resolution_key!r}].options[{option_index}] "
                        "must be [frames, batch_size, weight]"
                    )
                frames = cls._positive_integer(raw_option[0], "frames")
                batch_size = cls._positive_integer(raw_option[1], "batch_size")
                option_weight = cls._finite_weight(
                    raw_option[2],
                    f"mix_config[{resolution_key!r}].options[{option_index}].weight",
                    allow_zero=True,
                )
                options.append(_BucketOption(height, width, frames, batch_size, option_weight))
            if not any(option.weight > 0.0 for option in options):
                raise ValueError(f"mix_config[{resolution_key!r}] has no positive option weight")
            groups.append(_ResolutionGroup(height, width, resolution_weight, tuple(options)))
        if not any(group.weight > 0.0 for group in groups):
            raise ValueError("mix_config must have at least one positive resolution weight")
        return tuple(groups)

    def _generator(self) -> Any:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        return generator

    def group_by_bucket(self) -> OrderedDict[str, list[int]]:
        """Assign every dataset index to one weighted bucket deterministically."""

        generator = self._generator()
        resolution_weights = torch.tensor(
            [group.weight for group in self._groups], dtype=torch.double
        )
        resolution_choices = torch.multinomial(
            resolution_weights,
            len(self.dataset),
            replacement=True,
            generator=generator,
        ).tolist()
        positions_by_group: list[list[int]] = [[] for _ in self._groups]
        for index, group_index in enumerate(resolution_choices):
            positions_by_group[group_index].append(index)

        bucket_samples: OrderedDict[str, list[int]] = OrderedDict()
        for group_index, positions in enumerate(positions_by_group):
            if not positions:
                continue
            group = self._groups[group_index]
            option_weights = torch.tensor(
                [option.weight for option in group.options], dtype=torch.double
            )
            option_choices = torch.multinomial(
                option_weights,
                len(positions),
                replacement=True,
                generator=generator,
            ).tolist()
            for index, option_index in zip(positions, option_choices, strict=True):
                bucket = group.options[option_index]
                bucket_samples.setdefault(bucket.bucket_id, []).append(index)
        return bucket_samples

    @staticmethod
    def _cycle_pad(values: Sequence[_IndexT], count: int) -> list[_IndexT]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return []
        if not values:
            raise ValueError("cannot pad an empty bucket")
        repeats = (count + len(values) - 1) // len(values)
        return list((list(values) * repeats)[:count])

    @staticmethod
    def _shuffle(values: list[_IndexT], generator: Any) -> list[_IndexT]:
        if len(values) < 2:
            return values
        permutation = torch.randperm(len(values), generator=generator).tolist()
        return [values[index] for index in permutation]

    def _build_plan(self) -> tuple[tuple[tuple[str, ...], ...], ...]:
        generator = self._generator()
        bucket_samples = self.group_by_bucket()
        bucket_batch_sizes = {
            bucket_id: int(bucket_id.rsplit("-", 1)[-1])
            for bucket_id in bucket_samples
        }
        access_order: list[str] = []
        for bucket_id, samples in list(bucket_samples.items()):
            batch_size = bucket_batch_sizes[bucket_id]
            if self.drop_last:
                usable_count = len(samples) - len(samples) % batch_size
            else:
                usable_count = math.ceil(len(samples) / batch_size) * batch_size
            if usable_count == 0:
                del bucket_samples[bucket_id]
                continue
            prepared = list(samples)
            if len(prepared) < usable_count:
                prepared = self._cycle_pad(prepared, usable_count)
            if self.shuffle:
                prepared = self._shuffle(prepared, generator)
            bucket_samples[bucket_id] = prepared
            access_order.extend([bucket_id] * (usable_count // batch_size))

        if self.shuffle and len(access_order) > 1:
            access_order = self._shuffle(access_order, generator)
        remainder = len(access_order) % self.num_replicas
        if remainder:
            if self.drop_last:
                access_order = access_order[:-remainder]
            else:
                access_order.extend(access_order[: self.num_replicas - remainder])
        if not access_order:
            raise ValueError("mix_config produced zero batches; disable drop_last or add data")

        # Access padding can add another use of a bucket after its original
        # micro-batches were prepared.  Extend it before slicing to guarantee
        # complete batches on every rank.
        access_counts: dict[str, int] = {}
        for bucket_id in access_order:
            access_counts[bucket_id] = access_counts.get(bucket_id, 0) + 1
        for bucket_id, access_count in access_counts.items():
            required = access_count * bucket_batch_sizes[bucket_id]
            samples = bucket_samples[bucket_id]
            if len(samples) < required:
                bucket_samples[bucket_id] = samples + self._cycle_pad(
                    samples, required - len(samples)
                )

        consumed = {bucket_id: 0 for bucket_id in bucket_samples}
        rank_plans: list[list[tuple[str, ...]]] = [[] for _ in range(self.num_replicas)]
        for start in range(0, len(access_order), self.num_replicas):
            for rank in range(self.num_replicas):
                bucket_id = access_order[start + rank]
                batch_size = bucket_batch_sizes[bucket_id]
                begin = consumed[bucket_id]
                end = begin + batch_size
                consumed[bucket_id] = end
                height, width, frames, _ = bucket_id.split("-")
                rank_plans[rank].append(
                    tuple(
                        f"{index}-{frames}-{height}-{width}"
                        for index in bucket_samples[bucket_id][begin:end]
                    )
                )
        return tuple(tuple(plan) for plan in rank_plans)

    def _get_plan(self) -> tuple[tuple[tuple[str, ...], ...], ...]:
        if self._plan_cache is None:
            self._plan_cache = self._build_plan()
            self.approximate_num_batch = len(self._plan_cache[0])
        return self._plan_cache

    def __iter__(self) -> Iterator[list[str]]:
        plan = self._get_plan()[self.rank]
        start = self._resume_micro_batch_access_index // self.num_replicas
        if self._resume_micro_batch_access_index % self.num_replicas:
            raise ValueError("resume micro-batch index must be divisible by num_replicas")
        try:
            for batch_index in range(start, len(plan)):
                self.last_micro_batch_access_index = (batch_index + 1) * self.num_replicas
                yield list(plan[batch_index])
        finally:
            self.reset()

    def __len__(self) -> int:
        return len(self._get_plan()[self.rank])

    def get_num_batch(self) -> int:
        """Return the number of complete batches yielded per rank this epoch."""

        return len(self)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch
        self._plan_cache = None
        self.approximate_num_batch = None
        self.reset()

    def reset(self) -> None:
        self.last_micro_batch_access_index = 0
        self._resume_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict[str, int]:
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 0:
            raise ValueError("num_steps must be a non-negative integer")
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("sampler state must be a mapping")
        seed = state_dict.get("seed", self.seed)
        epoch = state_dict.get("epoch", self.epoch)
        access_index = state_dict.get("last_micro_batch_access_index", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("state seed must be a non-negative integer")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("state epoch must be a non-negative integer")
        if not isinstance(access_index, int) or isinstance(access_index, bool) or access_index < 0:
            raise ValueError("state last_micro_batch_access_index must be non-negative")
        if access_index % self.num_replicas:
            raise ValueError("state access index must be divisible by num_replicas")
        self.seed = seed
        self.epoch = epoch
        self._plan_cache = None
        self.approximate_num_batch = None
        self._resume_micro_batch_access_index = access_index
        self.last_micro_batch_access_index = access_index


__all__ = ["VariableVideoBatchSampler"]
