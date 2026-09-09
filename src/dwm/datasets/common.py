"""Shared local dataset adapters and bbox utilities.

Dataset-specific modules parse storage into frame records. ``DatasetAdapter``
applies configured transforms; composition and collation stay dataset-agnostic.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for config parsing
    torch = None  # type: ignore[assignment]


_DatasetBase = torch.utils.data.Dataset if torch is not None else object


class Copy:
    """Identity transform useful in declarative transform lists."""

    def __call__(self, value: Any) -> Any:
        return value


class FilterPoints:
    """Keep points whose Euclidean distance is in ``[min, max)``."""

    def __init__(self, min_distance: float = 0.0, max_distance: float = 1000.0) -> None:
        if min_distance < 0 or max_distance <= min_distance:
            raise ValueError("require 0 <= min_distance < max_distance")
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)

    def __call__(self, points: Any) -> Any:
        if torch is None or not torch.is_tensor(points):
            raise TypeError("FilterPoints expects a torch Tensor")
        distances = points[..., :3].norm(dim=-1)
        mask = torch.logical_and(
            distances >= self.min_distance,
            distances < self.max_distance,
        )
        return points[mask]


class TakePoints:
    """Randomly cap a point tensor without changing tensors below the cap."""

    def __init__(self, max_count: int = 32768) -> None:
        if max_count < 1:
            raise ValueError("max_count must be positive")
        self.max_count = int(max_count)

    def __call__(self, points: Any) -> Any:
        if torch is None or not torch.is_tensor(points):
            raise TypeError("TakePoints expects a torch Tensor")
        if points.shape[0] > self.max_count:
            points = points[torch.randperm(points.shape[0], device=points.device)[: self.max_count]]
        return points


class DatasetAdapter(_DatasetBase):
    """Apply a declarative transform graph to a base dataset.

    ``transform_list`` entries have ``old_key``, ``new_key``, and a callable
    ``transform``. Nested ``list``/``tuple`` values are transformed recursively.
    Static transforms are applied to each frame/view and stacked when all leaves
    become tensors. A transform may set ``is_temporal_transform=True`` and
    receive the complete temporal value through ``apply_temporal_transform``.

    String indices use ``idx-num_frame-height-width``. The base item is copied
    shallowly, temporal values are sliced, and image-like values receive a
    per-request Resize+ToTensor transform without mutating configured transforms.
    """

    _STATIC_KEYS = {"fps", "crossview_mask"}
    _IMAGE_KEYS = {"images", "bbox_condition_images"}

    @staticmethod
    def apply_transform(transform: Callable[[Any], Any], value: Any, stack: bool = True) -> Any:
        if isinstance(value, (list, tuple)):
            transformed = [DatasetAdapter.apply_transform(transform, item, stack) for item in value]
            if not stack:
                return transformed
            if torch is not None and transformed and all(torch.is_tensor(item) for item in transformed):
                shapes = {tuple(item.shape) for item in transformed}
                if len(shapes) != 1:
                    raise ValueError(f"cannot stack transformed values with shapes {sorted(shapes)}")
                return torch.stack(transformed)
            return transformed
        return transform(value)

    @staticmethod
    def apply_temporal_transform(transform: Callable[[Any], Any], value: Any) -> Any:
        """Apply a transform to an entire temporal value.

        Custom transforms can provide ``apply_temporal`` for explicit semantics;
        otherwise their callable interface receives the complete value.
        """

        apply_temporal = getattr(transform, "apply_temporal", None)
        return apply_temporal(value) if callable(apply_temporal) else transform(value)

    def __init__(
        self,
        base_dataset: Any,
        transform_list: list[dict[str, Any]] | None = None,
        pop_list: list[str] | None = None,
        temporal_keys: set[str] | None = None,
    ) -> None:
        if not hasattr(base_dataset, "__len__") or not hasattr(base_dataset, "__getitem__"):
            raise TypeError("base_dataset must implement __len__ and __getitem__")
        self.base_dataset = base_dataset
        self.transform_list = list(transform_list or [])
        self.pop_list = list(pop_list or [])
        self.temporal_keys = set(temporal_keys or ())
        for index, spec in enumerate(self.transform_list):
            if not isinstance(spec, dict):
                raise TypeError(f"transform_list[{index}] must be a mapping")
            if not callable(spec.get("transform")):
                raise TypeError(f"transform_list[{index}].transform must be callable")
            if not spec.get("is_dynamic_transform", False):
                if "old_key" not in spec or "new_key" not in spec:
                    raise ValueError(
                        f"transform_list[{index}] requires old_key and new_key"
                    )

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _dynamic_image_transform(height: int, width: int) -> Callable[[Any], Any]:
        if torch is None:
            raise ImportError("dynamic DatasetAdapter image transforms require PyTorch")
        from torchvision.transforms import Compose, Resize, ToTensor

        return Compose([Resize(size=[height, width]), ToTensor()])

    @staticmethod
    def _slice_temporal_value(value: Any, start: int, count: int) -> Any:
        stop = start + count
        if torch is not None and torch.is_tensor(value):
            if value.ndim == 0:
                return value
            if value.shape[0] < stop:
                return value
            return value[start:stop]
        if isinstance(value, (list, tuple)) and len(value) >= stop:
            return value[start:stop]
        return value

    def _apply_transforms(
        self,
        item: dict[str, Any],
        dynamic_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        for spec in self.transform_list:
            if spec.get("is_dynamic_transform", False):
                transformed_item = spec["transform"](item)
                if not isinstance(transformed_item, dict):
                    raise TypeError("dynamic DatasetAdapter transforms must return a mapping")
                item = transformed_item
                continue
            old_key = str(spec["old_key"])
            new_key = str(spec["new_key"])
            if old_key not in item:
                raise KeyError(f"DatasetAdapter source field is missing: {old_key}")
            transform = spec["transform"]
            if dynamic_size is not None and old_key in self._IMAGE_KEYS:
                transform = self._dynamic_image_transform(*dynamic_size)
            stack = bool(spec.get("stack", True))
            if getattr(transform, "is_temporal_transform", False):
                item[new_key] = self.apply_temporal_transform(transform, item[old_key])
            else:
                item[new_key] = self.apply_transform(transform, item[old_key], stack=stack)
        for key in self.pop_list:
            item.pop(key, None)
        return item

    def __getitem__(self, index: int | str) -> dict[str, Any]:
        if isinstance(index, str):
            fields = index.split("-")
            if len(fields) != 4 or not all(field.strip().lstrip("-").isdigit() for field in fields):
                raise ValueError("string dataset index must be idx-num_frame-height-width")
            base_index, num_frame, height, width = (int(field) for field in fields)
            if num_frame < 1 or height < 1 or width < 1:
                raise ValueError("string dataset index fields must be positive")
            item = dict(self.base_dataset[base_index])
            sequence_length = self._infer_sequence_length(item)
            if sequence_length < num_frame:
                raise ValueError(
                    f"requested {num_frame} frames from an item with {sequence_length} frames"
                )
            start = random.randint(0, sequence_length - num_frame)
            for key, value in list(item.items()):
                if key in self._STATIC_KEYS:
                    continue
                if self.temporal_keys and key not in self.temporal_keys:
                    continue
                item[key] = self._slice_temporal_value(value, start, num_frame)
            return self._apply_transforms(item, dynamic_size=(height, width))
        if not isinstance(index, int):
            raise TypeError(f"dataset index must be int or string, got {type(index)!r}")
        return self._apply_transforms(dict(self.base_dataset[index]))

    @classmethod
    def _infer_sequence_length(cls, item: dict[str, Any]) -> int:
        for key in ("images", "pts", "sample_ids", "bbox_condition_images"):
            value = item.get(key)
            if torch is not None and torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        raise ValueError("cannot infer temporal length from base dataset item")


@dataclass(frozen=True)
class BBoxAnnotation:
    """One validated 2D annotation in absolute pixel ``xyxy`` coordinates."""

    xyxy: tuple[float, float, float, float]
    label: str
    track_id: str | None = None
    score: float | None = None


def make_bbox_annotation(
    xyxy: Sequence[float],
    label: str,
    track_id: str | None = None,
    score: float | None = None,
) -> BBoxAnnotation:
    values = tuple(float(value) for value in xyxy)
    if len(values) != 4 or not np.all(np.isfinite(values)):
        raise ValueError(f"bbox must contain four finite values: {xyxy!r}")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"bbox must have positive area: {xyxy!r}")
    if score is not None and not np.isfinite(float(score)):
        raise ValueError(f"bbox score must be finite: {score!r}")
    normalized_label = str(label).strip()
    if not normalized_label:
        raise ValueError("bbox label must not be empty")
    return BBoxAnnotation(values, normalized_label, track_id, None if score is None else float(score))


@dataclass(frozen=True)
class BBoxViewRecord:
    """One camera view and its optional annotations for a frame."""

    name: str
    path: Path
    boxes: tuple[BBoxAnnotation, ...] = ()
    annotations_available: bool = True


@dataclass(frozen=True)
class BBoxFrameRecord:
    """Dataset-neutral frame record consumed by ``BBoxMotionDataset``."""

    sequence: str
    frame_id: str
    views: tuple[BBoxViewRecord, ...]
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


def draw_bbox_condition(
    image: Image.Image,
    boxes: Sequence[BBoxAnnotation],
    settings: dict[str, Any] | None = None,
) -> Image.Image:
    """Render 2D boxes on a black RGB condition canvas."""

    settings = settings or {}
    background = settings.get("background", (0, 0, 0))
    if isinstance(background, int):
        background = (background,) * 3
    condition = Image.new("RGB", image.size, tuple(int(value) for value in background))
    draw = ImageDraw.Draw(condition)
    color_table = settings.get(
        "color_table",
        {
            "human.pedestrian": (255, 0, 0),
            "vehicle.car": (0, 255, 0),
            "vehicle.truck": (0, 128, 255),
            "vehicle.bus": (255, 255, 0),
            "vehicle.bicycle": (255, 0, 255),
        },
    )
    default_color = tuple(int(value) for value in settings.get("default_color", (255, 255, 255)))
    pen_width = max(1, int(settings.get("pen_width", 3)))
    for annotation in boxes:
        color = tuple(int(value) for value in color_table.get(annotation.label, default_color))
        draw.rectangle(annotation.xyxy, outline=color, width=pen_width)
    return condition


class BBoxMotionDataset(_DatasetBase):
    """Unified temporal dataset for style, bbox, and joint samples.

    Dataset-specific ``*_common.py`` modules parse storage into
    ``BBoxFrameRecord`` objects. ``annotation_mode`` selects whether clips
    require annotations, optionally expose them, or omit them for style-only
    training.
    """

    annotation_source = "ground_truth"
    annotation_quality = 1.0

    def __init__(
        self,
        frame_records: Sequence[BBoxFrameRecord],
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 30.0,
        annotation_mode: str = "required",
        bbox_policy: str = "any",
        min_box_count: int = 1,
        return_annotations: bool = True,
        bbox_condition_settings: dict[str, Any] | None = None,
        normalize_uint16: bool = True,
        uint16_value_range: Sequence[float] = (0.0, 65535.0),
    ) -> None:
        if torch is None:
            raise ImportError("BBoxMotionDataset requires PyTorch")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not frame_records:
            raise ValueError("frame_records must not be empty")
        if not np.isfinite(float(source_fps)) or float(source_fps) <= 0:
            raise ValueError("source_fps must be finite and positive")
        if annotation_mode not in {"required", "optional", "none"}:
            raise ValueError("annotation_mode must be required, optional, or none")
        if bbox_policy not in {"any", "all"}:
            raise ValueError("bbox_policy must be 'any' or 'all'")
        if min_box_count < 1:
            raise ValueError("min_box_count must be positive")
        if len(uint16_value_range) != 2:
            raise ValueError("uint16_value_range must contain (low, high)")
        low, high = (float(value) for value in uint16_value_range)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("uint16_value_range must be finite with high > low")

        self.frame_records = tuple(frame_records)
        self.sequence_length = int(sequence_length)
        self.source_fps = float(source_fps)
        self.annotation_mode = annotation_mode
        self.bbox_policy = bbox_policy
        self.min_box_count = int(min_box_count)
        self.return_annotations = bool(return_annotations)
        self.bbox_condition_settings = dict(bbox_condition_settings or {})
        self.normalize_uint16 = bool(normalize_uint16)
        self.uint16_value_range = (low, high)
        self.dataset_name = str(getattr(type(self), "dataset_name", "dataset"))
        annotations_present = any(
            view.annotations_available
            for record in self.frame_records
            for view in record.views
        )
        self.annotation_source = (
            "none"
            if annotation_mode == "none" or not annotations_present
            else str(getattr(type(self), "annotation_source", "unknown"))
        )
        self.annotation_quality = (
            0.0
            if self.annotation_source == "none"
            else float(getattr(type(self), "annotation_quality", 1.0))
        )
        self._items = self._build_items(fps_stride_tuples or [(0.0, 1.0)])

    def _build_items(
        self,
        fps_stride_tuples: Sequence[Sequence[float]],
    ) -> tuple[tuple[tuple[BBoxFrameRecord, ...], float], ...]:
        groups: dict[str, list[BBoxFrameRecord]] = {}
        for record in self.frame_records:
            if not record.views:
                raise ValueError(f"frame {record.sequence}/{record.frame_id} has no views")
            groups.setdefault(record.sequence, []).append(record)

        items: list[tuple[tuple[BBoxFrameRecord, ...], float]] = []
        for pair in fps_stride_tuples:
            if len(pair) != 2:
                raise ValueError(f"fps_stride_tuples entries must be pairs: {pair!r}")
            fps, stride = (float(value) for value in pair)
            if not np.isfinite(fps) or not np.isfinite(stride) or fps < 0 or stride < 0:
                raise ValueError(f"invalid fps/stride pair: {pair!r}")
            if fps > self.source_fps:
                raise ValueError("target fps cannot exceed source_fps")
            if fps == 0.0:
                if not stride.is_integer() or stride < 1:
                    raise ValueError("fps=0 requires an integer stride >= 1")
                frame_step = 1
                clip_stride = int(stride)
                target_fps = self.source_fps
            else:
                frame_step = max(1, int(round(self.source_fps / fps)))
                clip_stride = max(1, int(round(stride * self.source_fps)))
                target_fps = fps
            for sequence, records in groups.items():
                sampled = records[::frame_step]
                stop = len(sampled) - self.sequence_length + 1
                for start in range(0, max(0, stop), clip_stride):
                    clip = tuple(sampled[start : start + self.sequence_length])
                    box_count = sum(
                        len(view.boxes)
                        for frame in clip
                        for view in frame.views
                        if view.annotations_available
                    )
                    nonempty_frames = sum(
                        any(view.annotations_available and view.boxes for view in frame.views)
                        for frame in clip
                    )
                    accepted = self.annotation_mode != "required" or (
                        box_count >= self.min_box_count
                        and (self.bbox_policy != "all" or nonempty_frames == len(clip))
                    )
                    if accepted:
                        items.append((clip, target_fps))
        if not items:
            raise ValueError("bbox policy produced no valid clips")
        return tuple(items)

    @staticmethod
    def _read_image(
        path: Path,
        normalize_uint16: bool,
        uint16_value_range: tuple[float, float],
    ) -> Image.Image:
        with Image.open(path) as opened:
            image = opened.copy()
        if normalize_uint16 and image.mode in {"I;16", "I;16B", "I;16L", "I"}:
            low, high = uint16_value_range
            array = np.asarray(image, dtype=np.float32)
            array = np.clip((array - low) / (high - low), 0.0, 1.0)
            image = Image.fromarray(array, mode="F")
        return image

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError(f"dataset index must be int, got {type(index)!r}")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        frames, fps = self._items[index]
        annotations_enabled = self.annotation_mode != "none"
        return_annotation_payload = annotations_enabled and self.return_annotations
        images: list[list[Image.Image]] = []
        conditions: list[list[Image.Image]] = []
        box_image_sizes: list[list[tuple[int, int]]] = []
        boxes: list[list[torch.Tensor]] = []
        labels: list[list[tuple[str, ...]]] = []
        track_ids: list[list[tuple[str | None, ...]]] = []
        sample_ids: list[list[str]] = []
        bbox_available: list[list[bool]] = []
        condition_valid: list[list[bool]] = []
        start_time = frames[0].timestamp if frames[0].timestamp is not None else 0.0
        pts: list[list[float]] = []

        for frame_index, frame in enumerate(frames):
            frame_time = (
                frame.timestamp - start_time
                if frame.timestamp is not None and frames[0].timestamp is not None
                else frame_index / fps
            )
            image_row: list[Image.Image] = []
            condition_row: list[Image.Image] = []
            size_row: list[tuple[int, int]] = []
            box_row: list[torch.Tensor] = []
            label_row: list[tuple[str, ...]] = []
            track_row: list[tuple[str | None, ...]] = []
            sample_row: list[str] = []
            availability_row: list[bool] = []
            valid_row: list[bool] = []
            point_row: list[float] = []
            for view in frame.views:
                image = self._read_image(view.path, self.normalize_uint16, self.uint16_value_range)
                image_row.append(image)
                size_row.append((image.height, image.width))
                annotation_available = annotations_enabled and view.annotations_available
                view_boxes = view.boxes if annotation_available else ()
                availability_row.append(bool(annotation_available))
                valid_row.append(bool(view_boxes))
                if annotations_enabled:
                    condition_row.append(
                        draw_bbox_condition(image, view_boxes, self.bbox_condition_settings)
                    )
                if return_annotation_payload:
                    box_row.append(
                        torch.tensor(
                            [box.xyxy for box in view_boxes],
                            dtype=torch.float32,
                        ).reshape(-1, 4)
                    )
                    label_row.append(tuple(box.label for box in view_boxes))
                    track_row.append(tuple(box.track_id for box in view_boxes))
                sample_row.append(f"{self.dataset_name}:{frame.sequence}:{frame.frame_id}:{view.name}")
                point_row.append(float(frame_time))
            images.append(image_row)
            box_image_sizes.append(size_row)
            if annotations_enabled:
                conditions.append(condition_row)
            if return_annotation_payload:
                boxes.append(box_row)
                labels.append(label_row)
                track_ids.append(track_row)
            sample_ids.append(sample_row)
            bbox_available.append(availability_row)
            condition_valid.append(valid_row)
            pts.append(point_row)

        result: dict[str, Any] = {
            "dataset": self.dataset_name,
            "sequence": frames[0].sequence,
            "annotation_mode": self.annotation_mode,
            "bbox_source": self.annotation_source,
            "bbox_quality": torch.tensor(self.annotation_quality, dtype=torch.float32),
            "bbox_available": torch.tensor(bbox_available, dtype=torch.bool),
            "condition_valid": torch.tensor(condition_valid, dtype=torch.bool),
            "fps": torch.tensor(float(fps), dtype=torch.float32),
            "pts": torch.tensor(pts, dtype=torch.float32),
            "pts_unit": "seconds",
            "box_image_sizes": torch.tensor(box_image_sizes, dtype=torch.float32),
            "images": images,
            "sample_ids": sample_ids,
        }
        if annotations_enabled:
            result["bbox_condition_images"] = conditions
        if return_annotation_payload:
            result.update({"boxes": boxes, "labels": labels, "track_ids": track_ids})
        return result

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("cannot collate an empty bbox batch")
        return CollateFnIgnoring(
            ["boxes", "labels", "track_ids", "sample_ids", "sequence"]
        )(batch)


class ConcatMotionDataset(_DatasetBase):
    """Concatenate datasets with explicit relative sampling ratios."""

    def __init__(self, datasets: Sequence[Any], ratios: Sequence[float]) -> None:
        if not datasets or len(datasets) != len(ratios):
            raise ValueError("datasets and ratios must be non-empty and have equal length")
        if any(len(dataset) < 1 for dataset in datasets):
            raise ValueError("ConcatMotionDataset does not accept empty datasets")
        if any(float(ratio) <= 0.0 for ratio in ratios):
            raise ValueError("ConcatMotionDataset ratios must be positive")
        self.datasets = list(datasets)
        self.ratios = [float(ratio) for ratio in ratios]
        self.full_size = math.ceil(
            max(len(dataset) / ratio for dataset, ratio in zip(self.datasets, self.ratios, strict=True))
        )
        self.ranges: list[int] = []
        cumulative = 0
        for ratio in self.ratios:
            cumulative += max(1, int(ratio * self.full_size))
            self.ranges.append(cumulative)

    def __len__(self) -> int:
        return self.ranges[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self.ranges, index)
        begin = 0 if dataset_index == 0 else self.ranges[dataset_index - 1]
        return self.datasets[dataset_index][(index - begin) % len(self.datasets[dataset_index])]


class CollateFnIgnoring:
    """Use PyTorch default collation while preserving selected metadata lists."""

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = list(keys)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if torch is None:
            raise ImportError("CollateFnIgnoring requires PyTorch")
        if not items:
            raise ValueError("cannot collate an empty item list")
        ignored: dict[str, list[Any]] = {}
        for key in self.keys:
            values = [item.pop(key) for item in items if key in item]
            if values and len(values) != len(items):
                raise KeyError(f"collate key {key!r} is missing from part of the batch")
            if values:
                ignored[key] = values
        result = torch.utils.data.default_collate(items)
        result.update(ignored)
        return result


def find_nearest(values: Sequence[float], value: float, return_item: bool = False) -> int | float:
    """Find the nearest value in an ascending sequence."""

    if not values:
        raise ValueError("cannot search an empty sequence")
    index = bisect.bisect_left(values, value)
    if index == 0:
        nearest = 0
    elif index == len(values):
        nearest = len(values) - 1
    else:
        nearest = index - 1 if value - values[index - 1] <= values[index] - value else index
    return values[nearest] if return_item else nearest

def limit_sequence(items: Sequence[Any], max_count: int | None) -> tuple[Any, ...]:
    """Select evenly spaced items while retaining sequence endpoints."""

    if max_count is None or len(items) <= max_count:
        return tuple(items)
    if max_count < 1:
        raise ValueError("max_count must be positive")
    if max_count == 1:
        return (items[len(items) // 2],)
    return tuple(
        items[round(index * (len(items) - 1) / (max_count - 1))]
        for index in range(max_count)
    )

# 文件讲解：
# 1. DatasetAdapter 是模型前的数据处理层：它从 base_dataset 取出字典，
#    对嵌套的 [T][V] 图像逐叶执行 transform，并按形状堆成 Tensor。
# 2. 普通整数索引只访问一个完整 clip；字符串索引
#    idx-num_frame-height-width 会额外随机截取时间窗口并按请求尺寸处理
#    图像。temporal_keys 非空时只有列出的字段会被时间切片。
# 3. BBoxMotionDataset 是 style、bbox、joint 共用的 clip 引擎；各数据集的
#    *_common.py 只负责把原始格式解析成 BBoxFrameRecord。
# 4. annotation_mode=required 只保留满足 bbox_policy 的 clip；optional 保留
#    无框帧并用 bbox_available/condition_valid 标记；none 不生成 condition，
#    也不构造 boxes/labels/track_ids，避免 style 模式的额外 tensor 开销。
# 5. bbox_policy=any 表示一个 clip 至少有一个可用框；bbox_policy=all 表示
#    每个时间帧都必须有可用框。ZUT 等含空帧的数据集默认用 any 保留上下文。
# 6. ConcatMotionDataset 做按比例的重复采样，CollateFnIgnoring 保留变长
#    boxes/labels/track_ids；模型 Tensor transform 应写在 YAML Adapter 中。
