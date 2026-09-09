"""Local BU-TIV bbox dataset loader.

BU-TIV is a 16-bit thermal-video dataset with native 2D boxes. The loader uses
an explicit synthetic ordinal timebase and does not fabricate missing sensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import bisect
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .butiv_common import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_SOURCE_FPS,
    FrameRecord,
    TIMESTAMP_SOURCE,
    load_multiview_records,
    split_metadata,
)


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]


_DatasetBase = torch.utils.data.Dataset if torch is not None else object
Box = tuple[float, float, float, float]
GeometryTransform = Callable[
    [Image.Image, Sequence[Box], Sequence[str], Sequence[str | None]],
    tuple[Image.Image, Sequence[Box], Sequence[str], Sequence[str | None]],
]

DEFAULT_UINT16_RANGE = (0.0, 65535.0)
DEFAULT_2DBOX_COLOR_TABLE = {
    "human.pedestrian": (0, 255, 0),
    "vehicle.bicycle": (0, 128, 255),
    "vehicle.motorcycle": (255, 0, 255),
    "vehicle.car": (255, 64, 64),
}


# 递归检查字段中是否包含 Tensor，用于区分“可堆叠数据”和 PIL/元数据。
def _iter_leaves(value: Any):
    if torch is not None and torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_leaves(child)
    else:
        yield None


# DatasetAdapter 之前通常仍是 PIL；如果上游已经转成 Tensor，则严格
# 按 [B,T,V,...] 堆叠，避免模型收到无法调用 .to() 的 list。
def _tree_as_tensor(value: Any, key: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"cannot stack {key}: non-tensor leaf {type(value)!r}")
    children = [
        _tree_as_tensor(child, f"{key}[{index}]")
        for index, child in enumerate(value)
    ]
    shapes = {tuple(child.shape) for child in children}
    if len(shapes) != 1:
        raise ValueError(
            f"cannot stack {key}: nested tensor shapes differ: {sorted(shapes)}"
        )
    return torch.stack(children, dim=0)


def _collate_tensor_or_preserve(values: list[Any], key: str) -> Any:
    leaves = list(_iter_leaves(values))
    # DatasetAdapter 负责把这些 PIL 图像变成 Tensor；这里不提前伪造 batch 维。
    if not leaves or all(leaf is None for leaf in leaves):
        # PIL 图像在 DatasetAdapter 前保持为 [B][T][V] 列表。
        return values
    if any(leaf is None for leaf in leaves):
        raise TypeError(f"cannot collate {key}: mixed tensor and non-tensor leaves")
    tensors = [_tree_as_tensor(value, key) for value in values]
    shapes = {tuple(value.shape) for value in tensors}
    if len(shapes) != 1:
        raise ValueError(f"cannot stack {key}: tensor shapes differ: {sorted(shapes)}")
    return torch.stack(tensors, dim=0)


class MotionDataset(_DatasetBase):
    """Load bbox-annotated BU-TIV clips from the local dataset root.

    BU-TIV exposes a single front thermal camera for Marathon and synchronized
    orange/red cameras for Atrium. The loader keeps only dataset-specific
    controls; image conversion and model transforms belong to DatasetAdapter.

    Args:
        dataset_root: Local BU-TIV root.
        sequence_length: Number of frames in each clip.
        fps_stride_tuples: ``(fps, stride)`` sampling pairs. ``fps == 0``
            samples source indices; positive FPS samples the synthetic ordinal
            timebase.
        split: Custom sequence-level split name.
        sensor_channels: Optional explicit view names.
        view_mode: ``single`` creates one sample source per view;
            ``multiview`` keeps synchronized views together.
        check_view_sync: Reject multiview clips outside the time tolerance.
        bbox_condition_settings: Drawing settings for bbox condition images.
        include_empty_frames: Keep clips containing empty annotation frames.
        return_annotations: Include variable-length boxes, labels, and IDs.
        geometry_transform: Joint image/box/label/ID spatial transform.
        normalize_uint16: Convert source 16-bit images to fixed-range float.
        uint16_value_range: Source intensity range used for normalization.
        source_fps: Synthetic ordinal timebase FPS.
        max_frames_per_sequence: Optional prefix cap per source sequence.
        synchronization_tolerance_ms: Maximum multiview timestamp mismatch.
    """

    dataset_name = "butiv"
    annotation_source = "ground_truth"
    annotation_quality = 1.0

    def __init__(
        self,
        dataset_root: str | Path,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[tuple[float, float]] | None = None,
        split: str | None = None,
        sensor_channels: list[str] | None = None,
        view_mode: str = "single",
        check_view_sync: bool = True,
        bbox_condition_settings: dict[str, Any] | None = None,
        include_empty_frames: bool = True,
        return_annotations: bool = False,
        geometry_transform: GeometryTransform | None = None,
        normalize_uint16: bool = True,
        uint16_value_range: tuple[float, float] = DEFAULT_UINT16_RANGE,
        source_fps: float = DEFAULT_SOURCE_FPS,
        max_frames_per_sequence: int | None = None,
        annotation_mode: str = "required",
        synchronization_tolerance_ms: float = 50.0,
    ) -> None:
        if torch is None:
            raise ImportError("BU-TIV MotionDataset requires PyTorch")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if view_mode not in {"single", "multiview"}:
            raise ValueError("view_mode must be 'single' or 'multiview'")
        if sensor_channels is not None and not sensor_channels:
            raise ValueError("sensor_channels must contain at least one view")
        if view_mode == "multiview" and sensor_channels is not None and len(sensor_channels) < 2:
            raise ValueError("view_mode='multiview' requires at least two sensor channels")
        if view_mode == "single" and sensor_channels is not None and len(sensor_channels) != 1:
            raise ValueError("view_mode='single' requires exactly one sensor channel")
        if annotation_mode not in {"required", "optional", "none"}:
            raise ValueError("annotation_mode must be required, optional, or none")
        if not np.isfinite(float(source_fps)) or source_fps <= 0:
            raise ValueError("source_fps must be finite and positive")
        if fps_stride_tuples is None:
            fps_stride_tuples = [(0.0, 1.0)]
        if not fps_stride_tuples:
            raise ValueError("fps_stride_tuples must not be empty")
        for fps, stride in fps_stride_tuples:
            if not np.isfinite(float(fps)) or not np.isfinite(float(stride)):
                raise ValueError("fps and stride must be finite")
            if fps < 0 or stride < 0:
                raise ValueError("fps and stride must be non-negative")
            if fps == 0 and not float(stride).is_integer():
                raise ValueError("fps=0 requires an integer index stride")
            if fps > float(source_fps):
                raise ValueError(
                    f"target fps {fps} exceeds synthetic timebase {source_fps}; "
                    "repeated source frames are not allowed"
                )
        if max_frames_per_sequence is not None and max_frames_per_sequence < 1:
            raise ValueError("max_frames_per_sequence must be positive")
        if not np.isfinite(float(synchronization_tolerance_ms)) or synchronization_tolerance_ms < 0:
            raise ValueError("synchronization_tolerance_ms must be finite and non-negative")
        low, high = (float(value) for value in uint16_value_range)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("uint16_value_range must be finite with high > low")
        self.root = self._resolve_root(dataset_root)
        self.sequence_length = int(sequence_length)
        self.fps_stride_tuples = [
            (float(fps), float(stride)) for fps, stride in fps_stride_tuples
        ]
        self.split = split
        self.sensor_channels = list(sensor_channels) if sensor_channels is not None else None
        self.view_mode = view_mode
        self.check_view_sync = bool(check_view_sync)
        self.annotation_mode = annotation_mode
        self.bbox_condition_settings = dict(bbox_condition_settings or {})
        self.synchronization_tolerance_ms = float(synchronization_tolerance_ms)
        self.include_empty_frames = bool(include_empty_frames)
        self.return_annotations = bool(return_annotations)
        self.geometry_transform = geometry_transform
        self.normalize_uint16 = bool(normalize_uint16)
        self.uint16_value_range = (low, high)
        self._uint16_inverse_range = 1.0 / (high - low)
        self.source_fps = float(source_fps)
        self.synthetic_source_fps = self.source_fps

        self.sequence_records = load_multiview_records(
            self.root,
            split,
            source_fps=self.source_fps,
            load_annotations=annotation_mode != "none",
            max_frames_per_sequence=max_frames_per_sequence,
        )
        self.records = [
            record
            for views in self.sequence_records.values()
            for records in views.values()
            for record in records
        ]
        if not self.records:
            raise ValueError(f"BU-TIV split {split!r} contains no usable frames")

        discovered = {label for record in self.records for label in record.labels}
        self.class_to_idx = {
            name: index + 1 for index, name in enumerate(DEFAULT_CLASS_NAMES)
        }
        for label in sorted(discovered - set(DEFAULT_CLASS_NAMES)):
            self.class_to_idx[label] = len(self.class_to_idx) + 1
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}
        self.items = tuple(self._build_items())
        if not self.items:
            raise ValueError("BU-TIV contains no clips for the requested sequence_length")
        view_counts = {len(item["views"]) for item in self.items}
        if len(view_counts) != 1:
            raise ValueError(
                "BU-TIV dataset mixes view counts; use view_mode='single' or "
                "construct a multiview dataset from one fixed-view split"
            )
        self.view_count = view_counts.pop()
        self.views_by_scene = {
            scene: list(views) for scene, views in self.sequence_records.items()
        }
        item_view_tuples = {tuple(item["views"]) for item in self.items}
        self.view_names = (
            list(next(iter(item_view_tuples)))
            if len(item_view_tuples) == 1
            else None
        )
        self.dataset_metadata = self._build_dataset_metadata()

    @staticmethod
    def _resolve_root(dataset_root: str | Path) -> Path:
        root = Path(dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"BU-TIV root does not exist: {root}")
        return root

    def _selection_plans(
        self, scene: str, available: dict[str, list[FrameRecord]]
    ) -> list[list[str]]:
        """Return per-scene view selections, one item group per selection."""

        if self.sensor_channels is not None:
            selected = list(self.sensor_channels)
            missing = [view for view in selected if view not in available]
            if missing:
                raise ValueError(
                    f"BU-TIV scene {scene!r} is missing requested views {missing}; "
                    f"available={list(available)}"
                )
            return [selected]
        if self.view_mode == "single":
            # single 模式把每个视角作为独立数据源，Atrium 的 orange/red
            # 都会各自生成 V=1 的 clip。
            return [[view] for view in available]
        if len(available) < 2:
            raise ValueError(
                f"view_mode='multiview' requires at least two views per scene; "
                f"scene {scene!r} only has {list(available)}"
            )
        return [list(available)]

    @staticmethod
    def _nearest_index(timestamps: Sequence[float], target_ms: float) -> int:
        position = bisect.bisect_left(timestamps, target_ms)
        candidates = [
            max(0, min(len(timestamps) - 1, position + offset))
            for offset in (-1, 0)
        ]
        return min(candidates, key=lambda index: abs(timestamps[index] - target_ms))

    @staticmethod
    def _same_source_segment(
        views: dict[str, list[FrameRecord]],
        selected_views: list[str],
        segment: list[list[int]],
    ) -> bool:
        anchor_view = selected_views[0]
        # 一个 clip 的全部时间步必须属于同一 segment；同时要求各视角
        # 在每个时间步也属于同一 segment，避免跨断点拼接视频。
        anchor_segment_ids = [
            views[anchor_view][indices[0]].segment_id for indices in segment
        ]
        if not anchor_segment_ids or len(set(anchor_segment_ids)) != 1:
            return False
        segment_id = anchor_segment_ids[0]
        return all(
            views[view][indices[view_index]].segment_id == segment_id
            for indices in segment
            for view_index, view in enumerate(selected_views)
        )

    # fps=0 按源帧索引取连续帧；正 fps 才按 synthetic timebase 抽帧。
    def _segment_indices(
        self,
        views: dict[str, list[FrameRecord]],
        selected_views: list[str],
        fps: float,
        stride: float,
    ) -> list[list[list[int]]]:
        anchor = views[selected_views[0]]
        anchor_timestamps = [record.timestamp_ms for record in anchor]
        if fps == 0.0:
            step = max(1, int(round(stride)))
            max_start = len(anchor) - self.sequence_length + 1
            segments: list[list[list[int]]] = []
            for start in range(0, max_start, step):
                segment = [
                    [start + offset for _view in selected_views]
                    for offset in range(self.sequence_length)
                ]
                if self._same_source_segment(views, selected_views, segment):
                    segments.append(segment)
            return segments

        # 这里的目标间隔和起点间隔都基于 synthetic ordinal timebase；
        # 它们不是 BU-TIV 的真实采集时间。
        interval_ms = 1000.0 / fps
        latest_start_ms = anchor_timestamps[-1] - self.sequence_length * interval_ms
        if latest_start_ms < anchor_timestamps[0]:
            return []
        if stride == 0.0:
            start_times = [
                timestamp
                for timestamp in anchor_timestamps
                if timestamp <= latest_start_ms + 1e-6
            ]
        else:
            start_times = []
            start_time = anchor_timestamps[0]
            while start_time <= latest_start_ms + 1e-6:
                start_times.append(start_time)
                start_time += stride * 1000.0

        segments = []
        seen: set[tuple[tuple[int, ...], ...]] = set()
        timestamp_lists = {
            view: [record.timestamp_ms for record in views[view]]
            for view in selected_views
        }
        for start_time in start_times:
            expected_times = [
                start_time + offset * interval_ms for offset in range(self.sequence_length)
            ]
            segment = [
                [
                    self._nearest_index(timestamp_lists[view], expected_time)
                    for view in selected_views
                ]
                for expected_time in expected_times
            ]
            key = tuple(tuple(row) for row in segment)
            if key in seen or not self._same_source_segment(views, selected_views, segment):
                continue
            seen.add(key)
            if self.check_view_sync and not self._within_expected_time_error(
                views, selected_views, segment, expected_times
            ):
                continue
            segments.append(segment)
        return segments

    def _within_expected_time_error(
        self,
        views: dict[str, list[FrameRecord]],
        selected_views: list[str],
        segment: list[list[int]],
        expected_times: list[float],
    ) -> bool:
        errors = [
            abs(
                views[view][indices[view_index]].timestamp_ms - expected_time
            )
            for indices, expected_time in zip(segment, expected_times, strict=True)
            for view_index, view in enumerate(selected_views)
        ]
        anchor = views[selected_views[0]]
        source_frame_delta = (
            anchor[1].timestamp_ms - anchor[0].timestamp_ms
            if len(anchor) > 1
            else 1000.0 / self.synthetic_source_fps
        )
        quantization_tolerance = max(0.5 * source_frame_delta, 1e-6)
        return max(errors, default=0.0) <= quantization_tolerance and (
            self._synchronized_segment(views, selected_views, segment)
        )

    def _synchronized_segment(
        self,
        views: dict[str, list[FrameRecord]],
        selected_views: list[str],
        segment: list[list[int]],
    ) -> bool:
        if not self._same_source_segment(views, selected_views, segment):
            return False
        if not self.check_view_sync or len(selected_views) <= 1:
            return True
        anchor_view = selected_views[0]
        for indices in segment:
            anchor_timestamp = views[anchor_view][indices[0]].timestamp_ms
            if any(
                abs(
                    views[view][indices[view_index]].timestamp_ms
                    - anchor_timestamp
                )
                > self.synchronization_tolerance_ms
                for view_index, view in enumerate(selected_views[1:], start=1)
            ):
                return False
        return True

    def _build_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for scene, views in self.sequence_records.items():
            for selected_views in self._selection_plans(scene, views):
                for fps, stride in self.fps_stride_tuples:
                    for segment in self._segment_indices(
                        views, selected_views, fps, stride
                    ):
                        if self.annotation_mode == "required" and not self.include_empty_frames:
                            has_empty_frame = any(
                                not views[view][indices[view_index]].boxes
                                for indices in segment
                                for view_index, view in enumerate(selected_views)
                            )
                            if has_empty_frame:
                                continue
                        items.append(
                            {
                                "scene": scene,
                                "start": segment[0][0],
                                "views": selected_views,
                                "segment": segment,
                                "fps": fps,
                            }
                        )
        return items

    def _build_dataset_metadata(self) -> dict[str, Any]:
        metadata = split_metadata(self.split)
        metadata.update(
            {
                "dataset": "BU-TIV",
                "timestamp_source": TIMESTAMP_SOURCE,
                "source_fps_verified": False,
                "timebase_fps": self.synthetic_source_fps,
                "timebase_note": "unverified synthetic ordinal timebase; not acquisition timestamps",
                "views": list(self.view_names) if self.view_names is not None else None,
                "views_by_scene": self.views_by_scene,
                "temporal_contract": "fps samples synthetic ordinal times; stride spaces clip starts; XML frame gaps and sequence boundaries are never crossed",
                "fps_stride_tuples": [list(value) for value in self.fps_stride_tuples],
                "include_empty_frames": self.include_empty_frames,
                "view_mode": self.view_mode,
                "view_count": self.view_count,
            }
        )
        return metadata

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    # 原始 PNG 保持 16-bit；mode F 只用于把固定范围强度安全交给 ToTensor。
    def uint16_to_float_image(
        image: Image.Image,
        value_range: tuple[float, float] = DEFAULT_UINT16_RANGE,
    ) -> Image.Image:
        """Convert a 16-bit image to fixed-range float32 PIL mode ``F``."""

        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError(f"BU-TIV image must be single-channel, got {array.shape}")
        low, high = value_range
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("value_range must be finite with high > low")
        array_float = array.astype(np.float32, copy=False)
        normalized = np.clip((array_float - low) / (high - low), 0.0, 1.0)
        return Image.fromarray(normalized.astype(np.float32, copy=False), mode="F")

    def _open_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            image.load()
            # BU-TIV 的原始帧是单通道 16-bit PNG；先校验模式，再决定是否归一化。
            if image.mode not in {"I;16", "I;16B", "I"}:
                raise ValueError(f"BU-TIV image is not 16-bit grayscale: {path} ({image.mode})")
            if self.normalize_uint16:
                # 使用固定范围转成 mode F，避免 torchvision 把 I;16 当作有符号 int16。
                array = np.asarray(image)
                if array.ndim != 2:
                    raise ValueError(f"BU-TIV image must be single-channel, got {array.shape}")
                array_float = array.astype(np.float32, copy=False)
                low, _high = self.uint16_value_range
                normalized = np.clip(
                    (array_float - low) * self._uint16_inverse_range,
                    0.0,
                    1.0,
                )
                return Image.fromarray(normalized.astype(np.float32, copy=False), mode="F")
            return image.copy()

    @staticmethod
    def _clip_box(box: Box, width: int, height: int) -> Box | None:
        x1, y1, x2, y2 = box
        clipped: Box = (
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            return None
        return clipped

    def _validated_annotations(
        self,
        boxes: Sequence[Box],
        labels: Sequence[str],
        track_ids: Sequence[str | None],
        width: int,
        height: int,
    ) -> tuple[tuple[Box, ...], tuple[str, ...], tuple[str | None, ...]]:
        # 过滤必须对 box、label、track_id 使用同一条有效性规则，
        # 否则三者会失去一一对应关系。
        if not (len(boxes) == len(labels) == len(track_ids)):
            raise ValueError("boxes, labels, and track_ids must have equal lengths")
        valid_boxes: list[Box] = []
        valid_labels: list[str] = []
        valid_track_ids: list[str | None] = []
        for box, label, track_id in zip(boxes, labels, track_ids, strict=True):
            clipped = self._clip_box(tuple(float(value) for value in box), width, height)
            if clipped is None:
                continue
            valid_boxes.append(clipped)
            valid_labels.append(str(label))
            valid_track_ids.append(track_id)
        return tuple(valid_boxes), tuple(valid_labels), tuple(valid_track_ids)

    def _box_tensors(
        self,
        boxes: Sequence[Box],
        labels: Sequence[str],
        track_ids: Sequence[str | None],
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str | None, ...], tuple[Box, ...], tuple[str, ...]]:
        valid_boxes, valid_labels, valid_track_ids = self._validated_annotations(
            boxes, labels, track_ids, width, height
        )
        box_tensor = torch.tensor(valid_boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(
            [self.class_to_idx[label] for label in valid_labels],
            dtype=torch.long,
        )
        return box_tensor, label_tensor, valid_track_ids, valid_boxes, valid_labels

    @staticmethod
    def _color_for_label(label: str, table: dict[str, Any]) -> tuple[int, int, int]:
        value = table.get(
            label,
            DEFAULT_2DBOX_COLOR_TABLE.get(label, (255, 255, 255)),
        )
        return tuple(int(channel) for channel in value)

    def _make_bbox_image(
        self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor
    ) -> Image.Image:
        settings = self.bbox_condition_settings
        background = tuple(settings.get("background_color", (0, 0, 0)))
        colors = dict(settings.get("color_table", {}))
        width = max(1, int(settings.get("pen_width", 4)))
        result = Image.new("RGB", image.size, background)
        draw = ImageDraw.Draw(result)
        for box, label in zip(boxes.tolist(), labels.tolist(), strict=True):
            name = self.idx_to_class[int(label)]
            draw.rectangle(box, outline=self._color_for_label(name, colors), width=width)
        return result


    def _sample_clip_geometry_transform(self) -> GeometryTransform | None:
        # 每个 clip 只采样一次随机几何参数，再复用于所有帧和视角。
        transform = self.geometry_transform
        if transform is None:
            return None
        sampler = getattr(transform, "sample_clip", None)
        if sampler is None:
            sampler = getattr(transform, "sample", None)
        if sampler is None:
            return transform
        sampled = sampler()
        if not callable(sampled):
            raise TypeError("geometry transform sampler must return a callable transform")
        return sampled

    def _prepare_record(
        self,
        record: FrameRecord,
        geometry_transform: GeometryTransform | None,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, tuple[str | None, ...]]:
        image = self._open_image(record.image_path)
        boxes: Sequence[Box] = record.boxes
        labels: Sequence[str] = record.labels
        track_ids: Sequence[str | None] = record.track_ids
        if geometry_transform is not None:
            transformed = geometry_transform(image, boxes, labels, track_ids)
            if not isinstance(transformed, tuple) or len(transformed) != 4:
                raise ValueError("geometry_transform must return (image, boxes, labels, track_ids)")
            image, boxes, labels, track_ids = transformed
            if not isinstance(image, Image.Image):
                raise TypeError("geometry_transform must return a PIL image before condition rendering")
        frame_boxes, frame_labels, valid_ids, _valid_boxes, _valid_labels = self._box_tensors(
            boxes, labels, track_ids, *image.size
        )
        return image, frame_boxes, frame_labels, valid_ids

    # 只在当前 clip 读取 T×V 张图像；初始化阶段仅建立路径和标注索引。
    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError("BU-TIV MotionDataset expects an integer index")
        item = self.items[index]
        views = item["views"]
        scene_records = self.sequence_records[item["scene"]]
        segment_records = [
            [
                scene_records[view][record_index]
                for view, record_index in zip(views, row, strict=True)
            ]
            for row in item["segment"]
        ]
        clip_geometry_transform = self._sample_clip_geometry_transform()
        images: list[list[Any]] = []
        boxes: list[list[torch.Tensor]] = []
        labels: list[list[torch.Tensor]] = []
        track_ids: list[list[tuple[str | None, ...]]] = []
        bbox_images: list[list[Image.Image]] = []
        box_image_sizes: list[list[tuple[int, int]]] = []
        bbox_available: list[list[bool]] = []
        condition_valid: list[list[bool]] = []

        for records_at_time in segment_records:
            image_row: list[Any] = []
            boxes_row: list[torch.Tensor] = []
            labels_row: list[torch.Tensor] = []
            track_ids_row: list[tuple[str | None, ...]] = []
            bbox_row: list[Image.Image] = []
            size_row: list[tuple[int, int]] = []
            available_row: list[bool] = []
            valid_row: list[bool] = []
            for record in records_at_time:
                image, frame_boxes, frame_labels, valid_ids = self._prepare_record(
                    record, clip_geometry_transform
                )
                image_row.append(image)
                size_row.append((image.height, image.width))
                boxes_row.append(frame_boxes)
                labels_row.append(frame_labels)
                track_ids_row.append(valid_ids)
                available = self.annotation_mode != "none"
                available_row.append(available)
                valid_row.append(bool(available and frame_boxes.shape[0]))
                if self.annotation_mode != "none":
                    bbox_row.append(self._make_bbox_image(image, frame_boxes, frame_labels))
            images.append(image_row)
            box_image_sizes.append(size_row)
            boxes.append(boxes_row)
            labels.append(labels_row)
            track_ids.append(track_ids_row)
            bbox_available.append(available_row)
            condition_valid.append(valid_row)
            if self.annotation_mode != "none":
                bbox_images.append(bbox_row)

        start_timestamp = segment_records[0][0].timestamp_ms
        pts = torch.tensor(
            [
                [record.timestamp_ms - start_timestamp for record in records_at_time]
                for records_at_time in segment_records
            ],
            dtype=torch.float32,
        )
        result: dict[str, Any] = {
            "dataset": self.dataset_name,
            "sequence": item["scene"],
            "annotation_mode": self.annotation_mode,
            "bbox_source": (
                "none" if self.annotation_mode == "none" else self.annotation_source
            ),
            "bbox_quality": torch.tensor(
                0.0 if self.annotation_mode == "none" else self.annotation_quality,
                dtype=torch.float32,
            ),
            "bbox_available": torch.tensor(bbox_available, dtype=torch.bool),
            "condition_valid": torch.tensor(condition_valid, dtype=torch.bool),
            "fps": torch.tensor(float(item["fps"]), dtype=torch.float32),
            "pts": pts,
            "pts_unit": "milliseconds",
            "box_image_sizes": torch.tensor(box_image_sizes, dtype=torch.float32),
            "images": images,
            "sample_ids": [
                [record.sample_id for record in records_at_time]
                for records_at_time in segment_records
            ],
        }
        if self.annotation_mode != "none":
            result["bbox_condition_images"] = bbox_images
        if self.return_annotations and self.annotation_mode != "none":
            result["boxes"] = boxes
            result["labels"] = labels
            result["track_ids"] = track_ids
        return result


    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # DatasetAdapter 会逐帧处理 PIL 图像；若这里拿到的已是 Tensor，
        # 则必须严格堆叠，不能把固定形状数据退化为 list。
        """Collate tensor fields strictly and preserve variable metadata lists."""

        if not batch:
            raise ValueError("cannot collate an empty BU-TIV batch")
        keys = set(batch[0])
        if any(set(item) != keys for item in batch[1:]):
            raise ValueError("BU-TIV batch items have different keys")
        result: dict[str, Any] = {}
        variable_keys = {"boxes", "labels", "track_ids", "sample_ids"}
        tensor_tree_keys = {"images", "bbox_condition_images"}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == "fps":
                if any(not torch.is_tensor(value) or value.ndim != 0 for value in values):
                    raise ValueError("fps must be a scalar tensor in every item")
                result[key] = torch.stack(values, dim=0)
            elif key == "pts":
                shapes = {tuple(value.shape) for value in values}
                if len(shapes) != 1:
                    raise ValueError(f"pts shapes differ across batch: {sorted(shapes)}")
                result[key] = torch.stack(values, dim=0)
            elif key in variable_keys:
                result[key] = values
            elif key in tensor_tree_keys:
                result[key] = _collate_tensor_or_preserve(values, key)
            elif key == "sequence":
                result[key] = values
            elif all(torch.is_tensor(value) for value in values):
                result[key] = _collate_tensor_or_preserve(values, key)
            elif all(isinstance(value, Number) for value in values):
                result[key] = torch.tensor(values)
            elif all(isinstance(value, str) for value in values):
                result[key] = values
            else:
                raise TypeError(f"unsupported BU-TIV batch field {key!r}")
        return result

# 文件讲解：
# 1. MotionDataset 只接受 dataset_root 和 BU-TIV 自身需要的采样、视角、
#    同步、bbox condition、16-bit 归一化参数；它不再接受 fs、3D box、
#    HD map、caption 或 sample_data 等外部数据集兼容参数。
# 2. butiv_common.py 先读取 XML 和图像路径，按 scene/view 建立帧索引；
#    _build_items 再按 sequence_length 和 fps_stride_tuples 生成 clip，并且
#    不跨可观测帧间隔、序列边界或不同视角错误拼接。
# 3. __getitem__ 只在真正取样时读取 T×V 张图，执行几何变换、裁剪框、
#    生成 bbox_condition_images，并可按 return_annotations 返回 boxes、
#    labels、track_ids。原始图像保持 PIL，Resize/ToTensor 由 config 的
#    DatasetAdapter 统一处理。
# 4. 单视角模式会把 Atrium 的 orange/red 分别作为独立样本；multiview
#    模式才把同步视角放在同一个 [T][V] clip 中。check_view_sync 和
#    synchronization_tolerance_ms 只用于这个真实存在的多视角同步问题。
# 5. 讲解给别人时可概括为“索引阶段不读图，取样阶段读 T×V，先保证
#    image/box 对齐，再画 condition，最后由 Adapter 变成模型 Tensor”。
# 6. 调试示例：
#    BUTIV_ROOT=/path/to/bu_tiv PYTHONPATH=src \
#      python scripts/prepare/load_dataset_config.py \
#      --cfg configs/datasets/butiv.yaml --loader --batches 8
