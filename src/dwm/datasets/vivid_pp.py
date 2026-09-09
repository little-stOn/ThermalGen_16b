"""VIVID++ raw and generated-bbox dataset through one loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.vivid_pp_common import (
    load_vivid_raw_records,
    load_vivid_records,
    merge_vivid_annotations,
)


class MotionDataset(BBoxMotionDataset):
    """Load raw VIVID++ clips or overlay generated bbox annotations."""

    dataset_name = "vivid_pp"
    annotation_source = "provisional_generated"
    annotation_quality = 0.5

    def __init__(
        self,
        annotation_root: str | Path | None = None,
        dataset_root: str | Path | None = None,
        modality: str = "thermal",
        view_mode: str = "single",
        subsets: Iterable[str] | None = None,
        max_frames_per_bag: int | None = None,
        synchronization_tolerance_ms: float = 50.0,
        raw_cache_dir: str | Path | None = None,
        use_raw_cache: bool = True,
        min_score: float = 0.0,
        class_whitelist: Iterable[str] | None = None,
        accepted_statuses: Iterable[str] | None = ("accepted",),
        box_keys: Iterable[str] | None = None,
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
        if annotation_mode not in {"required", "optional", "none"}:
            raise ValueError("annotation_mode must be required, optional, or none")
        selected_box_keys = box_keys if box_keys is not None else (
            "bbox_thr_lidar_xyxy",
            "bbox_thr_geometry_xyxy",
            "bbox_thr_unexpanded_xyxy",
            "bbox_thr_xyxy",
        )
        if annotation_mode == "none":
            if annotation_root is not None:
                records = load_vivid_records(
                    annotation_root,
                    modality=modality,
                    view_mode=view_mode,
                    accepted_statuses=accepted_statuses,
                    box_keys=selected_box_keys,
                    load_annotations=False,
                )
            elif dataset_root is not None:
                records = load_vivid_raw_records(
                    dataset_root,
                    subsets=subsets,
                    modality=modality,
                    view_mode=view_mode,
                    max_frames_per_bag=max_frames_per_bag,
                    synchronization_tolerance_ms=synchronization_tolerance_ms,
                    cache_dir=raw_cache_dir,
                    use_cache=use_raw_cache,
                )
            else:
                raise ValueError(
                    "VIVID++ dataset_root or annotation_root is required for style mode"
                )
        elif annotation_mode == "required":
            if annotation_root is None:
                raise ValueError("VIVID++ annotation_root is required for bbox mode")
            records = load_vivid_records(
                annotation_root,
                modality=modality,
                view_mode=view_mode,
                min_score=min_score,
                class_whitelist=class_whitelist,
                accepted_statuses=accepted_statuses,
                box_keys=selected_box_keys,
            )
        else:
            if annotation_root is not None and dataset_root is None:
                records = load_vivid_records(
                    annotation_root,
                    modality=modality,
                    view_mode=view_mode,
                    min_score=min_score,
                    class_whitelist=class_whitelist,
                    accepted_statuses=accepted_statuses,
                    box_keys=selected_box_keys,
                )
            elif dataset_root is None:
                raise ValueError("VIVID++ dataset_root is required for optional mode")
            else:
                raw_records = load_vivid_raw_records(
                    dataset_root,
                    subsets=subsets,
                    modality=modality,
                    view_mode=view_mode,
                    max_frames_per_bag=max_frames_per_bag,
                    synchronization_tolerance_ms=synchronization_tolerance_ms,
                    cache_dir=raw_cache_dir,
                    use_cache=use_raw_cache,
                )
                if annotation_root is None:
                    records = raw_records
                else:
                    annotated_records = load_vivid_records(
                        annotation_root,
                        modality=modality,
                        view_mode=view_mode,
                        min_score=min_score,
                        class_whitelist=class_whitelist,
                        accepted_statuses=accepted_statuses,
                        box_keys=selected_box_keys,
                    )
                    records = merge_vivid_annotations(raw_records, annotated_records)
        super().__init__(
            records,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            source_fps=source_fps,
            annotation_mode=annotation_mode,
            bbox_policy=bbox_policy,
            min_box_count=min_box_count,
            return_annotations=return_annotations,
            bbox_condition_settings=bbox_condition_settings,
            normalize_uint16=normalize_uint16,
            uint16_value_range=uint16_value_range,
        )


# 文件讲解：
# 1. 一个 MotionDataset 通过 annotation_mode 选择 raw style、generated bbox、
#    annotation-only style 或 raw+generated optional 三种路径，不复制 loader。
# 2. raw 模式从 ROS1 bag 流式提取 RGB/thermal 图像到缓存；generated 模式
#    读取 frames.jsonl/detections.jsonl；none + annotation_root 只读取 frames。
# 3. optional 以 raw 帧为全集叠加生成框，没有匹配框的帧保留并由 mask 标记。
# 4. max_frames_per_bag 是显式的数据规模控制，None 才表示完整提取；raw 与
#    generated 索引都使用 cache，缓存中不保存 bbox 以外的训练状态。
