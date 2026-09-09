"""MS2 raw and generated-bbox dataset through one loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.ms2_common import (
    load_ms2_raw_records,
    load_ms2_records,
    merge_ms2_annotations,
)


class MotionDataset(BBoxMotionDataset):
    """Load raw MS2 thermal clips or overlay generated bbox annotations."""

    dataset_name = "ms2"
    annotation_source = "pseudo_high_confidence"
    annotation_quality = 0.75

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_root: str | Path | None = None,
        records_file: str | Path = "records.jsonl",
        sequences: Iterable[str] | None = None,
        min_score: float = 0.0,
        class_whitelist: Iterable[str] | None = None,
        accepted_statuses: Iterable[str] | None = ("accepted_highconf",),
        index_cache: str | Path | None = None,
        use_index_cache: bool = True,
        max_frames_per_sequence: int | None = None,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 10.0,
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
        raw_records = None
        if annotation_mode in {"none", "optional"}:
            raw_records = load_ms2_raw_records(
                dataset_root,
                sequences=sequences,
                max_frames_per_sequence=max_frames_per_sequence,
                source_fps=source_fps,
            )
        if annotation_mode == "none":
            records = raw_records
        else:
            if annotation_root is None:
                raise ValueError("MS2 annotation_root is required for annotated modes")
            annotated = load_ms2_records(
                annotation_root,
                dataset_root,
                records_file=records_file,
                sequences=sequences,
                min_score=min_score,
                class_whitelist=class_whitelist,
                accepted_statuses=accepted_statuses,
                index_cache=index_cache,
                use_index_cache=use_index_cache,
            )
            records = annotated if annotation_mode == "required" else merge_ms2_annotations(
                raw_records,
                annotated,
            )
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
# 1. MS2 的单一 MotionDataset 根据 annotation_mode 选择原始 thermal 索引、
#    生成 bbox 索引或二者合并；没有 style/bbox 两套 loader 文件。
# 2. none 只读取 raw sync_data；required 只保留 records.jsonl 中有框的 clip；
#    optional 以 raw 帧为全集叠加生成框，没有框的帧保留但 condition_valid=False。
# 3. annotation_root 只在 required/optional 时需要；bbox 坐标已经是 thermal
#    像素坐标，不会在这里重复执行 RGB-to-thermal 投影。
