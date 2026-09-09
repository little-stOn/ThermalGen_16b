"""ZUT-FIR-ADAS bbox dataset through the shared DWM contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.zut_fir_adas_common import DEFAULT_CLASS_NAMES, load_zut_records


class MotionDataset(BBoxMotionDataset):
    """Load 16-bit ZUT recordings and their YOLO bbox sidecars."""

    dataset_name = "zut_fir_adas"
    annotation_source = "ground_truth"
    annotation_quality = 1.0

    def __init__(
        self,
        dataset_root: str | Path,
        frame_directory: str = "16BitFrames",
        split: str | None = "all",
        countries: Iterable[str] | None = None,
        routes: Iterable[str] | None = None,
        recordings: Iterable[str] | None = None,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 25.0,
        annotation_mode: str = "required",
        bbox_policy: str = "any",
        min_box_count: int = 1,
        include_empty_frames: bool = True,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        max_frames_per_sequence: int | None = None,
        class_whitelist: Iterable[str] | None = None,
        index_cache: str | Path | None = None,
        use_index_cache: bool = True,
        return_annotations: bool = True,
        bbox_condition_settings: dict[str, Any] | None = None,
        normalize_uint16: bool = True,
        uint16_value_range: Sequence[float] = (0.0, 65535.0),
    ) -> None:
        records = load_zut_records(
            dataset_root,
            frame_directory=frame_directory,
            split=split,
            countries=countries,
            routes=routes,
            recordings=recordings,
            include_empty_frames=include_empty_frames,
            class_names=class_names,
            class_whitelist=class_whitelist,
            index_cache=index_cache,
            use_index_cache=use_index_cache,
            load_annotations=annotation_mode != "none",
            max_frames_per_sequence=max_frames_per_sequence,
            source_fps=source_fps,
        )
        super().__init__(
            records,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            source_fps=source_fps,
            annotation_mode=annotation_mode,
            bbox_policy=bbox_policy,
            min_box_count=min_box_count,
            bbox_condition_settings=bbox_condition_settings,
            normalize_uint16=normalize_uint16,
            uint16_value_range=uint16_value_range,
        )

# 文件讲解：
# 1. ZUT 的目录层级是 country/route/recording；MotionDataset 只负责传递
#    这些真实筛选项，具体发现帧和读取 YOLO TXT 由 zut_fir_adas_common.py
#    完成。
# 2. frame_directory 决定读取原始 16BitFrames 或预处理的
#    16BitTransformed；两者都使用同一份归一化 YOLO 坐标，不能混用错误的
#    标注格式。
# 3. split 会识别官方 recording 名称末尾的 _b benchmark：train/val 使用
#    非 benchmark recording，test/benchmark 只使用 _b。bbox_policy=any
#    保留含框 clip 的时间上下文，all 可要求每一帧都有框。
# 4. ZUT 的 TXT 数量很大，index_cache 会缓存已经解析的 FrameRecord；缓存
#    只保存索引和标注，不保存图像像素，删除 .dwm_cache 即可强制重建。
# 5. 调试示例：
#    ZUT_ROOT=/path/to/zut_fir_adas PYTHONPATH=src \
#      python scripts/prepare/load_dataset_config.py \
#      --cfg configs/datasets/zut_fir_adas.yaml --loader --batches 8
