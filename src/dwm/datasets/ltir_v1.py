"""LTIR v1 bbox dataset through the shared DWM contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.ltir_v1_common import load_ltir_records


class MotionDataset(BBoxMotionDataset):
    """Load 8-bit or 16-bit LTIR sequences with per-frame ground truth."""

    dataset_name = "ltir_v1"
    annotation_source = "ground_truth"
    annotation_quality = 1.0

    def __init__(
        self,
        dataset_root: str | Path,
        mode: str = "16bit",
        sequence_names: Iterable[str] | None = None,
        max_frames_per_sequence: int | None = None,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 25.0,
        annotation_mode: str = "required",
        bbox_policy: str = "any",
        min_box_count: int = 1,
        return_annotations: bool = True,
        bbox_condition_settings: dict[str, Any] | None = None,
        normalize_uint16: bool = True,
        uint16_value_range: Sequence[float] = (0.0, 65535.0),
    ) -> None:
        records = load_ltir_records(
            dataset_root,
            mode=mode,
            sequence_names=sequence_names,
            load_annotations=annotation_mode != "none",
            max_frames_per_sequence=max_frames_per_sequence,
            source_fps=source_fps,
        )
        super().__init__(
            records,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            source_fps=source_fps,
            bbox_policy=bbox_policy,
            min_box_count=min_box_count,
            annotation_mode=annotation_mode,
            return_annotations=return_annotations,
            bbox_condition_settings=bbox_condition_settings,
            normalize_uint16=normalize_uint16,
            uint16_value_range=uint16_value_range,
        )

# 文件讲解：
# 1. LTIR v1 的 MotionDataset 只保留 root、8/16-bit 模式、sequence
#    筛选、时间采样、bbox 策略和强度归一化；它没有第二个同步视角。
# 2. ltir_v1_common.py 逐序列读取数字帧名和 groundtruth.txt，把四点或
#    八点标注转成绝对 xyxy。尺寸只从每个序列的第一帧读取，避免初始化时
#    对每张图重复打开文件。
# 3. 16-bit 默认转成 [0,1] 的 PIL F 图像；bbox_condition_images 始终由
#    shared BBoxMotionDataset 生成，模型 tensor 由配置 Adapter 生成。
# 4. 讲解这个 loader 时强调“一个目录就是一个时间序列、每帧一个目标框、
#    clip 不能跨序列”；mode=all 可显式合并 8-bit 与 16-bit 序列。
# 5. 调试示例：
#    LTIR_ROOT=/path/to/ltir_v1 PYTHONPATH=src \
#      python scripts/prepare/load_dataset_config.py \
#      --cfg configs/datasets/ltir_v1.yaml --loader --batches 8
