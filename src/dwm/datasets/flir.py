"""FLIR ADAS v2 bbox dataset through the shared DWM contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.flir_common import load_flir_records


class MotionDataset(BBoxMotionDataset):
    """Load COCO-annotated FLIR image/video sequences."""

    dataset_name = "flir"
    annotation_source = "ground_truth"
    annotation_quality = 1.0

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        modality: str = "thermal",
        view_mode: str = "single",
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 30.0,
        annotation_mode: str = "required",
        bbox_policy: str = "any",
        min_box_count: int = 1,
        category_whitelist: Iterable[str] | None = None,
        return_annotations: bool = True,
        bbox_condition_settings: dict[str, Any] | None = None,
        normalize_uint16: bool = False,
        uint16_value_range: Sequence[float] = (0.0, 255.0),
    ) -> None:
        if view_mode not in {"single", "multiview"}:
            raise ValueError("FLIR view_mode must be single or multiview")
        if view_mode == "multiview" and str(split).lower() != "test":
            raise ValueError("FLIR multiview pairing is only defined for video_test")
        records = load_flir_records(
            dataset_root,
            split=split,
            modality=modality,
            view_mode=view_mode,
            category_whitelist=category_whitelist,
            load_annotations=annotation_mode != "none",
        )
        super().__init__(
            records,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            source_fps=source_fps,
            bbox_policy=bbox_policy,
            min_box_count=min_box_count,
            return_annotations=return_annotations,
            annotation_mode=annotation_mode,
            bbox_condition_settings=bbox_condition_settings,
            normalize_uint16=normalize_uint16,
            uint16_value_range=uint16_value_range,
        )

# 文件讲解：
# 1. MotionDataset 是薄封装：它只接收 FLIR root、split、modality、视角
#    模式、clip 采样和 bbox 过滤参数，然后把解析工作交给 flir_common.py。
# 2. 默认读取 thermal COCO train；COCO 的 [x,y,w,h] 在 common parser 中
#    转成绝对 xyxy。category_whitelist 可在索引阶段丢弃不需要的类别。
# 3. multiview 只允许 video_test，因为 train/val 的 RGB 与 thermal 使用
#    不同 video ID；test 通过官方 rgb_to_thermal_vid_map.json 精确配对。
# 4. loader 本身不做 Resize/ToTensor。取样输出 PIL 图像和
#    bbox_condition_images，配置中的 DatasetAdapter 再生成 vae_images 和
#    box_condition_images。
# 5. 调试示例：
#    FLIR_ROOT=/path/to/flir PYTHONPATH=src \
#      python scripts/prepare/load_dataset_config.py \
#      --cfg configs/datasets/flir.yaml --loader --batches 8
