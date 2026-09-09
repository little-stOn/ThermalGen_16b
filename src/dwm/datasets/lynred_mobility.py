"""Lynred Mobility style-only thermal dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.lynred_mobility_common import load_lynred_records


class MotionDataset(BBoxMotionDataset):
    """Load Lynred image sequences without fabricating bbox annotations."""

    dataset_name = "lynred_mobility"
    annotation_source = "none"
    annotation_quality = 0.0

    def __init__(
        self,
        dataset_root: str | Path,
        bit_depth: str = "16bits",
        resolution: str = "vga",
        sequence_names: Iterable[str] | None = None,
        environments: Iterable[str] | None = None,
        index_cache: str | Path | None = None,
        use_index_cache: bool = True,
        max_frames_per_sequence: int | None = None,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 30.0,
        annotation_mode: str = "none",
        bbox_policy: str = "any",
        min_box_count: int = 1,
        return_annotations: bool = False,
        bbox_condition_settings: dict[str, Any] | None = None,
        normalize_uint16: bool = True,
        uint16_value_range: Sequence[float] = (0.0, 65535.0),
    ) -> None:
        if annotation_mode not in {"none", "optional"}:
            raise ValueError("Lynred Mobility has no bbox annotations")
        records = load_lynred_records(
            dataset_root,
            bit_depth=bit_depth,
            resolution=resolution,
            sequence_names=sequence_names,
            environments=environments,
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
            return_annotations=return_annotations,
            bbox_condition_settings=bbox_condition_settings,
            normalize_uint16=normalize_uint16,
            uint16_value_range=uint16_value_range,
        )


# 文件讲解：
# 1. MotionDataset 使用 metadata.csv 选择 Lynred 的 bit depth、分辨率和环境，
#    不扫描或伪造 bbox；它与其他 dataset.py 一样只负责参数校验和基类组装。
# 2. annotation_mode=none 用于 style，optional 用于 joint 的统一 schema；两种
#    模式都会将 bbox_available/condition_valid 置为 False。
# 3. 该 loader 适合学习传感器响应、场景环境和红外纹理，不应进入 bbox-only
#    配置。图像仍在 __getitem__ 时延迟读取。
# 4. 调试：LYNRED_ROOT=/path/to/range_dataset PYTHONPATH=src python
#    scripts/prepare/load_dataset_config.py --cfg configs/datasets/multi_style.yaml
#    --loader --batches 8
