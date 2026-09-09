"""TartanRGBT thermal style dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.tartanrgbt_common import load_tartan_records


class MotionDataset(BBoxMotionDataset):
    """Load TartanRGBT thermal sequences without fabricating annotations."""

    dataset_name = "tartanrgbt"
    annotation_source = "none"
    annotation_quality = 0.0

    def __init__(
        self,
        dataset_root: str | Path,
        camera: str = "left",
        view_mode: str = "single",
        days: Iterable[str] | None = None,
        index_cache: str | Path | None = None,
        use_index_cache: bool = True,
        max_frames_per_sequence: int | None = None,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Sequence[float]] | None = None,
        source_fps: float = 10.0,
        annotation_mode: str = "none",
        bbox_policy: str = "any",
        min_box_count: int = 1,
        return_annotations: bool = False,
        bbox_condition_settings: dict[str, Any] | None = None,
        normalize_uint16: bool = True,
        uint16_value_range: Sequence[float] = (0.0, 65535.0),
    ) -> None:
        if annotation_mode not in {"none", "optional"}:
            raise ValueError("TartanRGBT has no bbox annotations")
        records = load_tartan_records(
            dataset_root,
            camera=camera,
            view_mode=view_mode,
            days=days,
            index_cache=index_cache,
            use_index_cache=use_index_cache,
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
# 1. MotionDataset 只声明 TartanRGBT 的根目录、相机、day 筛选和 clip 参数，
#    不添加任何 bbox 兼容逻辑；数据解析集中在 tartanrgbt_common.py。
# 2. single 使用一个 thermal 视角，multiview 严格配对左右 thermal 帧；两种
#    模式都支持 style/optional annotation_mode，但数据本身不会产生 bbox。
# 3. 该 loader 服务红外风格与时序学习，condition_valid 始终为 False；图像
#    仍由 shared BBoxMotionDataset 延迟读取和统一归一化。
