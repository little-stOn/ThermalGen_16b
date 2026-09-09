"""Lynred Mobility image-sequence indexing."""

from __future__ import annotations

import csv
from hashlib import sha1
import math
from pathlib import Path
import pickle
import re
from typing import Iterable

from dwm.datasets.common import BBoxFrameRecord, BBoxViewRecord, limit_sequence


_FRAME_RE = re.compile(r"(\d+)$")


def resolve_lynred_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "range_dataset" and root.is_dir():
        return root
    nested = root / "range_dataset"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"Lynred Mobility root does not exist: {root}")


def _frame_key(path: Path) -> tuple[int, str]:
    match = _FRAME_RE.search(path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def _default_index_cache(root: Path, key: str) -> Path:
    digest = sha1(key.encode("utf-8")).hexdigest()[:16]
    return root / ".dwm_cache" / f"lynred_{digest}.pkl"


def load_lynred_records(
    dataset_root: str | Path,
    bit_depth: str = "16bits",
    resolution: str = "vga",
    sequence_names: Iterable[str] | None = None,
    environments: Iterable[str] | None = None,
    index_cache: str | Path | None = None,
    use_index_cache: bool = True,
    max_frames_per_sequence: int | None = None,
    source_fps: float = 30.0,
) -> tuple[BBoxFrameRecord, ...]:
    root = resolve_lynred_root(dataset_root)
    bit_depth = str(bit_depth).lower()
    resolution = str(resolution).lower()
    if bit_depth not in {"8bits", "16bits"}:
        raise ValueError("Lynred bit_depth must be 8bits or 16bits")
    if resolution not in {"qvga", "vga"}:
        raise ValueError("Lynred resolution must be qvga or vga")
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    metadata_path = root / "metadata" / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Lynred metadata file does not exist: {metadata_path}")
    selected_sequences = None if sequence_names is None else tuple(sorted(str(value) for value in sequence_names))
    selected_environments = None if environments is None else tuple(sorted(str(value) for value in environments))
    cache_key = repr(
        (
            "lynred-index-v2",
            str(root),
            str(metadata_path),
            metadata_path.stat().st_size,
            metadata_path.stat().st_mtime_ns,
            bit_depth,
            resolution,
            selected_sequences,
            selected_environments,
            max_frames_per_sequence,
            source_fps,
        )
    )
    cache_path = (
        _default_index_cache(root, cache_key)
        if index_cache is None
        else Path(index_cache).expanduser().resolve()
    )
    if use_index_cache:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("key") == cache_key and all(
                view.path.is_file()
                for record in payload["records"]
                for view in record.views
            ):
                return tuple(payload["records"])
        except (OSError, EOFError, KeyError, AttributeError, pickle.UnpicklingError):
            pass

    sequence_set = None if selected_sequences is None else set(selected_sequences)
    environment_set = None if selected_environments is None else set(selected_environments)
    records: list[BBoxFrameRecord] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            raw_relative = Path(str(row.get(f"{resolution}_path", "")).strip())
            if not raw_relative.parts:
                continue
            relative = (
                Path(*raw_relative.parts[1:])
                if raw_relative.parts[0].lower() == resolution
                else raw_relative
            )
            environment = str(row.get("environment", ""))
            if environment_set is not None and environment not in environment_set:
                continue
            sequence = f"{bit_depth}/{resolution}/{relative}"
            if sequence_set is not None and sequence not in sequence_set:
                continue
            sequence_root = root / bit_depth / resolution / relative
            if not sequence_root.is_dir():
                raise FileNotFoundError(f"Lynred sequence directory does not exist: {sequence_root}")
            images = sorted(
                [
                    path
                    for path in sequence_root.iterdir()
                    if path.is_file() and path.suffix.lower() == ".png"
                ],
                key=_frame_key,
            )
            if not images:
                continue
            selected_frames = limit_sequence(
                tuple(enumerate(images)),
                max_frames_per_sequence,
            )
            for index, image_path in selected_frames:
                records.append(
                    BBoxFrameRecord(
                        sequence=sequence,
                        frame_id=image_path.stem,
                        timestamp=index / source_fps,
                        views=(
                            BBoxViewRecord(
                                "thermal",
                                image_path,
                                (),
                                annotations_available=False,
                            ),
                        ),
                        metadata={
                            "environment": environment,
                            "distance_min": row.get("distance_min"),
                            "distance_max": row.get("distance_max"),
                            "time_of_day": row.get("time_of_day"),
                            "bit_depth": bit_depth,
                            "resolution": resolution,
                            "ordinal": index,
                        },
                    )
                )
    if not records:
        raise ValueError("Lynred selection produced no image records")
    result = tuple(records)
    if use_index_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary_path.open("wb") as handle:
                pickle.dump({"key": cache_key, "records": result}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path.replace(cache_path)
        except OSError:
            pass
    return result


# 文件讲解：
# 1. metadata.csv 是 Lynred 的轻量索引，提供 qvga/vga 的序列相对路径和
#    环境、距离、时间等属性；解析器使用它，避免遍历数十万张图像来找序列。
# 2. 本数据集没有 bbox 标注，所以每个 BBoxViewRecord 明确标记
#    annotations_available=False；annotation_mode=optional 可在 joint 配置中
#    输出无效 condition，annotation_mode=none 则省略 condition 以节省开销。
# 3. 图像仍然由 shared BBoxMotionDataset 延迟读取，序列边界、clip 和 batch
#    行为与带框数据集完全相同。
