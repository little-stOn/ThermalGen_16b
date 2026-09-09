"""TartanRGBT thermal sequence indexing."""

from __future__ import annotations

from hashlib import sha1
import math
from pathlib import Path
import pickle
import re
from typing import Iterable

from dwm.datasets.common import BBoxFrameRecord, BBoxViewRecord, limit_sequence


_FRAME_RE = re.compile(r"^(\d+)$")


def resolve_tartan_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "tartanrgbt" and root.is_dir():
        return root
    nested = root / "tartanrgbt"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"TartanRGBT root does not exist: {root}")


def _frame_key(path: Path) -> tuple[int, str]:
    match = _FRAME_RE.match(path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def _timestamps(path: Path, count: int) -> list[float | None]:
    if not path.is_file():
        return [None] * count
    values = [float(line.strip()) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(values) != count:
        raise ValueError(f"TartanRGBT timestamp/image count mismatch: {path}: {len(values)} != {count}")
    return values

def _default_index_cache(root: Path, key: str) -> Path:
    digest = sha1(key.encode("utf-8")).hexdigest()[:16]
    return root / ".dwm_cache" / f"tartan_{digest}.pkl"


def load_tartan_records(
    dataset_root: str | Path,
    camera: str = "left",
    view_mode: str = "single",
    days: Iterable[str] | None = None,
    index_cache: str | Path | None = None,
    use_index_cache: bool = True,
    max_frames_per_sequence: int | None = None,
    source_fps: float = 10.0,
) -> tuple[BBoxFrameRecord, ...]:
    root = resolve_tartan_root(dataset_root)
    camera = str(camera).lower()
    if camera not in {"left", "right"}:
        raise ValueError("TartanRGBT camera must be left or right")
    if view_mode not in {"single", "multiview"}:
        raise ValueError("TartanRGBT view_mode must be single or multiview")
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    selected_days = None if days is None else tuple(sorted(str(value) for value in days))
    cache_key = repr(
        (
            "tartan-index-v3",
            str(root),
            camera,
            view_mode,
            selected_days,
            max_frames_per_sequence,
            source_fps,
        )
    )
    selected_days_set = None if selected_days is None else set(selected_days)
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

    left_dirs = sorted(root.rglob("thermal_left_rect_16"))
    records: list[BBoxFrameRecord] = []
    for left_root in left_dirs:
        if not left_root.is_dir():
            continue
        day = left_root.relative_to(root).parts[0]
        if selected_days_set is not None and day not in selected_days_set:
            continue
        run_root = left_root.parent
        left_images = sorted(
            [
                path
                for path in left_root.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            ],
            key=_frame_key,
        )
        if not left_images:
            continue
        left_times = _timestamps(left_root / "timestamps.txt", len(left_images))
        selected_frames = limit_sequence(
            tuple(enumerate(zip(left_images, left_times, strict=True))),
            max_frames_per_sequence,
        )
        right_root = run_root / "thermal_right_rect_16"
        right_by_id = (
            {
                path.stem: path
                for path in right_root.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            }
            if right_root.is_dir()
            else {}
        )
        if view_mode == "multiview" and len(right_by_id) != len(left_images):
            raise ValueError(f"TartanRGBT stereo frame mismatch in {run_root}")
        sequence = str(run_root.relative_to(root))
        for original_index, (left_path, timestamp) in selected_frames:
            if view_mode == "multiview":
                right_path = right_by_id.get(left_path.stem)
                if right_path is None:
                    raise ValueError(f"TartanRGBT missing right frame for {left_path}")
                views = (
                    BBoxViewRecord("thermal_left", left_path, (), annotations_available=False),
                    BBoxViewRecord("thermal_right", right_path, (), annotations_available=False),
                )
            else:
                path = left_path if camera == "left" else right_by_id.get(left_path.stem)
                if path is None:
                    continue
                views = (BBoxViewRecord(f"thermal_{camera}", path, (), annotations_available=False),)
            records.append(
                BBoxFrameRecord(
                    sequence=sequence,
                    frame_id=left_path.stem,
                    views=views,
                    timestamp=(
                        original_index / source_fps
                        if timestamp is None
                        else timestamp
                    ),
                    metadata={"day": day, "run": sequence, "camera": camera},
                )
            )
    if not records:
        raise ValueError("TartanRGBT selection produced no image records")
    result = tuple(records)
    if use_index_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary_path.open("wb") as handle:
                pickle.dump(
                    {"key": cache_key, "records": result},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary_path.replace(cache_path)
        except OSError:
            pass
    return result


# 文件讲解：
# 1. TartanRGBT 的有效输入是 day/run 下的 thermal_left_rect_16 和
#    thermal_right_rect_16 PNG 序列；timestamps.txt 仅提供时间轴，不是 bbox。
# 2. single 模式选择一个 thermal 相机，multiview 模式按数字帧名严格配对
#    左右相机；帧数或帧名不一致会直接报错。
# 3. 本数据集没有 bbox，因此 annotations_available=False；它只进入 style
#    或 joint 的无标注分支，图像读取和 clip 构造仍复用 shared base。
