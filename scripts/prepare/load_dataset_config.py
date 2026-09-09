#!/usr/bin/env python3
"""Load a configured dataset, inspect one sample, and measure batch speed."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from numbers import Number
from pathlib import Path
import time
from typing import Any

from dwm.common import (
    load_dataloader_from_config,
    load_object_from_config,
    load_task_dataloader,
)


def summarize(value: Any) -> Any:
    """Return a compact, print-safe description of nested sample data."""

    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first": summarize(value[0]) if value else None,
        }
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": tuple(int(item) for item in shape)}
    size = getattr(value, "size", None)
    if isinstance(size, tuple):
        return {"type": type(value).__name__, "size": tuple(int(item) for item in size)}
    return type(value).__name__ if value is not None else None


def count_boxes(value: Any) -> int:
    """Count absolute ``xyxy`` leaves in native annotation trees."""

    if isinstance(value, dict):
        return sum(count_boxes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, Number) for item in value):
            return 1
        return sum(count_boxes(item) for item in value)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) == 2 and int(shape[-1]) == 4:
        return int(shape[0])
    return 0

def infer_batch_size(batch: Any) -> int:
    """Infer the number of samples actually consumed from a collated batch."""

    if isinstance(batch, dict):
        preferred_keys = ("vae_images", "images", "pixel_values", "dataset")
        values = [batch[key] for key in preferred_keys if key in batch]
        values.extend(value for key, value in batch.items() if key not in preferred_keys)
    else:
        values = [batch]
    for value in values:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        if isinstance(value, (list, tuple)) and value:
            return len(value)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, help="dataset YAML/JSON config")
    parser.add_argument("--task", choices=("bbox", "style", "joint"), help="load multi_<task>.yaml")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/datasets"),
        help="directory containing multi task configs",
    )
    parser.add_argument("--key", default="dataset", help="dataset object key")
    parser.add_argument("--loader", action="store_true", help="construct the configured DataLoader")
    parser.add_argument("--index", type=int, default=0, help="integer sample index")
    parser.add_argument(
        "--dynamic-index",
        default=None,
        help="optional adapter index in idx-num_frame-height-width form",
    )
    parser.add_argument("--batches", type=int, default=0, help="number of batches to time")
    parser.add_argument("--set", nargs="*", default=[], help="dotted key=value overrides")
    args = parser.parse_args(argv)
    if args.cfg is None and args.task is None:
        parser.error("one of --cfg or --task is required")
    if args.cfg is not None and args.task is not None:
        parser.error("--cfg and --task cannot be used together")
    if args.batches < 0:
        raise SystemExit("--batches must be non-negative")

    construction_start = time.perf_counter()
    loader = None
    if args.task is not None:
        loader, _ = load_task_dataloader(
            args.task,
            config_dir=args.config_dir,
            overrides=args.set,
        )
        dataset = loader.dataset
        config_path = args.config_dir / {"bbox": "multi_bbox.yaml", "style": "multi_style.yaml", "joint": "multi_joint.yaml"}[args.task]
    elif args.loader:
        loader, _ = load_dataloader_from_config(args.cfg, overrides=args.set)
        dataset = loader.dataset
        config_path = args.cfg
    else:
        dataset, _ = load_object_from_config(args.cfg, args.key, args.set)
        config_path = args.cfg
    construction_seconds = time.perf_counter() - construction_start

    index: int | str = args.dynamic_index if args.dynamic_index is not None else args.index
    if isinstance(index, int) and (index < 0 or index >= len(dataset)):
        raise SystemExit(f"index {index} outside dataset length {len(dataset)}")
    sample_start = time.perf_counter()
    sample = dataset[index]
    sample_seconds = time.perf_counter() - sample_start
    report = {
        "config": str(config_path.resolve()),
        "length": len(dataset),
        "construction_seconds": round(construction_seconds, 4),
        "sample_index": index,
        "sample_seconds": round(sample_seconds, 4),
        "bbox_count": count_boxes(sample.get("boxes", [])) if isinstance(sample, dict) else 0,
        "sample": summarize(sample),
    }
    print(report)

    if loader is not None and args.batches:
        batch_start = time.perf_counter()
        iterator = iter(loader)
        sample_count = 0
        for _ in range(args.batches):
            sample_count += infer_batch_size(next(iterator))
        elapsed = time.perf_counter() - batch_start
        print(
            {
                "batches": args.batches,
                "samples": sample_count,
                "batch_seconds": round(elapsed, 4),
                "samples_per_second": round(sample_count / max(elapsed, 1e-9), 4),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# 文件讲解：
# 1. 该脚本不实现任何数据集逻辑，只调用 dwm.common 的配置工厂；因此它
#    是验证“配置 -> loader -> sample/batch”链路的最小演示入口。
# 2. --cfg 用于任意单数据集或组合配置；--task=bbox/style/joint 会自动选择
#    configs/datasets/multi_<task>.yaml，训练和调试可以使用同一选择方式。
# 3. --loader 会把 dataset 注入配置中的 DataLoader；--batches 会统计连续
#    batch 吞吐，--dynamic-index 会验证 Adapter 的随机时间片和动态尺寸。
# 4. --set 使用 dotted key=value 覆盖，不改 YAML；例如 --set
#    dataset.base_dataset.sequence_length=8 dataloader.num_workers=2。
# 5. 输出中的 length、bbox_count、sample_seconds 和 samples_per_second 分别
#    证明数据索引、bbox 读取、单样本访问和 DataLoader 批处理均已实际执行。
