from __future__ import annotations

from PIL import Image
import pytest

from dwm.datasets.common import (
    ConcatMotionDataset,
    Copy,
    DatasetAdapter,
    draw_bbox_condition,
    find_nearest,
    make_bbox_annotation,
)


class _SequenceDataset:
    def __init__(self) -> None:
        self.item = {
            "images": [["a"], ["b"], ["c"]],
            "labels": [[1], [2], [3]],
            "sample_ids": ["s0", "s1", "s2"],
            "fps": 30.0,
            "metadata": {"scene": "demo"},
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        return dict(self.item)


def test_adapter_applies_nested_transform_and_pop_list_without_torch() -> None:
    adapter = DatasetAdapter(
        _SequenceDataset(),
        transform_list=[
            {"old_key": "images", "new_key": "transformed", "transform": Copy(), "stack": False}
        ],
        pop_list=["images"],
    )

    item = adapter[0]

    assert item["transformed"] == [["a"], ["b"], ["c"]]
    assert "images" not in item
    assert item["metadata"] == {"scene": "demo"}


def test_adapter_string_index_slices_only_configured_temporal_keys() -> None:
    adapter = DatasetAdapter(_SequenceDataset(), temporal_keys={"images", "labels"})

    item = adapter["0-2-16-32"]

    assert len(item["images"]) == 2
    assert len(item["labels"]) == 2
    assert item["sample_ids"] == ["s0", "s1", "s2"]
    assert item["fps"] == 30.0


def test_concat_motion_dataset_repeats_short_sources_to_match_ratio() -> None:
    first = ["a", "b"]
    second = ["x"]
    dataset = ConcatMotionDataset([first, second], [1.0, 1.0])

    assert len(dataset) == 4
    assert [dataset[index] for index in range(len(dataset))] == ["a", "b", "x", "x"]


def test_bbox_condition_renderer_draws_absolute_xyxy_boxes() -> None:
    image = Image.new("L", (8, 6))
    box = make_bbox_annotation([1, 1, 4, 4], "vehicle.car")

    condition = draw_bbox_condition(image, [box], {"pen_width": 1})

    assert condition.mode == "RGB"
    assert condition.getpixel((1, 1)) != (0, 0, 0)
    assert condition.getpixel((0, 0)) == (0, 0, 0)
    assert find_nearest([0.0, 2.0, 5.0], 3.0) == 1
