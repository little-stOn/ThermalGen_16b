from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare.visualize_dataset_batch import parse_sample_id


def _batch() -> dict:
    torch = pytest.importorskip("torch")
    images = torch.full((1, 3, 1, 1, 8, 8), 0.2, dtype=torch.float32)
    conditions = torch.zeros((1, 3, 1, 3, 8, 8), dtype=torch.float32)
    conditions[:, :, :, 1, 1, 1] = 1.0
    boxes = [
        [[torch.tensor([[1.0, 1.0, 4.0, 4.0]], dtype=torch.float32)] for _ in range(3)]
    ]
    return {
        "vae_images": images,
        "box_condition_images": conditions,
        "bbox_available": torch.ones((1, 3, 1), dtype=torch.bool),
        "condition_valid": torch.ones((1, 3, 1), dtype=torch.bool),
        "pts": torch.arange(3, dtype=torch.float32).reshape(1, 3, 1),
        "boxes": boxes,
        "sample_ids": [[
            [f"demo:scene:frame_{index:04d}:thermal"] for index in range(3)
        ]],
        "dataset": ["demo"],
        "sequence": ["scene"],
    }


def test_parse_sample_id_preserves_colons_in_sequence() -> None:
    parsed = parse_sample_id("flir:video_test:thermal:000123:thermal")

    assert parsed == {
        "raw": "flir:video_test:thermal:000123:thermal",
        "dataset": "flir",
        "sequence": "video_test:thermal",
        "frame_id": "000123",
        "view": "thermal",
    }

def test_single_channel_display_uses_clear_rgb_colormap() -> None:
    torch = pytest.importorskip("torch")
    from scripts.prepare.visualize_dataset_batch import _image_from_value

    values = torch.linspace(0.0, 1.0, 64, dtype=torch.float32).reshape(1, 8, 8)
    image = _image_from_value(values, auto_contrast=True, color_map="inferno", display_scale=2)

    assert image.mode == "RGB"
    assert image.size == (16, 16)
    assert image.getpixel((0, 0)) != image.getpixel((15, 15))


def test_validate_batch_checks_temporal_and_bbox_contract() -> None:
    from scripts.prepare.visualize_dataset_batch import validate_batch

    report = validate_batch(_batch())

    assert report["shape"] == (1, 3, 1, 1, 8, 8)
    assert report["samples"][0]["bbox_frames"] == 3
    assert report["samples"][0]["bbox_count"] == 3

def test_validate_batch_accepts_style_batch_without_bbox_fields() -> None:
    from scripts.prepare.visualize_dataset_batch import validate_batch

    batch = _batch()
    torch = pytest.importorskip("torch")
    batch.pop("box_condition_images")
    batch.pop("boxes")
    batch["bbox_available"] = torch.zeros((1, 3, 1), dtype=torch.bool)
    batch["condition_valid"] = torch.zeros((1, 3, 1), dtype=torch.bool)

    report = validate_batch(batch)

    assert report["condition_present"] is False
    assert report["boxes_present"] is False
    assert report["samples"][0]["bbox_frames"] == 0


def test_validate_batch_rejects_decreasing_pts() -> None:
    from scripts.prepare.visualize_dataset_batch import validate_batch

    batch = _batch()
    torch = pytest.importorskip("torch")
    batch["pts"] = torch.tensor([[[0.0], [2.0], [1.0]]])

    with pytest.raises(ValueError, match="decreasing pts"):
        validate_batch(batch)


def test_render_sample_writes_contact_sheet_and_gif(tmp_path: Path) -> None:
    from scripts.prepare.visualize_dataset_batch import render_sample, validate_batch

    batch = _batch()
    validation = validate_batch(batch)
    files = render_sample(batch, validation, 7, 0, tmp_path, 20, True)

    contact = Path(files["contact_sheet"])
    gif = Path(files["gif"])
    assert contact.is_file()
    assert gif.is_file()
    assert contact.stat().st_size > 0
    assert gif.stat().st_size > 0


def test_inspect_batch_writes_one_contact_sheet_for_the_batch(tmp_path: Path) -> None:
    from scripts.prepare.visualize_dataset_batch import inspect_batch

    report = inspect_batch(_batch(), 7, tmp_path, max_samples_per_batch=0, save_gif=False)

    batch_contact = Path(report["batch_contact"])
    assert batch_contact.name == "batch_0007_contact.png"
    assert batch_contact.is_file()
    assert batch_contact.stat().st_size > 0
