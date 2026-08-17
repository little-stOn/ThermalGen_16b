from types import SimpleNamespace

import pytest

from native16_gligen.requests import normalize_request, sanitize_filename


@pytest.fixture
def cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(width=512, height=512),
        data=SimpleNamespace(max_boxes=30),
    )


def test_normalize_request_prefers_phrases_and_normalizes_absolute_xywh(cfg: SimpleNamespace) -> None:
    boxes, phrases = normalize_request(
        {
            "boxes": [[64, 128, 128, 256]],
            "box_format": "xywh_abs",
            "phrases": ["pedestrian"],
            "labels": ["ignored"],
        },
        cfg,
    )

    assert boxes == [[0.125, 0.25, 0.375, 0.75]]
    assert phrases == ["pedestrian"]


def test_normalize_request_requires_matched_box_and_phrase(cfg: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="at least one box"):
        normalize_request({"boxes": [[0.1, 0.1, 0.2, 0.2]], "labels": []}, cfg)


def test_sanitize_filename_removes_path_syntax() -> None:
    assert sanitize_filename(" ../frame 01?.tiff ") == "frame_01_.tiff"
