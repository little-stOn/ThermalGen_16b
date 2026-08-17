"""Request normalization and visual diagnostics shared by Native16 generation."""

from __future__ import annotations

import re
from typing import Any

from PIL import Image, ImageDraw

from native16_gligen.data import convert_boxes_to_normalized_xyxy


def sanitize_filename(value: str) -> str:
    """Return a deterministic filename stem without path semantics."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return (cleaned or "sample")[:120]


def draw_box_overlay(
    image: Image.Image,
    boxes: list[list[float]],
    phrases: list[str],
) -> Image.Image:
    """Render normalized request boxes for human inspection only."""

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        rectangle = (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
        draw.rectangle(rectangle, outline=(255, 0, 0), width=2)
        draw.text(
            (rectangle[0] + 3, max(0, rectangle[1] - 12)),
            phrases[index],
            fill=(255, 0, 0),
        )
    return overlay


def normalize_request(request: dict[str, Any], cfg: Any) -> tuple[list[list[float]], list[str]]:
    """Validate one GLIGEN request and normalize boxes to ``xyxy`` in ``[0, 1]``."""

    tensor = convert_boxes_to_normalized_xyxy(
        request.get("boxes", []),
        request.get("box_format", "xyxy_norm"),
        int(cfg.model.width),
        int(cfg.model.height),
    )
    labels = request.get("phrases", request.get("labels", []))
    count = min(len(tensor), len(labels), int(cfg.data.max_boxes), 30)
    if count == 0:
        raise ValueError("GLIGEN generation requires at least one box and matching phrase")
    return tensor[:count].tolist(), [str(value) for value in labels[:count]]
