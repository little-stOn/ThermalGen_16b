from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

from native16_gligen.irzoom import choose_instance_index, make_scale_normalized_crop
from native16_gligen.thermal16 import RadiometricProfile, load_native16_tensor, load_thermal16_tensor


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def read_json_or_jsonl(path: str | Path) -> Any:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {input_path}")
    if input_path.suffix.lower() == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {input_path}") from exc
        return records
    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.load()
    return image


def convert_boxes_to_absolute_xyxy(
    boxes: list[list[float]], box_format: str, width: int, height: int
) -> torch.Tensor:
    tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if tensor.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    if box_format == "xyxy_abs":
        pass
    elif box_format == "xywh_abs":
        tensor[:, 2] += tensor[:, 0]
        tensor[:, 3] += tensor[:, 1]
    elif box_format == "xywh_norm":
        tensor[:, 2] += tensor[:, 0]
        tensor[:, 3] += tensor[:, 1]
        tensor[:, [0, 2]] *= max(float(width), 1.0)
        tensor[:, [1, 3]] *= max(float(height), 1.0)
    elif box_format == "xyxy_norm":
        tensor[:, [0, 2]] *= max(float(width), 1.0)
        tensor[:, [1, 3]] *= max(float(height), 1.0)
    else:
        raise ValueError(f"Unsupported box format: {box_format}")
    tensor[:, [0, 2]] = tensor[:, [0, 2]].clamp(0.0, float(width))
    tensor[:, [1, 3]] = tensor[:, [1, 3]].clamp(0.0, float(height))
    x1 = torch.minimum(tensor[:, 0], tensor[:, 2])
    y1 = torch.minimum(tensor[:, 1], tensor[:, 3])
    x2 = torch.maximum(tensor[:, 0], tensor[:, 2])
    y2 = torch.maximum(tensor[:, 1], tensor[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def normalize_absolute_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    normalized = boxes.clone().to(torch.float32)
    if normalized.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    normalized[:, [0, 2]] /= max(float(width), 1.0)
    normalized[:, [1, 3]] /= max(float(height), 1.0)
    return normalized.clamp(0.0, 1.0)


def convert_boxes_to_normalized_xyxy(
    boxes: list[list[float]], box_format: str, width: int, height: int
) -> torch.Tensor:
    absolute = convert_boxes_to_absolute_xyxy(boxes, box_format, width, height)
    return normalize_absolute_xyxy(absolute, width, height)


def crop_image_and_boxes(
    image: Image.Image | torch.Tensor,
    boxes: torch.Tensor,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    min_visible_fraction: float,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    width, height = image_size(image)
    if crop_width < 1 or crop_height < 1:
        raise ValueError("Crop dimensions must be positive")
    if left < 0 or top < 0 or left + crop_width > width or top + crop_height > height:
        raise ValueError("Crop lies outside the image")
    if torch.is_tensor(image):
        cropped = image[:, top : top + crop_height, left : left + crop_width]
    else:
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
    if boxes.numel() == 0:
        return cropped, torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.bool)
    original_areas = ((boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0))
    transformed = boxes.clone()
    transformed[:, [0, 2]] -= float(left)
    transformed[:, [1, 3]] -= float(top)
    transformed[:, [0, 2]] = transformed[:, [0, 2]].clamp(0.0, float(crop_width))
    transformed[:, [1, 3]] = transformed[:, [1, 3]].clamp(0.0, float(crop_height))
    visible_areas = (
        (transformed[:, 2] - transformed[:, 0]).clamp_min(0)
        * (transformed[:, 3] - transformed[:, 1]).clamp_min(0)
    )
    visible_fraction = visible_areas / original_areas.clamp_min(1e-8)
    keep = (visible_areas > 0) & (visible_fraction >= float(min_visible_fraction))
    return cropped, transformed[keep], keep


def choose_square_crop(
    width: int,
    height: int,
    boxes: torch.Tensor,
    random_crop: bool,
    bbox_focus_probability: float,
) -> tuple[int, int, int]:
    crop_size = min(width, height)
    max_left = width - crop_size
    max_top = height - crop_size
    if not random_crop:
        return max_left // 2, max_top // 2, crop_size
    if boxes.numel() > 0 and random.random() < float(bbox_focus_probability):
        chosen = boxes[random.randrange(int(boxes.shape[0]))]
        center_x = float((chosen[0] + chosen[2]) * 0.5)
        center_y = float((chosen[1] + chosen[3]) * 0.5)
        left = round(center_x - random.uniform(0.25, 0.75) * crop_size)
        top = round(center_y - random.uniform(0.25, 0.75) * crop_size)
        return min(max(left, 0), max_left), min(max(top, 0), max_top), crop_size
    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0
    return left, top, crop_size


def parse_box_records(raw_boxes: list[Any], raw_labels: list[Any] | None) -> tuple[list[list[float]], list[Any]]:
    boxes: list[list[float]] = []
    labels: list[Any] = []
    if raw_boxes and isinstance(raw_boxes[0], dict):
        for item in raw_boxes:
            coordinates = item.get("bbox", item.get("box"))
            if coordinates is None or len(coordinates) != 4:
                raise ValueError("Every box dictionary needs a four-value bbox or box field")
            boxes.append([float(value) for value in coordinates])
            labels.append(item.get("label", item.get("category", "unknown")))
    else:
        for coordinates in raw_boxes:
            if len(coordinates) != 4:
                raise ValueError("Every box needs four coordinates")
            boxes.append([float(value) for value in coordinates])
        labels = list(raw_labels or ["unknown"] * len(boxes))
    if len(labels) != len(boxes):
        raise ValueError("The numbers of labels and boxes must match")
    return boxes, labels


def record_has_instance_target(
    record: dict[str, Any], cfg: Any, allowed_names: set[str]
) -> bool:
    labels = [str(label).lower() for label in record.get("labels", [])]
    candidate_indices = [
        index for index, label in enumerate(labels) if not allowed_names or label in allowed_names
    ]
    if not candidate_indices:
        return False
    if not (
        str(record.get("dataset", "unknown")) == "DroneVehicle"
        and bool(getattr(cfg, "crop_dronevehicle_border", False))
    ):
        return True
    width = int(record.get("width", 0))
    height = int(record.get("height", 0))
    border = int(getattr(cfg, "dronevehicle_border", 100))
    if width <= 2 * border or height <= 2 * border:
        return False
    boxes = convert_boxes_to_absolute_xyxy(
        record.get("boxes", []), record.get("box_format", "xyxy_abs"), width, height
    )
    minimum_fraction = float(getattr(cfg, "min_visible_fraction", 0.3))
    for index in candidate_indices:
        box = boxes[index]
        area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-8)
        clipped_width = max(0.0, min(float(width - border), float(box[2])) - max(float(border), float(box[0])))
        clipped_height = max(0.0, min(float(height - border), float(box[3])) - max(float(border), float(box[1])))
        if clipped_width * clipped_height / area >= minimum_fraction:
            return True
    return False


def image_size(image: Image.Image | torch.Tensor) -> tuple[int, int]:
    if torch.is_tensor(image):
        if image.ndim != 3:
            raise ValueError(f"Expected CxHxW image tensor, got {tuple(image.shape)}")
        return int(image.shape[-1]), int(image.shape[-2])
    return image.size


def prepare_target_tensor(
    image: Image.Image | torch.Tensor, height: int, width: int, expected_channels: int = 3
) -> torch.Tensor:
    if torch.is_tensor(image):
        tensor = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    else:
        resized = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        tensor = TF.to_tensor(resized)
    tensor = tensor.mul(2.0).sub(1.0)
    if tensor.shape != (expected_channels, height, width):
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(tensor.shape)}")
    return tensor


def prepare_reference_tensor(image: Image.Image | None, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if image is None:
        tensor = torch.zeros((3, size, size), dtype=torch.float32)
        present = torch.tensor(False, dtype=torch.bool)
        return tensor, present
    resized = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, CLIP_MEAN, CLIP_STD)
    present = torch.tensor(True, dtype=torch.bool)
    return tensor, present


def pad_boxes_and_labels(
    boxes: torch.Tensor, labels: list[int], max_boxes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
    padded_labels = torch.zeros((max_boxes,), dtype=torch.long)
    padded_mask = torch.zeros((max_boxes,), dtype=torch.bool)
    valid_count = min(int(boxes.shape[0]), max_boxes, len(labels))
    if valid_count > 0:
        padded_boxes[:valid_count] = boxes[:valid_count]
        padded_labels[:valid_count] = torch.tensor(labels[:valid_count], dtype=torch.long)
        padded_mask[:valid_count] = True
    return padded_boxes, padded_labels, padded_mask


class BoxDatasetBase(Dataset):
    def __init__(self, cfg: Any, records: list[dict[str, Any]], class_to_idx: dict[str, int]):
        self.cfg = cfg
        self.records = records
        self.root = Path(cfg.root)
        self.height = int(cfg.height)
        self.width = int(cfg.width)
        self.max_boxes = int(cfg.max_boxes)
        self.random_flip = bool(cfg.random_flip)
        self.ref_size = int(cfg.ref_size)
        self.input_mode = str(getattr(cfg, "input_mode", "rgb8")).lower()
        if self.input_mode not in {"rgb8", "thermal16", "native16", "native16_bridge"}:
            raise ValueError(f"Unsupported data.input_mode: {self.input_mode}")
        self.thermal16_field = "thermal16"
        self.thermal16_profile = RadiometricProfile()
        if self.input_mode in {"thermal16", "native16", "native16_bridge"}:
            thermal_cfg = getattr(cfg, "thermal16", None)
            self.thermal16_field = str(getattr(thermal_cfg, "field", "thermal16"))
            profile_value = getattr(thermal_cfg, "profile", None)
            profile_path = getattr(thermal_cfg, "profile_path", None)
            if profile_path not in (None, "", "null"):
                profile_value = Path(str(profile_path))
                if not profile_value.is_absolute():
                    profile_value = self.root / profile_value
            self.thermal16_profile = RadiometricProfile.from_config(profile_value)
            self.thermal16_profile.validate()
        self.class_to_idx = dict(class_to_idx)
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        self.instance_crop_cfg = getattr(cfg, "instance_crop", None)
        self.instance_crop_enabled = bool(
            self.instance_crop_cfg is not None
            and getattr(self.instance_crop_cfg, "enabled", False)
        )
        if not self.records:
            raise ValueError("Dataset contains no usable samples")

    def __len__(self) -> int:
        length = len(self.records)
        if length < 1:
            raise RuntimeError("Dataset became empty")
        return length

    def resolve_label(self, label: Any) -> int:
        text = str(label)
        if text in self.class_to_idx:
            return int(self.class_to_idx[text])
        lowered = text.lower()
        if lowered in self.class_to_idx:
            return int(self.class_to_idx[lowered])
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.input_mode in {"thermal16", "native16", "native16_bridge"}:
            raw_value = record.get(self.thermal16_field)
            if not raw_value:
                raise ValueError(
                    f"Record {record.get('id', index)} has no {self.thermal16_field} path"
                )
            image_path = Path(str(raw_value))
            if not image_path.is_absolute():
                image_path = self.root / image_path
            if self.input_mode == "native16":
                image = load_native16_tensor(image_path, self.thermal16_profile)
            else:
                image = load_thermal16_tensor(image_path, self.thermal16_profile)
        else:
            image_path = self.root / record["image"]
            image = load_rgb_image(image_path)
        original_width, original_height = image_size(image)
        boxes = convert_boxes_to_absolute_xyxy(
            record.get("boxes", []),
            record.get("box_format", "xyxy_abs"),
            original_width,
            original_height,
        )
        raw_labels = list(record.get("labels", []))
        labels = [self.resolve_label(label) for label in raw_labels]
        phrase_aliases = {
            str(key).lower(): str(value)
            for key, value in dict(getattr(self.cfg, "phrase_aliases", {})).items()
        }
        phrases = [phrase_aliases.get(str(label).lower(), str(label)) for label in raw_labels]
        instance_source_side = 0.0
        instance_normalized_side = 0.0
        instance_crop_xyxy = torch.zeros(4, dtype=torch.long)

        def apply_keep(keep: torch.Tensor) -> None:
            nonlocal labels, phrases
            flags = keep.tolist()
            labels = [value for value, valid in zip(labels, flags) if valid]
            phrases = [value for value, valid in zip(phrases, flags) if valid]

        dataset_name = str(record.get("dataset", "unknown"))
        if dataset_name == "DroneVehicle" and bool(getattr(self.cfg, "crop_dronevehicle_border", False)):
            border = int(getattr(self.cfg, "dronevehicle_border", 100))
            width, height = image_size(image)
            if width <= 2 * border or height <= 2 * border:
                raise ValueError(f"DroneVehicle border {border} is invalid for image {image_path} size={width,height}")
            image, boxes, keep = crop_image_and_boxes(
                image,
                boxes,
                border,
                border,
                width - 2 * border,
                height - 2 * border,
                float(getattr(self.cfg, "min_visible_fraction", 0.3)),
            )
            apply_keep(keep)

        if self.instance_crop_enabled:
            if torch.is_tensor(image):
                raise ValueError("data.instance_crop is not supported with lossless thermal16 tensors")
            if boxes.shape[0] == 0:
                raise RuntimeError(f"IR-Zoom record has no boxes after preprocessing: {image_path}")
            allowed_names = {
                str(value).lower()
                for value in getattr(self.instance_crop_cfg, "allowed_phrases", [])
            }
            allowed = torch.tensor(
                [not allowed_names or phrase.lower() in allowed_names for phrase in phrases],
                dtype=torch.bool,
            )
            if not bool(allowed.any()):
                raise RuntimeError(f"IR-Zoom record has no allowed target after preprocessing: {image_path}")
            width, height = image_size(image)
            target_index = choose_instance_index(
                boxes,
                width,
                height,
                allowed,
                small_reference=float(getattr(self.instance_crop_cfg, "small_reference", 32.0)),
                small_weight=float(getattr(self.instance_crop_cfg, "small_weight", 4.0)),
                medium_weight=float(getattr(self.instance_crop_cfg, "medium_weight", 1.0)),
                large_weight=float(getattr(self.instance_crop_cfg, "large_weight", 0.25)),
            )
            crop = make_scale_normalized_crop(
                image,
                boxes[target_index],
                output_size=self.width,
                target_long_side_min=float(
                    getattr(self.instance_crop_cfg, "target_long_side_min", 112.0)
                ),
                target_long_side_max=float(
                    getattr(self.instance_crop_cfg, "target_long_side_max", 160.0)
                ),
                min_context_scale=float(
                    getattr(self.instance_crop_cfg, "min_context_scale", 2.5)
                ),
                center_jitter=float(getattr(self.instance_crop_cfg, "center_jitter", 0.12)),
            )
            image = crop.image
            boxes = crop.box_xyxy.unsqueeze(0) * torch.tensor(
                [self.width, self.height, self.width, self.height], dtype=torch.float32
            )
            labels = [labels[target_index]]
            phrases = [phrases[target_index]]
            instance_source_side = crop.source_side_px
            instance_normalized_side = crop.normalized_side_px
            instance_crop_xyxy = torch.tensor(crop.crop_xyxy, dtype=torch.long)

        elif bool(getattr(self.cfg, "square_crop", False)):
            width, height = image_size(image)
            left, top, crop_size = choose_square_crop(
                width,
                height,
                boxes,
                bool(getattr(self.cfg, "random_crop", False)),
                float(getattr(self.cfg, "bbox_focus_probability", 0.7)),
            )
            image, boxes, keep = crop_image_and_boxes(
                image,
                boxes,
                left,
                top,
                crop_size,
                crop_size,
                float(getattr(self.cfg, "min_visible_fraction", 0.3)),
            )
            apply_keep(keep)

        if self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            if boxes.numel() > 0:
                width, _ = image_size(image)
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = float(width) - old_x2
                boxes[:, 2] = float(width) - old_x1
        width, height = image_size(image)
        boxes = normalize_absolute_xyxy(boxes, width, height)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas > float(getattr(self.cfg, "min_box_area", 1e-5))
        boxes = boxes[keep]
        apply_keep(keep)
        padded_boxes, padded_labels, padded_mask = pad_boxes_and_labels(boxes, labels, self.max_boxes)
        padded_phrases = phrases[: self.max_boxes] + [""] * max(0, self.max_boxes - len(phrases))
        expected_channels = 1 if self.input_mode == "native16" else 3
        target = prepare_target_tensor(
            image, self.height, self.width, expected_channels=expected_channels
        )
        reference_path = record.get("reference")
        reference_image = load_rgb_image(self.root / reference_path) if reference_path else None
        reference, reference_present = prepare_reference_tensor(reference_image, self.ref_size)
        if bool(getattr(self.cfg, "prompt_from_labels", False)):
            unique_phrases = sorted(set(phrases))
            prefix = str(getattr(self.cfg, "prompt_prefix", "an image containing"))
            prompt = prefix + (" " + ", ".join(unique_phrases) if unique_phrases else "")
        else:
            prompt = str(record.get("prompt", ""))
        if self.instance_crop_enabled:
            prompt_template = str(
                getattr(
                    self.instance_crop_cfg,
                    "prompt_template",
                    "an infrared aerial image containing {phrase}",
                )
            )
            prompt = prompt_template.format(phrase=phrases[0])
        return {
            "pixel_values": target,
            "boxes": padded_boxes,
            "labels": padded_labels,
            "box_mask": padded_mask,
            "phrases": padded_phrases,
            "reference_pixel_values": reference,
            "reference_present": reference_present,
            "prompt": prompt,
            "image_path": str(image_path),
            "dataset": dataset_name,
            "sample_id": str(record.get("id", index)),
            "instance_source_side": torch.tensor(instance_source_side, dtype=torch.float32),
            "instance_normalized_side": torch.tensor(instance_normalized_side, dtype=torch.float32),
            "instance_crop_xyxy": instance_crop_xyxy,
        }


class CustomBoxDataset(BoxDatasetBase):
    def __init__(self, cfg: Any):
        raw = read_json_or_jsonl(cfg.annotation_file)
        if isinstance(raw, dict):
            raw_records = raw.get("samples", raw.get("records", []))
        else:
            raw_records = raw
        if not isinstance(raw_records, list):
            raise ValueError("Custom annotation root must be a list or contain samples/records")
        records: list[dict[str, Any]] = []
        discovered: set[str] = set()
        for raw_record in raw_records:
            boxes, labels = parse_box_records(raw_record.get("boxes", []), raw_record.get("labels"))
            record = dict(raw_record)
            record["boxes"] = boxes
            record["labels"] = labels
            records.append(record)
            for label in labels:
                discovered.add(str(label).lower())
        if str(getattr(cfg, "input_mode", "rgb8")).lower() in {"thermal16", "native16", "native16_bridge"}:
            field = str(getattr(getattr(cfg, "thermal16", None), "field", "thermal16"))
            before = len(records)
            records = [record for record in records if record.get(field)]
            if not records:
                raise ValueError(f"No records contain data.thermal16.field={field}")
            if bool(getattr(getattr(cfg, "thermal16", None), "require_all", False)) and len(records) != before:
                raise ValueError(
                    f"thermal16 mode requires every record to contain {field}; "
                    f"missing={before - len(records)}"
                )
        instance_crop = getattr(cfg, "instance_crop", None)
        if instance_crop is not None and bool(getattr(instance_crop, "enabled", False)):
            allowed = {str(value).lower() for value in getattr(instance_crop, "allowed_phrases", [])}
            if allowed:
                records = [
                    record
                    for record in records
                    if record_has_instance_target(record, cfg, allowed)
                ]
        configured_names = list(getattr(cfg, "class_names", []))
        names = [str(name).lower() for name in configured_names] if configured_names else sorted(discovered)
        class_to_idx = {name: position + 1 for position, name in enumerate(names)}
        super().__init__(cfg, records, class_to_idx)


class CocoBoxDataset(BoxDatasetBase):
    def __init__(self, cfg: Any):
        coco = read_json_or_jsonl(cfg.annotation_file)
        if not isinstance(coco, dict):
            raise ValueError("COCO annotations must use the standard dictionary format")
        categories = sorted(coco.get("categories", []), key=lambda item: int(item["id"]))
        category_name = {int(item["id"]): str(item["name"]).lower() for item in categories}
        class_to_idx = {name: position + 1 for position, name in enumerate(category_name.values())}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        records: list[dict[str, Any]] = []
        image_prefix = Path(getattr(cfg, "image_prefix", ""))
        prompt_prefix = str(getattr(cfg, "prompt_prefix", "an image containing"))
        for image_info in coco.get("images", []):
            image_id = int(image_info["id"])
            annotations = annotations_by_image.get(image_id, [])
            boxes = [list(map(float, item["bbox"])) for item in annotations]
            labels = [category_name.get(int(item["category_id"]), "unknown") for item in annotations]
            unique_names = sorted(set(labels))
            prompt = prompt_prefix + (" " + ", ".join(unique_names) if unique_names else "")
            records.append(
                {
                    "image": str(image_prefix / image_info["file_name"]),
                    "boxes": boxes,
                    "labels": labels,
                    "box_format": "xywh_abs",
                    "prompt": prompt,
                    "reference": None,
                }
            )
        instance_crop = getattr(cfg, "instance_crop", None)
        if instance_crop is not None and bool(getattr(instance_crop, "enabled", False)):
            allowed = {str(value).lower() for value in getattr(instance_crop, "allowed_phrases", [])}
            if allowed:
                records = [
                    record
                    for record in records
                    if record_has_instance_target(record, cfg, allowed)
                ]
        super().__init__(cfg, records, class_to_idx)


def build_dataset(cfg: Any) -> BoxDatasetBase:
    if cfg.type == "custom":
        dataset = CustomBoxDataset(cfg)
    elif cfg.type == "coco":
        dataset = CocoBoxDataset(cfg)
    else:
        raise ValueError(f"Unsupported data.type: {cfg.type}")
    if len(dataset.class_to_idx) + 1 > int(cfg.num_classes):
        needed = len(dataset.class_to_idx) + 1
        raise ValueError(f"data.num_classes={cfg.num_classes} is too small; at least {needed} is required")
    return dataset


def classify_layout_bucket(
    record: dict[str, Any], resolution: int = 512, prefer_small: bool = False
) -> str:
    width = int(record.get("width", resolution))
    height = int(record.get("height", resolution))
    boxes = convert_boxes_to_absolute_xyxy(
        record.get("boxes", []),
        record.get("box_format", "xyxy_abs"),
        width,
        height,
    )
    if boxes.numel() == 0:
        return "default"
    if prefer_small and str(record.get("dataset", "unknown")) == "DroneVehicle":
        border = 100
        if width > 2 * border and height > 2 * border:
            original_area = (
                (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
                * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
            )
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - border).clamp(0.0, float(width - 2 * border))
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - border).clamp(0.0, float(height - 2 * border))
            visible_area = (
                (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
                * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
            )
            boxes = boxes[(visible_area > 0) & (visible_area / original_area.clamp_min(1e-8) >= 0.5)]
            width -= 2 * border
            height -= 2 * border
    if boxes.numel() == 0:
        return "default"
    normalized = normalize_absolute_xyxy(boxes, width, height)
    side_lengths = (
        (normalized[:, 2] - normalized[:, 0])
        * (normalized[:, 3] - normalized[:, 1])
    ).sqrt() * float(resolution)
    has_small = bool((side_lengths <= 32.0).any())
    if prefer_small and has_small:
        return "small"
    if int(normalized.shape[0]) >= 8:
        return "crowded"
    if has_small:
        return "small"
    return "default"


def build_dataset_sample_weights(
    dataset: BoxDatasetBase,
    dataset_weights: dict[str, float],
    layout_weights: dict[str, float] | None = None,
    resolution: int = 512,
    prefer_small: bool = False,
) -> torch.Tensor:
    counts = Counter(str(record.get("dataset", "unknown")) for record in dataset.records)
    unknown = sorted(set(counts) - set(dataset_weights))
    if unknown:
        raise ValueError(f"Missing sampling weights for datasets: {unknown}")
    normalized = {str(name): float(weight) for name, weight in dataset_weights.items() if float(weight) > 0}
    if set(counts) - set(normalized):
        raise ValueError("Every dataset present in metadata needs a positive sampling weight")
    total_weight = sum(normalized[name] for name in counts)
    if total_weight <= 0:
        raise ValueError("Dataset sampling weights must sum to a positive value")
    target_share = {name: normalized[name] / total_weight for name in counts}
    layout_weights = {str(name): float(value) for name, value in (layout_weights or {}).items()}
    if any(value <= 0 for value in layout_weights.values()):
        raise ValueError("Layout sampling weights must be positive")

    def layout_multiplier(record: dict[str, Any]) -> float:
        if not layout_weights:
            return 1.0
        bucket = classify_layout_bucket(
            record, resolution=resolution, prefer_small=prefer_small
        )
        return float(layout_weights.get(bucket, layout_weights.get("default", 1.0)))

    sample_weights = [
        target_share[str(record.get("dataset", "unknown"))]
        / counts[str(record.get("dataset", "unknown"))]
        * layout_multiplier(record)
        for record in dataset.records
    ]
    tensor = torch.tensor(sample_weights, dtype=torch.double)
    if not torch.isfinite(tensor).all() or bool((tensor <= 0).any()):
        raise ValueError("Constructed non-positive or non-finite sample weights")
    return tensor


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic weighted sampling with disjoint rank shards per epoch."""

    def __init__(
        self,
        weights: torch.Tensor,
        samples_per_epoch: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        replacement: bool = True,
    ):
        if weights.ndim != 1 or weights.numel() < 1:
            raise ValueError("weights must be a non-empty one-dimensional tensor")
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("Invalid distributed sampler rank configuration")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        if not replacement and samples_per_epoch > int(weights.numel()):
            raise ValueError("Cannot sample more unique items than the dataset without replacement")
        self.weights = weights.detach().cpu().to(torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.global_samples = int(samples_per_epoch)
        self.num_samples = (self.global_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        shard = indices[self.rank:self.total_size:self.num_replicas]
        if len(shard) != self.num_samples:
            raise RuntimeError("Distributed weighted sampler produced an invalid shard")
        return iter(shard)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {
        "pixel_values": torch.stack([item["pixel_values"] for item in samples]),
        "boxes": torch.stack([item["boxes"] for item in samples]),
        "labels": torch.stack([item["labels"] for item in samples]),
        "box_mask": torch.stack([item["box_mask"] for item in samples]),
        "phrases": [item["phrases"] for item in samples],
        "reference_pixel_values": torch.stack([item["reference_pixel_values"] for item in samples]),
        "reference_present": torch.stack([item["reference_present"] for item in samples]),
        "prompt": [item["prompt"] for item in samples],
        "image_path": [item["image_path"] for item in samples],
        "dataset": [item["dataset"] for item in samples],
        "sample_id": [item["sample_id"] for item in samples],
        "instance_source_side": torch.stack([item["instance_source_side"] for item in samples]),
        "instance_normalized_side": torch.stack(
            [item["instance_normalized_side"] for item in samples]
        ),
        "instance_crop_xyxy": torch.stack([item["instance_crop_xyxy"] for item in samples]),
    }
    expected = len(samples)
    if batch["pixel_values"].shape[0] != expected:
        raise RuntimeError("Collation produced an unexpected batch size")
    return batch


def load_generation_requests(path: str | Path) -> list[dict[str, Any]]:
    raw = read_json_or_jsonl(path)
    if isinstance(raw, dict):
        records = raw.get("samples", raw.get("requests", [raw]))
    else:
        records = raw
    if not isinstance(records, list) or not records:
        raise ValueError("Generation request file contains no requests")
    normalized = []
    for index, record in enumerate(records):
        boxes, labels = parse_box_records(record.get("boxes", []), record.get("labels"))
        item = dict(record)
        item["boxes"] = boxes
        item["labels"] = labels
        item.setdefault("box_format", "xyxy_norm")
        item.setdefault("name", f"sample_{index:05d}")
        item.setdefault("prompt", "")
        item.setdefault("negative_prompt", "")
        normalized.append(item)
    return normalized
