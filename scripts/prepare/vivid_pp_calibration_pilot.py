#!/usr/bin/env python3
"""Extract and validate a small ViViD++ RGB/thermal calibration pilot.

The pilot uses the official ViViD++ Kalibr results, converts them to the
calibration contract used by ``ms2_pseudo_bbox.py``, reads ROS1 bags without a
ROS installation, pairs messages by ``msg.header.stamp``, undistorts both
cameras, and writes non-production review artifacts. It deliberately does not
create detection annotations or modify existing datasets.
"""

from __future__ import annotations

import argparse
import bz2
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image
import yaml


OFFICIAL_CALIBRATION_URL = (
    "https://urserver.kaist.ac.kr/publicdata/ViViD++/calibration/calibration_results.zip"
)


@dataclass(frozen=True)
class CameraModel:
    name: str
    topic: str
    camera_model: str
    distortion_model: str
    K: np.ndarray
    distortion: np.ndarray
    resolution: tuple[int, int]

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "topic": self.topic,
            "camera_model": self.camera_model,
            "distortion_model": self.distortion_model,
            "K": self.K.tolist(),
            "distortion": self.distortion.tolist(),
            "resolution": list(self.resolution),
        }


@dataclass(frozen=True)
class StereoCalibration:
    mode: str
    source_path: str
    rgb: CameraModel
    thermal: CameraModel
    rgb_to_thermal_m: np.ndarray
    source_transform_direction: str

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.rgb_to_thermal_m[:3, 3]))

    def as_json(self) -> dict[str, Any]:
        transform = self.rgb_to_thermal_m
        return {
            "mode": self.mode,
            "source_path": self.source_path,
            "source_transform_direction": self.source_transform_direction,
            "coordinate_contract": (
                "rgb_to_thermal_m maps a point in the undistorted RGB camera frame "
                "to the undistorted thermal camera frame"
            ),
            "translation_units": "meters",
            "baseline_m": self.baseline_m,
            "rgb": self.rgb.as_json(),
            "thermal": self.thermal.as_json(),
            "rgb_to_thermal_m": transform.tolist(),
        }


@dataclass(frozen=True)
class MessageRecord:
    topic: str
    connection: int
    bag_time: float | None
    payload: bytes


@dataclass(frozen=True)
class ImageSample:
    topic: str
    stamp: float
    sequence: int
    frame_id: str
    height: int
    width: int
    encoding: str
    array: np.ndarray


@dataclass(frozen=True)
class CameraInfoSample:
    topic: str
    stamp: float
    frame_id: str
    height: int
    width: int
    distortion_model: str
    K: np.ndarray


@dataclass(frozen=True)
class PairedSample:
    index: int
    rgb: ImageSample
    thermal: ImageSample
    delta_ms: float
    depth: ImageSample | None
    depth_delta_ms: float | None


class RosBagError(RuntimeError):
    """Raised for malformed or unsupported ROS1 bag data."""



def _field_map(buffer: bytes) -> dict[bytes, bytes]:
    fields: dict[bytes, bytes] = {}
    offset = 0
    while offset + 4 <= len(buffer):
        field_length = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        end = offset + field_length
        if end > len(buffer):
            raise RosBagError("record header field exceeds record")
        field = buffer[offset:end]
        offset = end
        key, separator, value = field.partition(b"=")
        if separator:
            fields[key] = value
    if offset != len(buffer):
        raise RosBagError("record header has a trailing partial field")
    return fields


def _field_uint(fields: dict[bytes, bytes], key: bytes, default: int = 0) -> int:
    value = fields.get(key)
    if value is None:
        return default
    if len(value) != 4:
        raise RosBagError(f"field {key!r} is not a uint32")
    return struct.unpack("<I", value)[0]


def _field_float_time(fields: dict[bytes, bytes], key: bytes = b"time") -> float | None:
    value = fields.get(key)
    if value is None:
        return None
    if len(value) != 8:
        return None
    sec, nsec = struct.unpack("<II", value)
    return float(sec) + float(nsec) / 1_000_000_000.0


def _read_record(handle: Any) -> tuple[dict[bytes, bytes], bytes] | None:
    length_bytes = handle.read(4)
    if not length_bytes:
        return None
    if len(length_bytes) != 4:
        raise RosBagError("truncated record header length")
    header_length = struct.unpack("<I", length_bytes)[0]
    header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise RosBagError("truncated record header")
    data_length_bytes = handle.read(4)
    if len(data_length_bytes) != 4:
        raise RosBagError("truncated record data length")
    data_length = struct.unpack("<I", data_length_bytes)[0]
    data = handle.read(data_length)
    if len(data) != data_length:
        raise RosBagError("truncated record payload")
    return _field_map(header_bytes), data


def _decode_connection_header(header: dict[bytes, bytes], data: bytes) -> tuple[int, str, str]:
    connection = _field_uint(header, b"conn", -1)
    values = _field_map(data)
    topic_bytes = values.get(b"topic", header.get(b"topic", b""))
    topic = topic_bytes.decode("utf-8", "replace")
    message_type = values.get(b"type", b"").decode("utf-8", "replace")
    return connection, topic, message_type


class RosBagReader:
    """Minimal ROS1 reader for uncompressed and bz2 chunks.

    It intentionally consumes only connection and message records. ROS bag
    index records are skipped. Message header timestamps are parsed later from
    the serialized ROS message and are the only timestamps used for pairing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connections: dict[int, tuple[str, str]] = {}

    def _register_connection(self, header: dict[bytes, bytes], data: bytes) -> None:
        connection, topic, message_type = _decode_connection_header(header, data)
        if connection >= 0 and topic:
            self.connections[connection] = (topic, message_type)

    def _decompress_chunk(self, header: dict[bytes, bytes], data: bytes) -> bytes:
        compression = header.get(b"compression", b"none")
        if compression == b"none":
            return data
        if compression == b"bz2":
            return bz2.decompress(data)
        if compression == b"lz4":
            try:
                import lz4.frame
            except ImportError as error:  # pragma: no cover - depends on bag variant
                raise RosBagError("lz4 chunk requires the optional lz4 package") from error
            return lz4.frame.decompress(data)
        raise RosBagError(f"unsupported ROS bag compression: {compression!r}")

    def _iter_chunk(self, data: bytes) -> Iterator[MessageRecord]:
        offset = 0
        while offset < len(data):
            if offset + 4 > len(data):
                raise RosBagError("truncated nested record header length")
            header_length = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            end_header = offset + header_length
            if end_header + 4 > len(data):
                raise RosBagError("truncated nested record header")
            header = _field_map(data[offset:end_header])
            offset = end_header
            data_length = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            end_data = offset + data_length
            if end_data > len(data):
                raise RosBagError("truncated nested record payload")
            payload = data[offset:end_data]
            offset = end_data
            operation = header.get(b"op", b"")
            if operation == b"\x07":
                self._register_connection(header, payload)
                continue
            if operation != b"\x02":
                continue
            connection = _field_uint(header, b"conn", -1)
            connection_info = self.connections.get(connection)
            if connection_info is None:
                continue
            yield MessageRecord(
                topic=connection_info[0],
                connection=connection,
                bag_time=_field_float_time(header),
                payload=payload,
            )

    def messages(self) -> Iterator[MessageRecord]:
        with self.path.open("rb") as handle:
            magic = handle.readline()
            if magic.rstrip(b"\r\n") != b"#ROSBAG V2.0":
                raise RosBagError(f"not a ROS1 bag: {self.path}")
            while True:
                record = _read_record(handle)
                if record is None:
                    return
                header, data = record
                operation = header.get(b"op", b"")
                if operation == b"\x07":
                    self._register_connection(header, data)
                elif operation == b"\x05":
                    yield from self._iter_chunk(self._decompress_chunk(header, data))


def _read_ros_string(buffer: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(buffer):
        raise RosBagError("truncated ROS string length")
    length = struct.unpack_from("<I", buffer, offset)[0]
    offset += 4
    end = offset + length
    if end > len(buffer):
        raise RosBagError("truncated ROS string")
    return buffer[offset:end].decode("utf-8", "replace"), end


def _read_ros_header(buffer: bytes) -> tuple[dict[str, Any], int]:
    if len(buffer) < 16:
        raise RosBagError("truncated ROS message header")
    sequence = struct.unpack_from("<I", buffer, 0)[0]
    seconds, nanoseconds = struct.unpack_from("<II", buffer, 4)
    frame_id, offset = _read_ros_string(buffer, 12)
    return {
        "sequence": sequence,
        "stamp": float(seconds) + float(nanoseconds) / 1_000_000_000.0,
        "frame_id": frame_id,
    }, offset


def _read_image_message(topic: str, payload: bytes) -> ImageSample:
    header, offset = _read_ros_header(payload)
    if offset + 8 > len(payload):
        raise RosBagError("truncated sensor_msgs/Image dimensions")
    height, width = struct.unpack_from("<II", payload, offset)
    offset += 8
    encoding, offset = _read_ros_string(payload, offset)
    if offset + 5 > len(payload):
        raise RosBagError("truncated sensor_msgs/Image layout")
    is_bigendian = payload[offset]
    offset += 1
    step = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    data_length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    end = offset + data_length
    if end > len(payload):
        raise RosBagError("truncated sensor_msgs/Image data")
    if height == 0 or width == 0:
        raise RosBagError("empty sensor_msgs/Image")
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        itemsize = 1
        dtype = np.dtype("u1")
    elif encoding in {"mono8"}:
        channels = 1
        itemsize = 1
        dtype = np.dtype("u1")
    elif encoding in {"mono16", "16UC1", "16SC1"}:
        channels = 1
        itemsize = 2
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
    elif encoding in {"32FC1"}:
        channels = 1
        itemsize = 4
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
    else:
        raise RosBagError(f"unsupported image encoding {encoding!r} on {topic}")
    expected_step = width * channels * itemsize
    if step < expected_step or step % itemsize:
        raise RosBagError(f"invalid image step {step} for {encoding} {width}x{height}")
    expected_bytes = step * height
    if data_length < expected_bytes:
        raise RosBagError("image payload is shorter than step*height")
    array = np.frombuffer(payload, dtype=dtype, count=expected_bytes // itemsize, offset=offset)
    array = array.reshape(height, step // itemsize)
    if channels > 1:
        array = array[:, : width * channels].reshape(height, width, channels)
    else:
        array = array[:, :width]
    array = np.array(array, copy=True)
    if array.dtype.byteorder not in {"=", "|"} and not array.dtype.isnative:
        array = array.byteswap().view(array.dtype.newbyteorder("="))
    return ImageSample(
        topic=topic,
        stamp=float(header["stamp"]),
        sequence=int(header["sequence"]),
        frame_id=str(header["frame_id"]),
        height=int(height),
        width=int(width),
        encoding=encoding,
        array=array,
    )


def _read_camera_info_message(topic: str, payload: bytes) -> CameraInfoSample:
    header, offset = _read_ros_header(payload)
    if offset + 8 > len(payload):
        raise RosBagError("truncated sensor_msgs/CameraInfo dimensions")
    height, width = struct.unpack_from("<II", payload, offset)
    offset += 8
    distortion_model, offset = _read_ros_string(payload, offset)
    if offset + 4 > len(payload):
        raise RosBagError("truncated CameraInfo distortion array")
    distortion_count = struct.unpack_from("<I", payload, offset)[0]
    offset += 4 + distortion_count * 8
    if offset + 9 * 8 > len(payload):
        raise RosBagError("truncated CameraInfo K")
    K = np.asarray(struct.unpack_from("<9d", payload, offset), dtype=np.float64).reshape(3, 3)
    return CameraInfoSample(
        topic=topic,
        stamp=float(header["stamp"]),
        frame_id=str(header["frame_id"]),
        height=int(height),
        width=int(width),
        distortion_model=distortion_model,
        K=K,
    )


def _camera_from_payload(name: str, payload: dict[str, Any]) -> CameraModel:
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (4,):
        raise ValueError(f"unexpected intrinsics for {name}: {intrinsics.shape}")
    fx, fy, cx, cy = intrinsics
    resolution_values = tuple(int(value) for value in payload["resolution"])
    if len(resolution_values) != 2:
        raise ValueError(f"unexpected resolution for {name}")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.asarray(payload["distortion_coeffs"], dtype=np.float64)
    return CameraModel(
        name=name,
        topic=str(payload["rostopic"]),
        camera_model=str(payload["camera_model"]),
        distortion_model=str(payload["distortion_model"]),
        K=K,
        distortion=distortion,
        resolution=resolution_values,
    )


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {transform.shape}")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _validate_rotation(rotation: np.ndarray, label: str) -> None:
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
        raise ValueError(
            f"{label} is not a proper rotation: "
            f"orthogonality_error={orthogonality_error:.3g}, determinant={determinant:.9f}"
        )


def load_stereo_calibration(calibration_root: Path, mode: str) -> StereoCalibration:
    if calibration_root.is_file():
        calibration_path = calibration_root
    else:
        relative = (
            Path("driving_results") / "camchain-rgbther.yaml"
            if mode == "driving"
            else Path("handheld_results") / "camchain-therrgb.yaml"
        )
        calibration_path = calibration_root / relative
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "cam0" not in payload or "cam1" not in payload:
        raise ValueError(f"invalid Kalibr camchain: {calibration_path}")
    cam0 = _camera_from_payload("cam0", payload["cam0"])
    cam1 = _camera_from_payload("cam1", payload["cam1"])
    transform = np.asarray(payload["cam1"]["T_cn_cnm1"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"invalid Kalibr transform in {calibration_path}: {transform.shape}")
    if mode == "driving":
        rgb, thermal = cam0, cam1
        rgb_to_thermal = transform
        direction = "Kalibr cam1 thermal <- cam0 RGB"
    elif mode == "handheld":
        thermal, rgb = cam0, cam1
        rgb_to_thermal = _invert_transform(transform)
        direction = "inverse of Kalibr cam1 RGB <- cam0 thermal"
    else:
        raise ValueError(f"unsupported mode: {mode}")
    _validate_rotation(rgb_to_thermal[:3, :3], "RGB-to-thermal rotation")
    for camera in (rgb, thermal):
        if camera.K[0, 0] <= 0 or camera.K[1, 1] <= 0:
            raise ValueError(f"invalid focal length in {camera.name}")
        if min(camera.resolution) <= 0:
            raise ValueError(f"invalid resolution in {camera.name}")
    return StereoCalibration(
        mode=mode,
        source_path=str(calibration_path),
        rgb=rgb,
        thermal=thermal,
        rgb_to_thermal_m=rgb_to_thermal,
        source_transform_direction=direction,
    )


def current_calibration_payload(calibration: StereoCalibration) -> dict[str, np.ndarray]:
    """Create the npy payload consumed by ms2_pseudo_bbox.load_calibration.

    The synthetic ``nir`` frame is deliberately defined as the RGB frame. This
    keeps the existing loader unchanged while preserving the official
    RGB-to-thermal transform and converting Kalibr meters to the project's
    millimeter translation contract.
    """

    rotation = calibration.rgb_to_thermal_m[:3, :3]
    translation_mm = calibration.rgb_to_thermal_m[:3, 3] * 1000.0
    return {
        "K_rgb": np.array(calibration.rgb.K, copy=True),
        "K_thr": np.array(calibration.thermal.K, copy=True),
        "R_nir2rgb": np.eye(3, dtype=np.float64),
        "R_nir2thr": np.array(rotation, copy=True),
        "T_nir2rgb": np.zeros(3, dtype=np.float64),
        "T_nir2thr": np.array(translation_mm, copy=True),
        "R_rgb_to_thr": np.array(rotation, copy=True),
        "T_rgb_to_thr_m": np.array(calibration.rgb_to_thermal_m[:3, 3], copy=True),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_calibration_artifacts(calibration: StereoCalibration, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / f"{calibration.mode}_calib.npy", current_calibration_payload(calibration))
    metadata = calibration.as_json()
    metadata.update(
        {
            "official_calibration_url": OFFICIAL_CALIBRATION_URL,
            "project_translation_contract": "millimeters in *_calib.npy",
            "undistortion_contract": (
                "images are undistorted with the official model and the same K; "
                "projection uses the resulting pinhole pixel coordinates"
            ),
            "manual_review_required": True,
            "production_rollout_allowed": False,
        }
    )
    metadata["source_sha256"] = _sha256(Path(calibration.source_path))
    (output_root / f"{calibration.mode}_calibration.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _topic_set(mode: str) -> dict[str, str | None]:
    if mode == "handheld":
        return {
            "rgb": "/rgb/image",
            "thermal": "/thermal/image_raw",
            "depth": "/depth/image_raw",
            "rgb_info": "/rgb/camera_info",
            "thermal_info": "/thermal/camera_info",
        }
    if mode == "driving":
        return {
            "rgb": "/camera/image_color",
            "thermal": "/thermal/image_raw",
            "depth": None,
            "rgb_info": "/camera/camera_info",
            "thermal_info": "/thermal/camera_info",
        }
    raise ValueError(f"unsupported mode: {mode}")


def collect_bag_samples(
    bag_path: Path,
    mode: str,
    source_cap: int,
    skip_source: int,
) -> tuple[dict[str, list[ImageSample]], dict[str, CameraInfoSample | None], dict[str, Any]]:
    topics = _topic_set(mode)
    image_topics = {
        value for value in topics.values() if value and not value.endswith("camera_info")
    }
    camera_info_topics = {
        value for value in topics.values() if value and value.endswith("camera_info")
    }
    images: dict[str, list[ImageSample]] = defaultdict(list)
    camera_infos: dict[str, CameraInfoSample | None] = {
        topic: None for topic in camera_info_topics
    }
    reader = RosBagReader(bag_path)
    total_messages = 0
    image_seen: defaultdict[str, int] = defaultdict(int)
    for message in reader.messages():
        total_messages += 1
        if message.topic in image_topics:
            image_seen[message.topic] += 1
            if (
                image_seen[message.topic] > skip_source
                and len(images[message.topic]) < source_cap
            ):
                images[message.topic].append(_read_image_message(message.topic, message.payload))
        elif message.topic in camera_info_topics and camera_infos[message.topic] is None:
            camera_infos[message.topic] = _read_camera_info_message(message.topic, message.payload)
        required = [topics["rgb"], topics["thermal"]]
        if all(len(images[topic or ""]) >= source_cap for topic in required):
            break
    if not images[topics["rgb"]] or not images[topics["thermal"]]:
        raise RosBagError(f"missing RGB/thermal images in {bag_path}")
    summary = {
        "bag": str(bag_path),
        "topics": sorted({topic for topic, _ in reader.connections.values()}),
        "message_records_scanned": total_messages,
        "image_counts": {topic: len(values) for topic, values in images.items()},
        "image_messages_skipped": dict(image_seen),
        "camera_info": {
            topic: (
                {
                    "stamp": info.stamp,
                    "frame_id": info.frame_id,
                    "width": info.width,
                    "height": info.height,
                    "distortion_model": info.distortion_model,
                    "K": info.K.tolist(),
                }
                if info is not None
                else None
            )
            for topic, info in camera_infos.items()
        },
        "has_tf_topic": "/tf" in reader.connections or "/tf_static" in reader.connections,
    }
    return images, camera_infos, summary


def _nearest_sample(
    timestamp: float,
    candidates: Sequence[ImageSample],
) -> tuple[int, float] | None:
    if not candidates:
        return None
    timestamps = [sample.stamp for sample in candidates]
    position = int(np.searchsorted(timestamps, timestamp))
    candidate_indices = {
        index
        for index in range(max(0, position - 8), min(len(candidates), position + 9))
    }
    index = min(candidate_indices, key=lambda value: abs(candidates[value].stamp - timestamp))
    return index, abs(candidates[index].stamp - timestamp)


def pair_samples(
    images: dict[str, list[ImageSample]],
    mode: str,
    max_sync_delta_ms: float,
    max_depth_delta_ms: float,
    pair_limit: int,
) -> list[PairedSample]:
    topics = _topic_set(mode)
    rgb_samples = sorted(images[topics["rgb"]], key=lambda sample: sample.stamp)
    thermal_samples = sorted(images[topics["thermal"]], key=lambda sample: sample.stamp)
    depth_samples = sorted(
        images.get(topics["depth"] or "", []), key=lambda sample: sample.stamp
    )
    pairs: list[PairedSample] = []
    max_sync_seconds = max_sync_delta_ms / 1000.0
    max_depth_seconds = max_depth_delta_ms / 1000.0
    for rgb in rgb_samples:
        thermal_match = _nearest_sample(rgb.stamp, thermal_samples)
        if thermal_match is None:
            continue
        thermal_index, delta = thermal_match
        if delta > max_sync_seconds:
            continue
        depth: ImageSample | None = None
        depth_delta_ms: float | None = None
        if depth_samples:
            depth_match = _nearest_sample(rgb.stamp, depth_samples)
            if depth_match is not None:
                depth_index, depth_delta = depth_match
                if depth_delta <= max_depth_seconds:
                    depth = depth_samples[depth_index]
                    depth_delta_ms = depth_delta * 1000.0
        pairs.append(
            PairedSample(
                index=len(pairs),
                rgb=rgb,
                thermal=thermal_samples[thermal_index],
                delta_ms=delta * 1000.0,
                depth=depth,
                depth_delta_ms=depth_delta_ms,
            )
        )
    if pair_limit > 0 and len(pairs) > pair_limit:
        indices = np.linspace(0, len(pairs) - 1, pair_limit, dtype=np.int64)
        pairs = [pairs[int(index)] for index in sorted(set(indices))]
    return [replace(pair, index=index) for index, pair in enumerate(pairs)]


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency availability
        raise RuntimeError("the pilot requires opencv-python-headless") from error
    return cv2


def build_undistort_map(camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    width, height = camera.resolution
    size = (width, height)
    identity = np.eye(3, dtype=np.float64)
    if camera.distortion_model == "equidistant":
        return cv2.fisheye.initUndistortRectifyMap(
            camera.K,
            camera.distortion.reshape(1, -1),
            identity,
            camera.K,
            size,
            cv2.CV_32FC1,
        )
    return cv2.initUndistortRectifyMap(
        camera.K,
        camera.distortion,
        identity,
        camera.K,
        size,
        cv2.CV_32FC1,
    )


def undistort_image(array: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    cv2 = _cv2()
    interpolation = cv2.INTER_NEAREST if array.ndim == 2 and array.dtype.itemsize > 1 else cv2.INTER_LINEAR
    return cv2.remap(array, maps[0], maps[1], interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)


def _save_image(array: np.ndarray, path: Path, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if encoding == "bgr8":
        array = array[..., ::-1]
    if array.dtype == np.uint16:
        Image.fromarray(array, mode="I;16").save(path)
    elif array.dtype == np.float32:
        scaled = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        finite = scaled[np.isfinite(scaled)]
        high = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        preview = np.clip(scaled * (255.0 / max(high, 1e-6)), 0, 255).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(path)
    else:
        Image.fromarray(array).save(path)


def _thermal_preview(thermal: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    values = thermal.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        normalized = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1.0, 99.0])
        normalized = np.clip((values - low) * 255.0 / max(float(high - low), 1.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def _depth_points_undistorted(
    depth: np.ndarray,
    calibration: StereoCalibration,
    depth_scale: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    depth_m = depth.astype(np.float64, copy=False) * depth_scale
    rows, columns = np.nonzero(
        np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 250.0)
    )
    total_valid = int(rows.size)
    if rows.size > max_points:
        indices = np.linspace(0, rows.size - 1, max_points, dtype=np.int64)
        rows, columns = rows[indices], columns[indices]
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), total_valid
    raw_uv = np.column_stack((columns, rows)).astype(np.float64).reshape(-1, 1, 2)
    cv2 = _cv2()
    if calibration.rgb.distortion_model == "equidistant":
        undistorted_uv = cv2.fisheye.undistortPoints(
            raw_uv,
            calibration.rgb.K,
            calibration.rgb.distortion.reshape(-1, 1),
            R=np.eye(3, dtype=np.float64),
            P=calibration.rgb.K,
        ).reshape(-1, 2)
    else:
        undistorted_uv = cv2.undistortPoints(
            raw_uv,
            calibration.rgb.K,
            calibration.rgb.distortion,
            R=np.eye(3, dtype=np.float64),
            P=calibration.rgb.K,
        ).reshape(-1, 2)
    return undistorted_uv, depth_m[rows, columns], total_valid


def _project_depth_grid(
    depth: np.ndarray,
    calibration: StereoCalibration,
    depth_scale: float,
    max_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    usv, z, total_valid = _depth_points_undistorted(
        depth, calibration, depth_scale, max_points
    )
    finite = np.isfinite(z) & (z > 0.1) & (z < 250.0)
    us, vs = usv[:, 0], usv[:, 1]
    k_rgb = calibration.rgb.K
    points_rgb = np.column_stack(
        (
            (us - k_rgb[0, 2]) * z / k_rgb[0, 0],
            (vs - k_rgb[1, 2]) * z / k_rgb[1, 1],
            z,
        )
    )
    transform = calibration.rgb_to_thermal_m
    points_thermal = points_rgb @ transform[:3, :3].T + transform[:3, 3]
    positive = np.isfinite(points_thermal).all(axis=-1) & (points_thermal[..., 2] > 0.1)
    uv_h = points_thermal @ calibration.thermal.K.T
    uv = uv_h[..., :2] / np.maximum(uv_h[..., 2:3], 1e-9)
    thermal_width, thermal_height = calibration.thermal.resolution
    inside = (
        finite
        & positive
        & np.isfinite(uv).all(axis=-1)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < thermal_width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < thermal_height)
    )
    diagnostics = {
        "depth_pixels_total": int(depth.shape[0] * depth.shape[1]),
        "depth_valid_total": total_valid,
        "sampled_points": int(finite.size),
        "depth_valid_count": int(finite.sum()),
        "positive_thermal_count": int((finite & positive).sum()),
        "projected_in_frame_count": int(inside.sum()),
        "projected_in_frame_fraction_over_depth": float(inside.sum() / max(int(finite.sum()), 1)),
        "projected_in_frame_fraction_over_sample": float(inside.mean()) if finite.size else 0.0,
        "thermal_depth_median_m": (
            float(np.median(points_thermal[..., 2][inside])) if inside.any() else None
        ),
    }
    return usv, uv, diagnostics


def _edge_support(
    rgb: np.ndarray,
    thermal: np.ndarray,
    depth: np.ndarray,
    calibration: StereoCalibration,
    depth_scale: float,
) -> dict[str, Any]:
    cv2 = _cv2()
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb
    rgb_edges = cv2.Canny(gray, 50, 150)
    thermal_preview = _thermal_preview(thermal)
    thermal_gray = cv2.cvtColor(thermal_preview, cv2.COLOR_BGR2GRAY)
    thermal_edges = cv2.Canny(thermal_gray, 50, 150)
    edge_pixels = int(np.count_nonzero(rgb_edges))
    if edge_pixels < 32:
        return {"rgb_edge_pixels": edge_pixels, "edge_support_ratio": None}
    depth_uv, z, _ = _depth_points_undistorted(
        depth, calibration, depth_scale, max_points=65536
    )
    if not z.size:
        return {
            "rgb_edge_pixels": edge_pixels,
            "depth_valid_points": 0,
            "edge_support_ratio": None,
        }
    h_rgb, w_rgb = rgb_edges.shape
    rounded = np.rint(depth_uv).astype(np.int64)
    rgb_inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < w_rgb)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < h_rgb)
    )
    rgb_edge_dilated = cv2.dilate(rgb_edges, np.ones((5, 5), dtype=np.uint8))
    edge_at_depth = np.zeros(len(rounded), dtype=bool)
    valid_indices = np.flatnonzero(rgb_inside)
    edge_at_depth[valid_indices] = rgb_edge_dilated[
        rounded[valid_indices, 1], rounded[valid_indices, 0]
    ] > 0
    if int(edge_at_depth.sum()) < 32:
        return {
            "rgb_edge_pixels": edge_pixels,
            "depth_valid_points": int(z.size),
            "depth_supported_rgb_edges": int(edge_at_depth.sum()),
            "edge_support_ratio": None,
        }
    depth_uv = depth_uv[edge_at_depth]
    z = z[edge_at_depth]
    if z.size > 4096:
        indices = np.linspace(0, z.size - 1, 4096, dtype=np.int64)
        depth_uv, z = depth_uv[indices], z[indices]
    k_rgb = calibration.rgb.K
    points = np.column_stack(
        (
            (depth_uv[:, 0] - k_rgb[0, 2]) * z / k_rgb[0, 0],
            (depth_uv[:, 1] - k_rgb[1, 2]) * z / k_rgb[1, 1],
            z,
        )
    )
    points = points @ calibration.rgb_to_thermal_m[:3, :3].T + calibration.rgb_to_thermal_m[:3, 3]
    uv_h = points @ calibration.thermal.K.T
    uv = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-9)
    h_thermal, w_thermal = thermal_edges.shape
    inside = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w_thermal)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h_thermal)
        & (points[:, 2] > 0.1)
    )
    uv = np.rint(uv[inside]).astype(np.int64)
    if uv.shape[0] < 32:
        return {
            "rgb_edge_pixels": edge_pixels,
            "depth_valid_points": int(z.size),
            "depth_supported_rgb_edges": int(depth_uv.shape[0]),
            "projected_edge_pixels": int(uv.shape[0]),
            "edge_support_ratio": None,
        }
    support = []
    for x, y in uv:
        x0, x1 = max(0, x - 2), min(w_thermal, x + 3)
        y0, y1 = max(0, y - 2), min(h_thermal, y + 3)
        support.append(bool(thermal_edges[y0:y1, x0:x1].any()))
    rng = np.random.default_rng(0)
    random_x = rng.integers(0, w_thermal, size=len(uv))
    random_y = rng.integers(0, h_thermal, size=len(uv))
    random_support = []
    for x, y in zip(random_x, random_y):
        x0, x1 = max(0, x - 2), min(w_thermal, x + 3)
        y0, y1 = max(0, y - 2), min(h_thermal, y + 3)
        random_support.append(bool(thermal_edges[y0:y1, x0:x1].any()))
    support_ratio = float(np.mean(support))
    random_ratio = float(np.mean(random_support))
    return {
        "rgb_edge_pixels": edge_pixels,
        "depth_valid_points": int(z.size),
        "depth_supported_rgb_edges": int(depth_uv.shape[0]),
        "projected_edge_pixels": int(uv.shape[0]),
        "edge_support_ratio": support_ratio,
        "random_edge_support_ratio": random_ratio,
        "edge_support_gain": support_ratio - random_ratio,
    }


def _sample_filename(stamp: float) -> str:
    seconds = int(math.floor(stamp))
    nanoseconds = int(round((stamp - seconds) * 1_000_000_000.0))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return f"{seconds:010d}_{nanoseconds:09d}.png"


def _select_pairs(pairs: list[PairedSample], requested: int) -> list[PairedSample]:
    if requested <= 0 or requested >= len(pairs):
        return pairs
    indices = np.linspace(0, len(pairs) - 1, requested, dtype=np.int64)
    selected = [pairs[int(index)] for index in sorted(set(indices))]
    return [replace(pair, index=index) for index, pair in enumerate(selected)]


def run_bag_pilot(
    bag_path: Path,
    mode: str,
    calibration: StereoCalibration,
    output_root: Path,
    source_cap: int,
    skip_source: int,
    pair_limit: int,
    max_sync_delta_ms: float,
    max_depth_delta_ms: float,
    depth_scale: float,
) -> dict[str, Any]:
    images, camera_infos, bag_summary = collect_bag_samples(
        bag_path, mode, source_cap, skip_source
    )
    pairs = pair_samples(images, mode, max_sync_delta_ms, max_depth_delta_ms, pair_limit)
    if not pairs:
        raise RosBagError(f"no synchronized pairs in {bag_path}")
    pairs = _select_pairs(pairs, pair_limit)
    rgb_map = build_undistort_map(calibration.rgb)
    thermal_map = build_undistort_map(calibration.thermal)
    bag_output = output_root / mode / bag_path.stem
    rows: list[dict[str, Any]] = []
    projection_metrics: list[dict[str, Any]] = []
    sync_deltas = [pair.delta_ms for pair in pairs]
    depth_deltas = [pair.depth_delta_ms for pair in pairs if pair.depth_delta_ms is not None]
    for pair in pairs:
        rgb = undistort_image(pair.rgb.array, rgb_map)
        thermal = undistort_image(pair.thermal.array, thermal_map)
        rgb_path = bag_output / "rgb" / _sample_filename(pair.rgb.stamp)
        thermal_path = bag_output / "thermal" / _sample_filename(pair.thermal.stamp)
        _save_image(rgb, rgb_path, pair.rgb.encoding)
        _save_image(thermal, thermal_path, pair.thermal.encoding)
        row: dict[str, Any] = {
            "index": pair.index,
            "rgb_stamp": pair.rgb.stamp,
            "thermal_stamp": pair.thermal.stamp,
            "sync_delta_ms": pair.delta_ms,
            "rgb_sequence": pair.rgb.sequence,
            "thermal_sequence": pair.thermal.sequence,
            "rgb_frame_id": pair.rgb.frame_id,
            "thermal_frame_id": pair.thermal.frame_id,
            "rgb_path": str(rgb_path),
            "thermal_path": str(thermal_path),
            "rgb_raw_encoding": pair.rgb.encoding,
            "thermal_raw_encoding": pair.thermal.encoding,
            "rgb_raw_resolution": [pair.rgb.width, pair.rgb.height],
            "thermal_raw_resolution": [pair.thermal.width, pair.thermal.height],
        }
        if pair.depth is not None:
            depth = pair.depth.array
            depth_path = bag_output / "depth" / _sample_filename(pair.depth.stamp)
            _save_image(depth, depth_path, pair.depth.encoding)
            row.update(
                {
                    "depth_stamp": pair.depth.stamp,
                    "depth_delta_ms": pair.depth_delta_ms,
                    "depth_path": str(depth_path),
                    "depth_encoding": pair.depth.encoding,
                    "depth_coordinate_contract": (
                        "raw RGB-depth pixels; projection undistorts depth coordinates "
                        "with the official RGB model"
                    ),
                }
            )
            _, projected_uv, projection = _project_depth_grid(
                depth,
                calibration,
                depth_scale,
            )
            projection.update(_edge_support(rgb, thermal, depth, calibration, depth_scale))
            overlay = _thermal_preview(thermal)
            cv2 = _cv2()
            valid_uv = projected_uv.reshape(-1, 2)
            for x, y in valid_uv:
                if (
                    np.isfinite(x)
                    and np.isfinite(y)
                    and 0 <= x < calibration.thermal.resolution[0]
                    and 0 <= y < calibration.thermal.resolution[1]
                ):
                    cv2.circle(overlay, (int(round(float(x))), int(round(float(y)))), 2, (0, 255, 0), -1)
            overlay_path = bag_output / "overlays" / f"{pair.index:04d}.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
            row["overlay_path"] = str(overlay_path)
            row["projection"] = projection
            projection_metrics.append(projection)
        rows.append(row)
    pairs_path = bag_output / "pairs.jsonl"
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    projection_fractions = [
        float(row["projection"]["projected_in_frame_fraction_over_depth"])
        for row in rows
        if "projection" in row
    ]
    valid_projection_rows = [
        row
        for row in rows
        if row.get("projection", {}).get("depth_valid_count", 0) >= 32
    ]
    valid_projection_fractions = [
        float(row["projection"]["projected_in_frame_fraction_over_depth"])
        for row in valid_projection_rows
    ]
    sync_p95_ms = float(np.percentile(sync_deltas, 95))
    resolution_match = all(
        tuple(row["rgb_raw_resolution"]) == calibration.rgb.resolution
        and tuple(row["thermal_raw_resolution"]) == calibration.thermal.resolution
        for row in rows
    )
    projection_valid_median = (
        float(np.median(valid_projection_fractions))
        if valid_projection_fractions
        else None
    )
    quality_gate = {
        "header_timestamp_sync_p95_ms_threshold": max_sync_delta_ms,
        "header_timestamp_sync_pass": sync_p95_ms <= max_sync_delta_ms,
        "calibration_resolution_pass": resolution_match,
        "depth_projection_status": (
            "not_evaluated"
            if not projection_metrics
            else (
                "pass"
                if projection_valid_median is not None and projection_valid_median >= 0.8
                else "fail"
            )
        ),
        "depth_projection_valid_sample_count": len(valid_projection_rows),
        "depth_projection_median_in_frame_fraction": projection_valid_median,
        "manual_overlay_review": "required",
        "production_rollout_allowed": False,
    }
    report = {
        **bag_summary,
        "mode": mode,
        "calibration_source": calibration.source_path,
        "selected_pairs": len(rows),
        "quality_gate": quality_gate,
        "sync_delta_ms": {
            "min": float(min(sync_deltas)),
            "p95": sync_p95_ms,
            "max": float(max(sync_deltas)),
        },
        "depth_delta_ms": (
            {
                "min": float(min(depth_deltas)),
                "median": float(np.median(depth_deltas)),
                "p95": float(np.percentile(depth_deltas, 95)),
                "max": float(max(depth_deltas)),
            }
            if depth_deltas
            else None
        ),
        "projection": {
            "samples": len(projection_metrics),
            "valid_depth_samples": len(valid_projection_rows),
            "depth_valid_total_median": (
                float(
                    np.median(
                        [
                            row["projection"]["depth_valid_total"]
                            for row in valid_projection_rows
                        ]
                    )
                )
                if valid_projection_rows
                else None
            ),
            "in_frame_fraction_median": (
                float(np.median(projection_fractions)) if projection_fractions else None
            ),
            "in_frame_fraction_median_valid_depth": (
                float(np.median(valid_projection_fractions))
                if valid_projection_fractions
                else None
            ),
            "in_frame_fraction_p05_valid_depth": (
                float(np.percentile(valid_projection_fractions, 5))
                if valid_projection_fractions
                else None
            ),
            "in_frame_fraction_min_valid_depth": (
                min(valid_projection_fractions) if valid_projection_fractions else None
            ),
            "edge_support_median": (
                float(
                    np.median(
                        [
                            value["edge_support_ratio"]
                            for value in projection_metrics
                            if value.get("edge_support_ratio") is not None
                        ]
                    )
                )
                if any(value.get("edge_support_ratio") is not None for value in projection_metrics)
                else None
            ),
        },
        "camera_info_observed": {
            topic: (
                {
                    "stamp": value.stamp,
                    "frame_id": value.frame_id,
                    "resolution": [value.width, value.height],
                    "distortion_model": value.distortion_model,
                    "K": value.K.tolist(),
                    "valid_intrinsics": bool(value.K[0, 0] > 0 and value.K[1, 1] > 0),
                }
                if value is not None
                else None
            )
            for topic, value in camera_infos.items()
        },
        "pairs_jsonl": str(pairs_path),
        "manual_review_required": True,
        "production_rollout_allowed": False,
    }
    (bag_output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("handheld", "driving"), required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--source-cap", type=int, default=80)
    parser.add_argument(
        "--skip-source",
        type=int,
        default=0,
        help="discard this many image messages per topic before retaining pilot samples",
    )
    parser.add_argument("--pair-limit", type=int, default=24)
    parser.add_argument("--max-sync-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-depth-delta-ms", type=float, default=50.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.source_cap <= 0 or args.pair_limit <= 0 or args.skip_source < 0:
        raise SystemExit("source-cap and pair-limit must be positive; skip-source must be nonnegative")
    calibration = load_stereo_calibration(args.calibration_root, args.mode)
    output_root = args.output_root.resolve()
    write_calibration_artifacts(calibration, output_root / "calibration")
    reports = []
    for bag in args.bag:
        report = run_bag_pilot(
            bag.resolve(),
            args.mode,
            calibration,
            output_root / "samples",
            args.source_cap,
            args.skip_source,
            args.pair_limit,
            args.max_sync_delta_ms,
            args.max_depth_delta_ms,
            args.depth_scale,
        )
        reports.append(report)
        print(
            f"{bag.name}: pairs={report['selected_pairs']} "
            f"sync_p95_ms={report['sync_delta_ms']['p95']:.3f} "
            f"projection_valid_median={report['projection']['in_frame_fraction_median_valid_depth']}",
            flush=True,
        )
    summary = {
        "mode": args.mode,
        "calibration": calibration.as_json(),
        "bags": reports,
        "sampling": {
            "source_cap_per_topic": args.source_cap,
            "skip_source_messages_per_topic": args.skip_source,
            "pair_limit_per_bag": args.pair_limit,
            "max_sync_delta_ms": args.max_sync_delta_ms,
            "max_depth_delta_ms": args.max_depth_delta_ms,
            "depth_scale": args.depth_scale,
        },
        "quality_gate": {
            "all_header_timestamp_sync_pass": all(
                report["quality_gate"]["header_timestamp_sync_pass"] for report in reports
            ),
            "all_resolution_pass": all(
                report["quality_gate"]["calibration_resolution_pass"] for report in reports
            ),
            "all_depth_projection_pass_or_not_evaluated": all(
                report["quality_gate"]["depth_projection_status"]
                in {"pass", "not_evaluated"}
                for report in reports
            ),
        },
        "manual_review_required": True,
        "production_rollout_allowed": False,
    }
    (output_root / "pilot_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
