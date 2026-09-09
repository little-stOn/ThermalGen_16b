#!/usr/bin/env python3
"""Validate ViViD++ RGB/thermal calibration with LiDAR overlays.

Driving full bags publish Ouster ``sensor_msgs/PointCloud2`` and handheld
outdoor bags publish Velodyne ``sensor_msgs/PointCloud2``. This pilot uses the
official Kalibr RGB/thermal and LiDAR/RGB results, pairs all streams by
message-header time, projects LiDAR points into both cameras, and writes
review-only artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image
import yaml


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import vivid_pp_calibration_pilot as base


@dataclass(frozen=True)
class PointCloudSample:
    stamp: float
    sequence: int
    frame_id: str
    points: np.ndarray


@dataclass(frozen=True)
class LidarCalibration:
    source_path: str
    camera_model: str
    distortion_model: str
    K: np.ndarray
    distortion: np.ndarray
    resolution: tuple[int, int]
    lidar_to_rgb: np.ndarray
    transform_key: str
    dt_lidar_to_ros: float | None
    def as_json(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "camera_model": self.camera_model,
            "distortion_model": self.distortion_model,
            "K": self.K.tolist(),
            "distortion": self.distortion.tolist(),
            "resolution": list(self.resolution),
            "lidar_to_rgb": self.lidar_to_rgb.tolist(),
            "transform_key": self.transform_key,
            "dt_lidar_to_ros": self.dt_lidar_to_ros,
        }


def _read_pointcloud2(topic: str, payload: bytes) -> PointCloudSample:
    header, offset = base._read_ros_header(payload)
    if offset + 8 > len(payload):
        raise base.RosBagError("truncated PointCloud2 dimensions")
    height, width = np.frombuffer(payload, dtype="<u4", count=2, offset=offset)
    offset += 8
    if offset + 4 > len(payload):
        raise base.RosBagError("truncated PointCloud2 fields length")
    field_count = int(np.frombuffer(payload, dtype="<u4", count=1, offset=offset)[0])
    offset += 4
    fields: dict[str, tuple[int, int, int]] = {}
    for _ in range(field_count):
        name, offset = base._read_ros_string(payload, offset)
        if offset + 9 > len(payload):
            raise base.RosBagError("truncated PointField")
        field_offset = int(np.frombuffer(payload, dtype="<u4", count=1, offset=offset)[0])
        datatype = int(payload[offset + 4])
        count = int(np.frombuffer(payload, dtype="<u4", count=1, offset=offset + 5)[0])
        offset += 9
        fields[name] = (field_offset, datatype, count)
    if offset + 13 > len(payload):
        raise base.RosBagError("truncated PointCloud2 layout")
    is_bigendian = bool(payload[offset])
    point_step = int(np.frombuffer(payload, dtype="<u4", count=1, offset=offset + 1)[0])
    row_step = int(np.frombuffer(payload, dtype="<u4", count=1, offset=offset + 5)[0])
    data_length = int(np.frombuffer(payload, dtype="<u4", count=1, offset=offset + 9)[0])
    offset += 13
    end = offset + data_length
    if end > len(payload):
        raise base.RosBagError("truncated PointCloud2 data")
    if height <= 0 or width <= 0 or point_step <= 0 or row_step < width * point_step:
        raise base.RosBagError("invalid PointCloud2 dimensions")
    if not all(name in fields for name in ("x", "y", "z")):
        raise base.RosBagError("PointCloud2 does not contain x/y/z fields")
    components = []
    for name in ("x", "y", "z"):
        field_offset, datatype, count = fields[name]
        if datatype != 7 or count < 1 or field_offset + 4 > point_step:
            raise base.RosBagError(f"PointCloud2 field {name} is not float32")
        components.append(field_offset)
    endian = ">" if is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [endian + "f4"] * 3,
            "offsets": components,
            "itemsize": point_step,
        }
    )
    rows: list[np.ndarray] = []
    for row in range(int(height)):
        row_start = offset + row * row_step
        row_bytes = payload[row_start : row_start + width * point_step]
        rows.append(np.frombuffer(row_bytes, dtype=dtype, count=int(width)))
    structured = np.concatenate(rows) if rows else np.empty(0, dtype=dtype)
    points = np.column_stack((structured["x"], structured["y"], structured["z"])).astype(
        np.float64, copy=False
    )
    finite = np.isfinite(points).all(axis=1)
    points = np.array(points[finite], copy=True)
    return PointCloudSample(
        stamp=float(header["stamp"]),
        sequence=int(header["sequence"]),
        frame_id=str(header["frame_id"]),
        points=points,
    )


def load_lidar_calibration(calibration_root: Path, mode: str) -> LidarCalibration:
    if mode == "driving":
        relative = Path("driving_results") / "rgblidar.yaml"
        transform_key = "T_lidar_rgb"
    elif mode == "handheld":
        relative = Path("handheld_results") / "rgblidar.yaml"
        transform_key = "T_lidar_cam"
    else:
        raise ValueError(f"unsupported LiDAR calibration mode: {mode}")
    path = calibration_root / relative
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera = payload["cam0"]
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = np.asarray(camera[transform_key], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"invalid LiDAR transform shape: {transform.shape}")
    base._validate_rotation(transform[:3, :3], "LiDAR-to-RGB rotation")
    dt_lidar_to_ros = (
        float(camera["dt_lidar_to_ros"])
        if camera.get("dt_lidar_to_ros") is not None
        else None
    )
    return LidarCalibration(
        source_path=str(path),
        camera_model=str(camera["camera_model"]),
        distortion_model=str(camera["distortion_model"]),
        K=K,
        distortion=np.asarray(camera["distortion_coeffs"], dtype=np.float64),
        resolution=tuple(int(value) for value in camera["resolution"]),
        lidar_to_rgb=transform,
        transform_key=transform_key,
        dt_lidar_to_ros=dt_lidar_to_ros,
    )


def collect_streams(
    bag_path: Path,
    mode: str,
    source_cap: int,
    lidar_topic: str,
) -> tuple[dict[str, list[base.ImageSample]], list[PointCloudSample], dict[str, Any]]:
    if mode == "driving":
        topics = {
            "rgb": "/camera/image_color",
            "thermal": "/thermal/image_raw",
        }
    elif mode == "handheld":
        topics = {
            "rgb": "/rgb/image",
            "thermal": "/thermal/image_raw",
        }
    else:
        raise ValueError(f"unsupported LiDAR pilot mode: {mode}")
    images: dict[str, list[base.ImageSample]] = {topic: [] for topic in topics.values()}
    clouds: list[PointCloudSample] = []
    reader = base.RosBagReader(bag_path)
    scanned = 0
    for message in reader.messages():
        scanned += 1
        if message.topic in images and len(images[message.topic]) < source_cap:
            images[message.topic].append(base._read_image_message(message.topic, message.payload))
        elif message.topic == lidar_topic and len(clouds) < source_cap:
            clouds.append(_read_pointcloud2(message.topic, message.payload))
        if all(len(images[topic]) >= source_cap for topic in topics.values()) and clouds:
            break
    if not all(images.values()):
        raise base.RosBagError(f"missing RGB/thermal stream in {bag_path}")
    return images, clouds, {
        "bag": str(bag_path),
        "mode": mode,
        "topics": sorted({topic for topic, _ in reader.connections.values()}),
        "message_records_scanned": scanned,
        "image_counts": {topic: len(values) for topic, values in images.items()},
        "pointcloud_count": len(clouds),
        "pointcloud_topic": lidar_topic,
    }


def nearest_cloud(
    timestamp: float,
    clouds: Sequence[PointCloudSample],
    dt_lidar_to_ros: float | None = None,
) -> tuple[int, float] | None:
    if not clouds:
        return None
    offset = float(dt_lidar_to_ros or 0.0)
    index = min(
        range(len(clouds)),
        key=lambda value: abs(clouds[value].stamp + offset - timestamp),
    )
    return index, abs(clouds[index].stamp + offset - timestamp)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(
    points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
    distortion_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    cv2 = base._cv2()
    positive = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if not positive.any():
        return uv, positive
    xyz = points[positive].reshape(-1, 1, 3).astype(np.float64)
    if distortion_model == "equidistant":
        projected, _ = cv2.fisheye.projectPoints(
            xyz,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            distortion.reshape(-1, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            xyz,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            distortion,
        )
    uv[positive] = projected.reshape(-1, 2)
    return uv, positive


def projection_metrics(
    points_lidar: np.ndarray,
    lidar_calibration: LidarCalibration,
    stereo: base.StereoCalibration,
    lidar_to_rgb: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    points_rgb = apply_transform(points_lidar, lidar_to_rgb)
    rgb_uv, rgb_positive = project_points(
        points_rgb,
        lidar_calibration.K,
        lidar_calibration.distortion,
        lidar_calibration.distortion_model,
    )
    points_thermal = apply_transform(points_rgb, stereo.rgb_to_thermal_m)
    thermal_uv, thermal_positive = project_points(
        points_thermal,
        stereo.thermal.K,
        stereo.thermal.distortion,
        stereo.thermal.distortion_model,
    )
    rgb_width, rgb_height = lidar_calibration.resolution
    thermal_width, thermal_height = stereo.thermal.resolution
    rgb_inside = (
        rgb_positive
        & np.isfinite(rgb_uv).all(axis=1)
        & (rgb_uv[:, 0] >= 0)
        & (rgb_uv[:, 0] < rgb_width)
        & (rgb_uv[:, 1] >= 0)
        & (rgb_uv[:, 1] < rgb_height)
    )
    thermal_inside = (
        thermal_positive
        & np.isfinite(thermal_uv).all(axis=1)
        & (thermal_uv[:, 0] >= 0)
        & (thermal_uv[:, 0] < thermal_width)
        & (thermal_uv[:, 1] >= 0)
        & (thermal_uv[:, 1] < thermal_height)
    )
    metrics = {
        "lidar_points": int(len(points_lidar)),
        "rgb_positive_points": int(rgb_positive.sum()),
        "rgb_in_frame_points": int(rgb_inside.sum()),
        "rgb_in_frame_fraction_over_positive": float(
            rgb_inside.sum() / max(int(rgb_positive.sum()), 1)
        ),
        "thermal_positive_points": int(thermal_positive.sum()),
        "thermal_in_frame_points": int(thermal_inside.sum()),
        "thermal_in_frame_fraction_over_positive": float(
            thermal_inside.sum() / max(int(thermal_positive.sum()), 1)
        ),
        "rgb_thermal_joint_in_frame_points": int((rgb_inside & thermal_inside).sum()),
        "joint_in_frame_fraction_over_lidar": float((rgb_inside & thermal_inside).mean()),
    }
    return metrics, rgb_uv, thermal_uv


def _thermal_preview(thermal: np.ndarray) -> np.ndarray:
    return base._thermal_preview(thermal)


def save_overlay(
    rgb: np.ndarray,
    thermal: np.ndarray,
    rgb_uv: np.ndarray,
    thermal_uv: np.ndarray,
    lidar_points: np.ndarray,
    lidar_calibration: LidarCalibration,
    stereo: base.StereoCalibration,
    output_dir: Path,
) -> tuple[Path, Path]:
    cv2 = base._cv2()
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        rgb_image = rgb.copy()
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    else:
        rgb_image = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    thermal_image = _thermal_preview(thermal)
    rgb_width, rgb_height = lidar_calibration.resolution
    thermal_width, thermal_height = stereo.thermal.resolution
    rgb_inside = (
        np.isfinite(rgb_uv).all(axis=1)
        & (rgb_uv[:, 0] >= 0)
        & (rgb_uv[:, 0] < rgb_width)
        & (rgb_uv[:, 1] >= 0)
        & (rgb_uv[:, 1] < rgb_height)
    )
    thermal_inside = (
        np.isfinite(thermal_uv).all(axis=1)
        & (thermal_uv[:, 0] >= 0)
        & (thermal_uv[:, 0] < thermal_width)
        & (thermal_uv[:, 1] >= 0)
        & (thermal_uv[:, 1] < thermal_height)
    )
    if len(lidar_points) > 8192:
        indices = np.linspace(0, len(lidar_points) - 1, 8192, dtype=np.int64)
    else:
        indices = np.arange(len(lidar_points), dtype=np.int64)
    for index in indices:
        if rgb_inside[index]:
            cv2.circle(
                rgb_image,
                (int(round(float(rgb_uv[index, 0]))), int(round(float(rgb_uv[index, 1])))),
                2,
                (0, 255, 0),
                -1,
            )
        if thermal_inside[index]:
            cv2.circle(
                thermal_image,
                (int(round(float(thermal_uv[index, 0]))), int(round(float(thermal_uv[index, 1])))),
                2,
                (0, 255, 0),
                -1,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "lidar_rgb_overlay.png"
    thermal_path = output_dir / "lidar_thermal_overlay.png"
    cv2.imwrite(str(rgb_path), rgb_image)
    cv2.imwrite(str(thermal_path), thermal_image)
    return rgb_path, thermal_path


def select_cloud_pairs(
    pairs: Sequence[base.PairedSample],
    clouds: Sequence[PointCloudSample],
    dt_lidar_to_ros: float | None,
    max_lidar_delta_ms: float,
    pair_limit: int,
) -> list[tuple[base.PairedSample, PointCloudSample, float]]:
    best_by_cloud: dict[int, tuple[float, base.PairedSample, PointCloudSample]] = {}
    for pair in pairs:
        cloud_match = nearest_cloud(pair.rgb.stamp, clouds, dt_lidar_to_ros)
        if cloud_match is None:
            continue
        cloud_index, cloud_delta = cloud_match
        delta_ms = cloud_delta * 1000.0
        if delta_ms > max_lidar_delta_ms:
            continue
        candidate = (delta_ms, pair, clouds[cloud_index])
        current = best_by_cloud.get(cloud_index)
        if current is None or candidate[0] < current[0]:
            best_by_cloud[cloud_index] = candidate
    selected = sorted(best_by_cloud.values(), key=lambda value: value[1].rgb.stamp)
    if pair_limit > 0 and len(selected) > pair_limit:
        indices = np.linspace(0, len(selected) - 1, pair_limit, dtype=np.int64)
        selected = [selected[int(index)] for index in sorted(set(indices))]
    return [(pair, cloud, delta_ms) for delta_ms, pair, cloud in selected]


def run(args: argparse.Namespace) -> int:
    calibration_root = args.calibration_root.resolve()
    stereo = base.load_stereo_calibration(calibration_root, args.mode)
    lidar_calibration = load_lidar_calibration(calibration_root, args.mode)
    output_root = args.output_root.resolve()
    lidar_topic = args.lidar_topic or (
        "/os1_cloud_node/points" if args.mode == "driving" else "/velodyne_point_cloud"
    )
    base.write_calibration_artifacts(stereo, output_root / "calibration")
    all_reports: list[dict[str, Any]] = []
    for bag in args.bag:
        bag_path = bag.resolve()
        images, clouds, collection = collect_streams(
            bag_path, args.mode, args.source_cap, lidar_topic
        )
        rgb_topic = "/camera/image_color" if args.mode == "driving" else "/rgb/image"
        thermal_topic = "/thermal/image_raw"
        rgb_samples = sorted(images[rgb_topic], key=lambda sample: sample.stamp)
        thermal_samples = sorted(images[thermal_topic], key=lambda sample: sample.stamp)
        pairs = base.pair_samples(
            {rgb_topic: rgb_samples, thermal_topic: thermal_samples},
            args.mode,
            args.max_sync_delta_ms,
            args.max_sync_delta_ms,
            0,
        )
        if not pairs:
            raise base.RosBagError(f"no synchronized RGB/thermal pairs in {bag_path}")
        matches = select_cloud_pairs(
            pairs,
            clouds,
            lidar_calibration.dt_lidar_to_ros,
            args.max_lidar_delta_ms,
            args.pair_limit,
        )
        if not matches:
            raise base.RosBagError(f"no synchronized LiDAR/image pairs in {bag_path}")
        bag_output = output_root / "samples" / bag_path.stem
        direction_metrics: dict[str, list[dict[str, Any]]] = {"as_stored": [], "inverse": []}
        stored = lidar_calibration.lidar_to_rgb
        inverse = base._invert_transform(stored)
        for _, cloud, _ in matches:
            for name, transform in (("as_stored", stored), ("inverse", inverse)):
                metrics, _, _ = projection_metrics(cloud.points, lidar_calibration, stereo, transform)
                direction_metrics[name].append(metrics)
        def median(name: str, key: str) -> float:
            values = [float(row[key]) for row in direction_metrics[name]]
            return float(np.median(values)) if values else 0.0
        default_direction = "as_stored" if args.mode == "handheld" else "inverse"
        selected_direction = args.lidar_direction or default_direction
        selected_transform = stored if selected_direction == "as_stored" else inverse
        lidar_rows: list[dict[str, Any]] = []
        for pair, cloud, cloud_delta in matches:
            metrics, rgb_uv, thermal_uv = projection_metrics(
                cloud.points, lidar_calibration, stereo, selected_transform
            )
            pair_dir = bag_output / "overlays" / f"{len(lidar_rows):04d}"
            rgb_path, thermal_path = save_overlay(
                pair.rgb.array,
                pair.thermal.array,
                rgb_uv,
                thermal_uv,
                cloud.points,
                lidar_calibration,
                stereo,
                pair_dir,
            )
            cloud_path = bag_output / "pointcloud" / f"{len(lidar_rows):04d}.npy"
            cloud_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cloud_path, cloud.points.astype(np.float32, copy=False))
            lidar_rows.append(
                {
                    "index": len(lidar_rows),
                    "rgb_stamp": pair.rgb.stamp,
                    "thermal_stamp": pair.thermal.stamp,
                    "lidar_stamp_raw": cloud.stamp,
                    "lidar_stamp_ros": cloud.stamp + float(lidar_calibration.dt_lidar_to_ros or 0.0),
                    "lidar_to_ros_offset_s": lidar_calibration.dt_lidar_to_ros,
                    "rgb_thermal_delta_ms": pair.delta_ms,
                    "lidar_rgb_delta_ms": cloud_delta,
                    "rgb_sequence": pair.rgb.sequence,
                    "thermal_sequence": pair.thermal.sequence,
                    "pointcloud_path": str(cloud_path),
                    "rgb_overlay_path": str(rgb_path),
                    "thermal_overlay_path": str(thermal_path),
                    "projection": metrics,
                }
            )
        rgb_thermal_deltas = [pair.delta_ms for pair, _, _ in matches]
        lidar_deltas = [row["lidar_rgb_delta_ms"] for row in lidar_rows]
        selected_direction_reason = (
            "handheld T_lidar_cam applied as stored after overlay validation"
            if args.mode == "handheld"
            else "driving T_lidar_rgb inverted after overlay validation on raw OS1 points"
        )
        selected_rgb_median = median(selected_direction, "rgb_in_frame_fraction_over_positive")
        selected_thermal_median = median(
            selected_direction, "thermal_in_frame_fraction_over_positive"
        )
        selected_joint_median = median(
            selected_direction, "joint_in_frame_fraction_over_lidar"
        )
        quality_gate = {
            "rgb_thermal_sync_p95_ms": (
                float(np.percentile(rgb_thermal_deltas, 95)) if rgb_thermal_deltas else None
            ),
            "lidar_rgb_sync_p95_ms": (
                float(np.percentile(lidar_deltas, 95)) if lidar_deltas else None
            ),
            "sync_pass": bool(
                rgb_thermal_deltas
                and lidar_deltas
                and np.percentile(rgb_thermal_deltas, 95) <= args.max_sync_delta_ms
                and np.percentile(lidar_deltas, 95) <= args.max_lidar_delta_ms
            ),
            "selected_direction": selected_direction,
            "selected_rgb_in_frame_median": selected_rgb_median,
            "selected_thermal_in_frame_median": selected_thermal_median,
            "selected_joint_in_frame_median": selected_joint_median,
            "projection_status": "manual_overlay_review_required",
            "production_rollout_allowed": False,
        }
        pair_path = bag_output / "lidar_pairs.jsonl"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        with pair_path.open("w", encoding="utf-8") as handle:
            for row in lidar_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report = {
            **collection,
            "mode": f"{args.mode}_lidar",
            "calibration": stereo.as_json(),
            "lidar_calibration": lidar_calibration.as_json(),
            "selected_pairs": len(lidar_rows),
            "selected_lidar_direction": selected_direction,
            "selected_lidar_direction_reason": selected_direction_reason,
            "quality_gate": quality_gate,
            "direction_metrics": {
                direction: {
                    "samples": len(values),
                    "rgb_in_frame_median": median(direction, "rgb_in_frame_fraction_over_positive"),
                    "thermal_in_frame_median": median(direction, "thermal_in_frame_fraction_over_positive"),
                    "joint_in_frame_median": median(direction, "joint_in_frame_fraction_over_lidar"),
                }
                for direction, values in direction_metrics.items()
            },
            "pairs_jsonl": str(pair_path),
            "manual_review_required": True,
            "production_rollout_allowed": False,
        }
        (bag_output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_reports.append(report)
        print(
            f"{bag.name}: pairs={len(lidar_rows)} direction={selected_direction} "
            f"rgb_median={report['direction_metrics'][selected_direction]['rgb_in_frame_median']:.3f} "
            f"thermal_median={report['direction_metrics'][selected_direction]['thermal_in_frame_median']:.3f}",
            flush=True,
        )
    summary = {
        "mode": f"{args.mode}_lidar",
        "calibration": stereo.as_json(),
        "lidar_calibration": lidar_calibration.as_json(),
        "bags": all_reports,
        "manual_review_required": True,
        "production_rollout_allowed": False,
    }
    (output_root / "pilot_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("handheld", "driving"), required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--lidar-topic")
    parser.add_argument("--source-cap", type=int, default=24)
    parser.add_argument("--pair-limit", type=int, default=12)
    parser.add_argument("--max-sync-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-lidar-delta-ms", type=float, default=50.0)
    parser.add_argument(
        "--lidar-direction",
        choices=("as_stored", "inverse"),
        help="override the platform-specific official transform direction",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.source_cap <= 0 or args.pair_limit <= 0:
        raise SystemExit("source-cap and pair-limit must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
