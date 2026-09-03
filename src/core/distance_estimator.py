"""
Distance and Spatial Orientation Estimation Module for Second Eye.
Combines:
1. Multi-Constraint Adaptive Pinhole Geometry (Aspect-Ratio Truncation Aware)
2. Ground-Plane Horizon Perspective Projection (Floor Contact Prior)
3. Object-Level Temporal Tracking & Exponential Moving Average (EMA) Smoothing
4. Persistent Camera Calibration for MacBook / Webcams / Continuity Camera
"""

import os
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

from src.config import (
    INDOOR_CLASSES,
    DIST_DANGER_THRESHOLD,
    DIST_WARNING_THRESHOLD,
    ZONE_LEFT_MAX,
    ZONE_CENTER_MAX,
    DEFAULT_FOCAL_LENGTH,
    DEFAULT_CAMERA_HEIGHT,
    DATA_DIR
)

CALIBRATION_FILE = DATA_DIR / "camera_config.json"


def compute_bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter_area / float(areaA + areaB - inter_area + 1e-6)


@dataclass
class DetectedObject:
    class_key: str
    name_vi: str
    name_en: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    distance: float                  # Estimated distance in meters (temporally smoothed)
    direction_vi: str                # "Bên trái", "Phía trước", "Bên phải"
    direction_en: str                # "left", "center", "right"
    risk_level: str                  # "DANGER", "WARNING", "SAFE"
    rel_x: float                     # Normalized horizontal center [0.0 - 1.0]
    rel_y: float                     # Normalized vertical bottom [0.0 - 1.0]
    coord_3d: Tuple[float, float]    # (X_lateral, Z_depth) in meters

    def to_dict(self):
        return {
            "class_key": self.class_key,
            "name_vi": self.name_vi,
            "name_en": self.name_en,
            "confidence": round(float(self.confidence), 2),
            "bbox": [int(x) for x in self.bbox],
            "distance": round(float(self.distance), 2),
            "direction_vi": self.direction_vi,
            "direction_en": self.direction_en,
            "risk_level": self.risk_level,
            "rel_x": round(float(self.rel_x), 3),
            "rel_y": round(float(self.rel_y), 3),
            "coord_3d": [round(float(self.coord_3d[0]), 2), round(float(self.coord_3d[1]), 2)]
        }


class DistanceEstimator:
    def __init__(
        self,
        focal_length: float = DEFAULT_FOCAL_LENGTH,
        camera_height: float = DEFAULT_CAMERA_HEIGHT
    ):
        """
        Initialize Enhanced Distance Estimator.
        :param focal_length: Camera focal length in pixels (default: 500.0 for 70° FOV)
        :param camera_height: Camera height above ground in meters (default: 1.0m)
        """
        self.focal_length = focal_length
        self.camera_height = camera_height
        self.tilt_angle_deg = 5.0 # Slight downward tilt angle in degrees

        # Temporal object tracks for smooth, jitter-free distance estimation:
        # track_id -> {"class_key": str, "bbox": tuple, "smoothed_dist": float, "last_time": float}
        self.tracks: Dict[str, dict] = {}
        self._track_counter = 0

        # Load persisted calibration if present
        self.load_calibration()

    def load_calibration(self):
        """Load calibrated camera parameters from data/camera_config.json if available."""
        try:
            if CALIBRATION_FILE.exists():
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "focal_length" in data:
                        self.focal_length = max(100.0, float(data["focal_length"]))
                    if "camera_height" in data:
                        self.camera_height = max(0.3, float(data["camera_height"]))
                    if "tilt_angle_deg" in data:
                        self.tilt_angle_deg = float(data["tilt_angle_deg"])
        except Exception as e:
            print(f"[DistanceEstimator] Lưu ý không thể tải calibration: {e}")

    def save_calibration(self):
        """Save calibrated camera parameters to data/camera_config.json."""
        try:
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "focal_length": round(self.focal_length, 1),
                    "camera_height": round(self.camera_height, 2),
                    "tilt_angle_deg": round(self.tilt_angle_deg, 1)
                }, f, indent=2)
            print(f"[DistanceEstimator] Đã lưu thông số hiệu chuẩn: Tiêu cự = {self.focal_length:.0f}px")
        except Exception as e:
            print(f"[DistanceEstimator] Lỗi khi lưu calibration: {e}")

    def update_focal_length(self, new_focal_length: float, save: bool = True):
        """Update focal length calibration value."""
        self.focal_length = max(100.0, float(new_focal_length))
        if save:
            self.save_calibration()

    def calibrate_with_known_distance(
        self,
        known_distance_m: float,
        observed_bbox_h: int,
        real_height_m: float
    ) -> Optional[float]:
        """
        Auto-calibrate focal length from a known reference distance:
        focal_length = (known_distance * observed_bbox_height) / real_height
        """
        if observed_bbox_h > 15 and real_height_m > 0:
            self.focal_length = (known_distance_m * float(observed_bbox_h)) / float(real_height_m)
            self.save_calibration()
            return self.focal_length
        return None

    def estimate_distance(
        self,
        class_key: str,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Calculate metric distance (meters) using multi-constraint geometric fusion:
        1. Aspect-Ratio Truncation Awareness (Detects sitting/torso person vs standing)
        2. Boundary Edge Truncation Detection (Height cut off at top/bottom)
        3. Ground-Plane Horizon Perspective Projection (Floor contact prior)
        """
        x1, y1, x2, y2 = bbox
        bbox_height = max(1, y2 - y1)
        bbox_width = max(1, x2 - x1)
        frame_h, frame_w = frame_shape[:2]

        info = INDOOR_CLASSES.get(class_key)
        if not info:
            real_h = 1.0
            real_w = 0.5
        else:
            real_h = info.real_height
            real_w = info.real_width

        obs_ar = bbox_height / float(bbox_width)
        is_vertically_cropped = (y1 <= 4) or (y2 >= frame_h - 4)
        is_horizontally_cropped = (x1 <= 4) or (x2 >= frame_w - 4)

        effective_h = real_h
        effective_w = real_w

        # 1. Aspect-Ratio & Posture Adaptation
        # In typical indoor camera use (e.g. sitting in front of laptop or walking indoors):
        # Full standing person: AR ~ 2.4 - 3.5 (H=1.65m, W=0.50m)
        # Sitting / Torso / Desk person: AR ~ 1.2 - 1.9 (H=0.88m, W=0.50m)
        # Head / Shoulders only: AR < 1.2 (H=0.45m, W=0.38m)
        if class_key == "person":
            if is_vertically_cropped or obs_ar < 1.9:
                if obs_ar < 1.2:
                    effective_h = 0.45
                    effective_w = 0.38
                else:
                    effective_h = 0.88
                    effective_w = 0.50

        # 2. Geometric Pinhole Distances
        dist_h = (self.focal_length * effective_h) / float(bbox_height)
        dist_w = (self.focal_length * effective_w) / float(bbox_width)

        # 3. Ground-plane horizon projection
        # If the object touches the floor and bottom edge is below optical horizon
        c_y = frame_h / 2.0
        dist_ground = None
        if y2 > c_y + 15:
            theta = math.radians(self.tilt_angle_deg)
            phi = math.atan((y2 - c_y) / self.focal_length)
            tan_total = math.tan(theta + phi)
            if tan_total > 0.05:
                dist_ground = self.camera_height / tan_total

        # 4. Adaptive Geometric Fusion
        if is_vertically_cropped and not is_horizontally_cropped:
            # Top or bottom cut off: trust width and floor projection much more than height
            if dist_ground is not None and abs(dist_w - dist_ground) < 2.0:
                raw_dist = 0.65 * dist_w + 0.35 * dist_ground
            else:
                raw_dist = dist_w
        elif is_horizontally_cropped and not is_vertically_cropped:
            # Left or right cut off: trust height
            raw_dist = dist_h
        else:
            # Uncropped or standard view:
            if class_key == "person" and obs_ar < 1.9:
                raw_dist = 0.65 * dist_h + 0.35 * dist_w
            else:
                raw_dist = dist_h

        # Ground plane sanity check for indoor obstacles (floor bounds)
        if dist_ground is not None and not is_vertically_cropped:
            if raw_dist > dist_ground * 2.2:
                raw_dist = 0.5 * raw_dist + 0.5 * dist_ground
            elif raw_dist < dist_ground * 0.45:
                raw_dist = 0.6 * raw_dist + 0.4 * dist_ground

        # Physical clamping (0.25m minimum to 12.0m maximum indoor range)
        return float(np.clip(raw_dist, 0.25, 12.0))

    def _smooth_distance(
        self,
        class_key: str,
        bbox: Tuple[int, int, int, int],
        raw_distance: float,
        timestamp: float
    ) -> float:
        """
        Track objects across consecutive video frames and apply an
        Exponential Moving Average (EMA) with Outlier Rejection to eliminate jitter.
        """
        best_id = None
        best_iou = 0.28

        for tid, track in list(self.tracks.items()):
            if track["class_key"] == class_key:
                iou = compute_bbox_iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

        if best_id is not None:
            prev_dist = self.tracks[best_id]["smoothed_dist"]
            # Outlier damping: if a single frame jumps by > 1.2m, limit the jump
            delta = raw_distance - prev_dist
            if abs(delta) > 1.2:
                raw_distance = prev_dist + (1.2 if delta > 0 else -1.2)

            # EMA Smoothing (alpha=0.38 provides crisp responsiveness with rock-solid stability)
            alpha = 0.38
            smoothed = alpha * raw_distance + (1.0 - alpha) * prev_dist

            self.tracks[best_id]["bbox"] = bbox
            self.tracks[best_id]["smoothed_dist"] = smoothed
            self.tracks[best_id]["last_time"] = timestamp
            return float(smoothed)
        else:
            # Register new tracked object
            self._track_counter += 1
            new_id = f"{class_key}_{self._track_counter}"
            self.tracks[new_id] = {
                "class_key": class_key,
                "bbox": bbox,
                "smoothed_dist": raw_distance,
                "last_time": timestamp
            }
            # Clean up stale tracks older than 1.2 seconds
            self._prune_tracks(timestamp)
            return float(raw_distance)

    def _prune_tracks(self, current_time: float):
        """Remove tracks that haven't been seen for > 1.2 seconds."""
        stale_ids = [
            tid for tid, trk in self.tracks.items()
            if (current_time - trk["last_time"]) > 1.2
        ]
        for tid in stale_ids:
            del self.tracks[tid]

    def determine_direction(self, bbox: Tuple[int, int, int, int], frame_width: int) -> Tuple[str, str, float]:
        """
        Determine spatial direction (Left, Center, Right) based on bbox horizontal center.
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        rel_x = center_x / float(frame_width)

        if rel_x < ZONE_LEFT_MAX:
            return "Bên trái", "left", rel_x
        elif rel_x <= ZONE_CENTER_MAX:
            return "Phía trước", "center", rel_x
        else:
            return "Bên phải", "right", rel_x

    def determine_risk(self, distance: float, class_key: str, direction_en: str) -> str:
        """
        Determine risk level: DANGER (Red), WARNING (Orange/Yellow), or SAFE (Green).
        Critical objects (e.g. stairs, person in front) get increased sensitivity.
        """
        info = INDOOR_CLASSES.get(class_key)
        danger_dist = DIST_DANGER_THRESHOLD
        warning_dist = DIST_WARNING_THRESHOLD

        if info:
            danger_dist = max(danger_dist, info.min_safe_dist)
            if info.priority == 1: # High priority obstacle (stairs, fan, trash_can)
                danger_dist += 0.2
                warning_dist += 0.3

        if direction_en == "center":
            danger_dist += 0.15

        if distance <= danger_dist:
            return "DANGER"
        elif distance <= warning_dist:
            return "WARNING"
        else:
            return "SAFE"

    def compute_3d_coordinates(
        self,
        bbox: Tuple[int, int, int, int],
        distance: float,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float]:
        """
        Compute top-down radar coordinates (X_lateral, Z_depth) in meters.
        X > 0: Right, X < 0: Left, Z: Forward distance.
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        principal_x = frame_width / 2.0
        lateral_x = ((center_x - principal_x) * distance) / self.focal_length
        return lateral_x, distance

    def process_detection(
        self,
        class_key: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        timestamp: Optional[float] = None
    ) -> DetectedObject:
        """
        Assemble comprehensive DetectedObject with multi-feature distance,
        temporal stabilization, risk level, direction, and 3D position.
        """
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        info = INDOOR_CLASSES.get(class_key, INDOOR_CLASSES["obstacle"])

        # 1. Multi-feature geometric distance calculation
        raw_distance = self.estimate_distance(class_key, bbox, (frame_h, frame_w))

        # 2. Object-level temporal smoothing (removes frame-to-frame jitter)
        now = timestamp if timestamp is not None else time.time()
        distance = self._smooth_distance(class_key, bbox, raw_distance, now)

        dir_vi, dir_en, rel_x = self.determine_direction(bbox, frame_w)
        rel_y = y2 / float(frame_h)
        risk = self.determine_risk(distance, class_key, dir_en)
        coord_3d = self.compute_3d_coordinates(bbox, distance, frame_w, frame_h)

        return DetectedObject(
            class_key=class_key,
            name_vi=info.name_vi,
            name_en=info.name_en,
            confidence=confidence,
            bbox=bbox,
            distance=round(distance, 2),
            direction_vi=dir_vi,
            direction_en=dir_en,
            risk_level=risk,
            rel_x=rel_x,
            rel_y=rel_y,
            coord_3d=coord_3d
        )

