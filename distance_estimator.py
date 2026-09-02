"""
Distance and Spatial Orientation Estimation Module for Second Eye.
Uses Pinhole Camera Geometry and Object Physical Priors to compute metric distances
and 3D coordinates (X, Z) for spatial radar mapping.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from config import (
    INDOOR_CLASSES,
    DIST_DANGER_THRESHOLD,
    DIST_WARNING_THRESHOLD,
    ZONE_LEFT_MAX,
    ZONE_CENTER_MAX,
    DEFAULT_FOCAL_LENGTH
)

@dataclass
class DetectedObject:
    class_key: str
    name_vi: str
    name_en: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    distance: float                  # Estimated distance in meters
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
    def __init__(self, focal_length: float = DEFAULT_FOCAL_LENGTH):
        """
        Initialize Distance Estimator.
        :param focal_length: Camera focal length in pixels (default: 650.0 for 640x480)
        """
        self.focal_length = focal_length

    def update_focal_length(self, new_focal_length: float):
        """Update focal length calibration value."""
        self.focal_length = max(100.0, float(new_focal_length))

    def estimate_distance(self, class_key: str, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> float:
        """
        Calculate metric distance (meters) using triangle similarity model.
        d = (focal_length * real_height) / bbox_height_in_pixels
        """
        x1, y1, x2, y2 = bbox
        bbox_height = max(1, y2 - y1)
        bbox_width = max(1, x2 - x1)
        
        info = INDOOR_CLASSES.get(class_key)
        if not info:
            real_h = 1.0
            real_w = 0.5
        else:
            real_h = info.real_height
            real_w = info.real_width

        # Primary distance estimation using physical height (robust against 3D rotations)
        dist_h = (self.focal_length * real_h) / float(bbox_height)
        dist_w = (self.focal_length * real_w) / float(bbox_width)
        
        # Check if object is cropped by the vertical frame edges (y1 near 0 or y2 near frame_h)
        is_vertically_cropped = (y1 <= 2) or (y2 >= frame_shape[0] - 2)
        
        if is_vertically_cropped:
            # If cropped vertically, object might be very close; take minimum conservative distance
            distance = min(dist_h, dist_w)
        else:
            # Standard uncropped object -> reliable height estimation
            distance = dist_h

        # Physical clamping (0.25m minimum to 12.0m maximum indoor range)
        return float(np.clip(distance, 0.25, 12.0))

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
            if info.priority == 1: # High priority obstacle (e.g., stairs, fan)
                danger_dist += 0.2
                warning_dist += 0.3

        # Center objects are directly in path -> slightly higher danger sensitivity
        if direction_en == "center":
            danger_dist += 0.15

        if distance <= danger_dist:
            return "DANGER"
        elif distance <= warning_dist:
            return "WARNING"
        else:
            return "SAFE"

    def compute_3d_coordinates(self, bbox: Tuple[int, int, int, int], distance: float, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """
        Compute top-down radar coordinates (X_lateral, Z_depth) in meters.
        X > 0: Right, X < 0: Left, Z: Forward distance.
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        principal_x = frame_width / 2.0
        
        # Lateral X offset in meters
        lateral_x = ((center_x - principal_x) * distance) / self.focal_length
        return lateral_x, distance

    def process_detection(
        self,
        class_key: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int]
    ) -> DetectedObject:
        """
        Assemble comprehensive DetectedObject with distance, risk level, direction, and 3D position.
        """
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        
        info = INDOOR_CLASSES.get(class_key, INDOOR_CLASSES["obstacle"])
        distance = self.estimate_distance(class_key, bbox, (frame_h, frame_w))
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
            distance=distance,
            direction_vi=dir_vi,
            direction_en=dir_en,
            risk_level=risk,
            rel_x=rel_x,
            rel_y=rel_y,
            coord_3d=coord_3d
        )
