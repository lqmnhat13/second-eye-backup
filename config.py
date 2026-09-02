"""
Configuration for Second Eye - Visually Impaired Indoor Assistance System.
Includes class definitions for 15 indoor objects, physical dimensions,
distance thresholds, and alert priority mapping.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ObjectClassInfo:
    id: int
    name_en: str
    name_vi: str
    real_height: float   # Real-world average height in meters
    real_width: float    # Real-world average width in meters
    priority: int        # 1 (Highest/Dangerous) to 3 (Normal)
    min_safe_dist: float # Danger threshold in meters
    coco_classes: list   # Associated standard COCO class names / indices

# 15 Indoor Classes Configuration
INDOOR_CLASSES: Dict[str, ObjectClassInfo] = {
    "stairs": ObjectClassInfo(
        id=0,
        name_en="stairs",
        name_vi="cầu thang",
        real_height=1.0,
        real_width=1.0,
        priority=1, # CRITICAL DANGER
        min_safe_dist=1.2,
        coco_classes=["stairs", "step"]
    ),
    "person": ObjectClassInfo(
        id=1,
        name_en="person",
        name_vi="người",
        real_height=1.65,
        real_width=0.50,
        priority=1, # HIGH
        min_safe_dist=1.0,
        coco_classes=["person"]
    ),
    "door": ObjectClassInfo(
        id=2,
        name_en="door",
        name_vi="cửa",
        real_height=2.0,
        real_width=0.85,
        priority=2, # MEDIUM (Navigation landmark)
        min_safe_dist=0.8,
        coco_classes=["door", "entrance"]
    ),
    "chair": ObjectClassInfo(
        id=3,
        name_en="chair",
        name_vi="ghế",
        real_height=0.85,
        real_width=0.50,
        priority=2, # HIGH (Trip hazard)
        min_safe_dist=0.8,
        coco_classes=["chair"]
    ),
    "couch": ObjectClassInfo(
        id=4,
        name_en="couch",
        name_vi="ghế sofa",
        real_height=0.85,
        real_width=1.80,
        priority=2, # MEDIUM
        min_safe_dist=0.9,
        coco_classes=["couch", "sofa"]
    ),
    "table": ObjectClassInfo(
        id=5,
        name_en="table",
        name_vi="bàn",
        real_height=0.75,
        real_width=1.20,
        priority=2, # HIGH (Waist-height obstacle)
        min_safe_dist=0.9,
        coco_classes=["dining table", "desk", "table"]
    ),
    "bed": ObjectClassInfo(
        id=6,
        name_en="bed",
        name_vi="giường",
        real_height=0.60,
        real_width=1.60,
        priority=2, # MEDIUM
        min_safe_dist=0.9,
        coco_classes=["bed"]
    ),
    "tv": ObjectClassInfo(
        id=7,
        name_en="tv",
        name_vi="tivi",
        real_height=0.60,
        real_width=1.00,
        priority=3, # LOW
        min_safe_dist=0.7,
        coco_classes=["tv", "monitor", "screen"]
    ),
    "refrigerator": ObjectClassInfo(
        id=8,
        name_en="refrigerator",
        name_vi="tủ lạnh",
        real_height=1.70,
        real_width=0.75,
        priority=2, # MEDIUM
        min_safe_dist=0.9,
        coco_classes=["refrigerator", "fridge"]
    ),
    "toilet": ObjectClassInfo(
        id=9,
        name_en="toilet",
        name_vi="bồn cầu",
        real_height=0.75,
        real_width=0.45,
        priority=2, # MEDIUM (Bathroom landmark)
        min_safe_dist=0.7,
        coco_classes=["toilet"]
    ),
    "sink": ObjectClassInfo(
        id=10,
        name_en="sink",
        name_vi="bồn rửa",
        real_height=0.85,
        real_width=0.60,
        priority=2, # MEDIUM
        min_safe_dist=0.7,
        coco_classes=["sink"]
    ),
    "trash_can": ObjectClassInfo(
        id=11,
        name_en="trash_can",
        name_vi="thùng rác",
        real_height=0.45,
        real_width=0.35,
        priority=1, # HIGH (Ground obstacle)
        min_safe_dist=0.7,
        coco_classes=["trash can", "wastebin", "bin"]
    ),
    "fan": ObjectClassInfo(
        id=12,
        name_en="fan",
        name_vi="quạt",
        real_height=1.10,
        real_width=0.45,
        priority=1, # HIGH (Blade/wire danger)
        min_safe_dist=0.8,
        coco_classes=["fan", "electric fan"]
    ),
    "bottle_cup": ObjectClassInfo(
        id=13,
        name_en="bottle_cup",
        name_vi="chai ly",
        real_height=0.22,
        real_width=0.08,
        priority=3, # MEDIUM (Spill hazard)
        min_safe_dist=0.5,
        coco_classes=["bottle", "cup", "wine glass"]
    ),
    "obstacle": ObjectClassInfo(
        id=14,
        name_en="obstacle",
        name_vi="vật cản",
        real_height=1.40,
        real_width=0.60,
        priority=1, # HIGH
        min_safe_dist=0.9,
        coco_classes=["obstacle", "pillar", "column", "potted plant", "suitcase"]
    )
}

# Mapping COCO names directly to our indoor classes for broad compatibility
COCO_TO_INDOOR_MAP: Dict[str, str] = {
    "person": "person",
    "chair": "chair",
    "couch": "couch",
    "dining table": "table",
    "bed": "bed",
    "tv": "tv",
    "laptop": "tv",
    "refrigerator": "refrigerator",
    "toilet": "toilet",
    "sink": "sink",
    "bottle": "bottle_cup",
    "cup": "bottle_cup",
    "wine glass": "bottle_cup",
    "bowl": "bottle_cup",
    "potted plant": "obstacle",
    "suitcase": "obstacle",
    "backpack": "obstacle",
    "handbag": "obstacle",
    "umbrella": "obstacle",
    "dog": "obstacle",
    "cat": "obstacle",
    "bird": "obstacle",
    "vase": "obstacle",
    "microwave": "obstacle",
    "oven": "obstacle",
    "toaster": "obstacle",
    "book": "obstacle",
    "clock": "obstacle",
    "stairs": "stairs",
    "door": "door",
    "trash can": "trash_can",
    "fan": "fan"
}

# Distance Thresholds (in meters)
DIST_DANGER_THRESHOLD = 1.0     # Red alert: Critical (< 1.0m)
DIST_WARNING_THRESHOLD = 2.0    # Yellow alert: Warning (1.0m - 2.0m)
DIST_SAFE_THRESHOLD = 3.5       # Green: Safe (> 2.0m)

# Spatial Direction Boundaries (Relative horizontal position in frame [0.0, 1.0])
ZONE_LEFT_MAX = 0.35            # 0.00 - 0.35: "Bên trái"
ZONE_CENTER_MAX = 0.65          # 0.35 - 0.65: "Phía trước"
                                # 0.65 - 1.00: "Bên phải"

# Camera Calibration Default Parameters
DEFAULT_FOCAL_LENGTH = 650.0    # Standard webcam focal length in pixels (for 640x480)
DEFAULT_CAMERA_HEIGHT = 1.2     # Handheld / Chest-mounted camera height (meters)

# Alert Cooldowns (seconds) - Chống đọc liên tục dồn dập
GLOBAL_ALERT_COOLDOWN = 4.0     # Khoảng nghỉ tối thiểu giữa 2 lần phát giọng nói bất kỳ
ALERT_REPEAT_COOLDOWN = 5.5     # Khoảng nghỉ tối thiểu trước khi đọc lại cùng 1 vật thể
DANGER_REPEAT_COOLDOWN = 2.5    # Khoảng nghỉ cho vật thể nguy hiểm khẩn cấp (< 1m)

# Visual HUD Colors (BGR format for OpenCV, Hex for Web)
COLOR_DANGER = (0, 0, 255)       # Red
COLOR_WARNING = (0, 180, 255)    # Orange/Yellow
COLOR_SAFE = (0, 220, 0)         # Green

HEX_DANGER = "#ef4444"
HEX_WARNING = "#f59e0b"
HEX_SAFE = "#10b981"
