"""
Object Detection Engine for Second Eye.
Wraps YOLO model and maps raw detections to 15 prioritized indoor classes,
combining object bounding boxes with metric distance estimation and HUD rendering.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO

from src.config import (
    INDOOR_CLASSES,
    COCO_TO_INDOOR_MAP,
    COLOR_DANGER,
    COLOR_WARNING,
    COLOR_SAFE,
    DEFAULT_FOCAL_LENGTH,
    DEFAULT_MODEL_PATH
)
from src.core.distance_estimator import DistanceEstimator, DetectedObject

def remove_accents_vi(text: str) -> str:
    """Helper to convert Vietnamese accents to ASCII for OpenCV putText."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return ascii_text.replace('đ', 'd').replace('Đ', 'D')

class IndoorDetector:
    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        focal_length: float = DEFAULT_FOCAL_LENGTH,
        device: Optional[str] = None
    ):
        """
        Initialize Indoor Detector.
        :param model_name: YOLO model weights (default: 'models/yolov8n.pt' or fallback to 'yolov8n.pt')
        :param conf_threshold: Confidence threshold [0.0 - 1.0]
        :param iou_threshold: IOU threshold for NMS
        :param focal_length: Initial focal length in pixels
        :param device: 'mps', 'cuda', or 'cpu' (auto-detected if None)
        """
        if model_name is None:
            if os.path.exists(DEFAULT_MODEL_PATH):
                model_name = DEFAULT_MODEL_PATH
            else:
                model_name = "yolov8n.pt"

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[IndoorDetector] Loading YOLO model '{model_name}' on device: {self.device}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.distance_estimator = DistanceEstimator(focal_length=focal_length)
        self.enabled_classes = set(INDOOR_CLASSES.keys())

    def set_focal_length(self, f: float):
        """Update focal length for distance estimation."""
        self.distance_estimator.update_focal_length(f)

    def toggle_class(self, class_key: str, enabled: bool):
        """Enable or disable specific class detection."""
        if class_key in INDOOR_CLASSES:
            if enabled:
                self.enabled_classes.add(class_key)
            else:
                self.enabled_classes.discard(class_key)

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Run object detection on an input BGR frame.
        Returns sorted list of DetectedObject (highest risk / closest distance first).
        """
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        
        # Inference with YOLO
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detected_objects: List[DetectedObject] = []

        if not results or len(results) == 0:
            return detected_objects

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detected_objects

        for box in boxes:
            cls_id = int(box.cls[0].item())
            raw_class_name = self.model.names.get(cls_id, "").lower()
            conf = float(box.conf[0].item())

            # Map to 15 indoor classes
            mapped_key = COCO_TO_INDOOR_MAP.get(raw_class_name)
            if not mapped_key:
                # Check if it directly matches an indoor class key
                if raw_class_name in INDOOR_CLASSES:
                    mapped_key = raw_class_name
                else:
                    continue  # Skip outdoor or unmapped objects

            if mapped_key not in self.enabled_classes:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            # Clamp coordinates to frame boundaries
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w - 1, x2))
            y2 = max(0, min(frame_h - 1, y2))

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            detected_obj = self.distance_estimator.process_detection(
                class_key=mapped_key,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                frame_shape=(frame_h, frame_w)
            )
            detected_objects.append(detected_obj)

        # Sort detections: DANGER first, then WARNING, then SAFE, ordered by ascending distance
        risk_priority = {"DANGER": 0, "WARNING": 1, "SAFE": 2}
        detected_objects.sort(key=lambda obj: (risk_priority.get(obj.risk_level, 3), obj.distance))

        return detected_objects

    def draw_hud(self, frame: np.ndarray, detected_objects: List[DetectedObject], fps: float = 0.0) -> np.ndarray:
        """
        Render a high-tech HUD overlay on the OpenCV frame.
        Includes color-coded bounding boxes, distance badges, directional zones, and status bar.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw subtle vertical zone guideline dividers
        z_left = int(w * 0.35)
        z_right = int(w * 0.65)
        cv2.line(annotated, (z_left, 0), (z_left, h), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(annotated, (z_right, 0), (z_right, h), (70, 70, 70), 1, cv2.LINE_AA)

        # Zone labels at bottom
        cv2.putText(annotated, "TRAI (LEFT)", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(annotated, "PHIA TRUOC (CENTER)", (z_left + 15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, "PHAI (RIGHT)", (z_right + 15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # Render each detected object
        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Select color based on risk level
            if obj.risk_level == "DANGER":
                color = COLOR_DANGER
                line_thickness = 3
            elif obj.risk_level == "WARNING":
                color = COLOR_WARNING
                line_thickness = 2
            else:
                color = COLOR_SAFE
                line_thickness = 2

            # Bounding box with corner accents
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 3:
                # Top-left corner
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, line_thickness + 2)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, line_thickness + 2)
                # Bottom-right corner
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, line_thickness + 2)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, line_thickness + 2)

            # Label text (clean ASCII for OpenCV renderer)
            clean_name = remove_accents_vi(obj.name_vi).upper()
            clean_dir = remove_accents_vi(obj.direction_vi)
            label = f"{clean_name} | {obj.distance:.1f}m ({clean_dir})"
            
            # Label background header
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            bg_y1 = max(0, y1 - lbl_h - 10)
            bg_y2 = y1
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + lbl_w + 12, bg_y2), color, -1)
            
            # Text color (white or black for contrast)
            text_color = (255, 255, 255) if obj.risk_level != "WARNING" else (0, 0, 0)
            cv2.putText(annotated, label, (x1 + 6, bg_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

        # Header status overlay (Top bar)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Top bar info
        danger_count = sum(1 for o in detected_objects if o.risk_level == "DANGER")
        status_text = f"SECOND EYE AI | FPS: {fps:.1f} | Objects: {len(detected_objects)}"
        cv2.putText(annotated, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if danger_count > 0:
            cv2.putText(annotated, f"CANH BAO: {danger_count} VAT CAN NGUY HIEM!", (w - 320, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        return annotated
