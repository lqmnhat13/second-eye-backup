"""
Demo script to run Second Eye AI pipeline on static sample images.
Saves annotated visual HUD and generates Vietnamese voice alerts.
"""

import os
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DEFAULT_MODEL_PATH
from src.core.detector import IndoorDetector
from src.services.alert_manager import AlertManager

def run_demo(
    image_path: str = str(DATA_DIR / "samples" / "indoor_demo.jpg"),
    output_path: str = str(DATA_DIR / "outputs" / "annotated_demo.jpg")
):
    if not os.path.exists(image_path):
        print(f"[Error] File not found: {image_path}")
        return

    print("=" * 65)
    print("      SECOND EYE - DEMO INFERENCE ON SAMPLE SCENE")
    print("=" * 65)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Không thể đọc ảnh: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"Loaded image: {image_path} ({w}x{h} px)")

    detector = IndoorDetector(model_name=DEFAULT_MODEL_PATH, conf_threshold=0.30, focal_length=650.0)
    alert_mgr = AlertManager(enable_local_audio=False)

    # 1. Detect objects & estimate metric distances
    detected_objects = detector.detect(frame)
    print(f"\n[Detected {len(detected_objects)} Indoor Objects]:")
    print(f"{'No':<3} | {'Class (VI)':<14} | {'Distance':<9} | {'Direction':<12} | {'Risk Level':<8} | {'Confidence'}")
    print("-" * 65)
    for idx, obj in enumerate(detected_objects, 1):
        print(f"{idx:<3} | {obj.name_vi.capitalize():<14} | {obj.distance:>5.2f}m   | {obj.direction_vi:<12} | {obj.risk_level:<8} | {obj.confidence*100:>4.1f}%")

    # 2. Process speech alert phrases
    alerts = alert_mgr.process_detections(detected_objects)
    print(f"\n[Generated Spoken Warnings ({len(alerts)})]:")
    for a in alerts:
        print(f"  🔊 [{a['risk_level']}] \"{a['text_vi']}\"")

    # 3. Render AR HUD
    annotated = detector.draw_hud(frame, detected_objects, fps=30.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved annotated result to: {output_path}")

    alert_mgr.stop()

if __name__ == "__main__":
    run_demo()
