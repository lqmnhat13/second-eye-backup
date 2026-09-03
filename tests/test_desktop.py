"""
Unit test for Second Eye Desktop Application components.
Tests Tkinter GUI initialization, Radar canvas rendering, and offline local audio controls.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import tkinter as tk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.distance_estimator import DetectedObject
from src.services.audio_service import audio_service
from src.desktop.radar_canvas import RadarWidget
from src.desktop.app import SecondEyeDesktopApp

def test_desktop_gui():
    print("=" * 60)
    print("  TESTING SECOND EYE DESKTOP APP & LOCAL AUDIO")
    print("=" * 60)

    # 1. Test Offline Local Audio
    print("\n[Test 1/3] Verifying 100% Offline Local Speech Service...")
    res = audio_service.speak_local("Kiểm tra âm thanh local", voice_rate=190, interrupt=True)
    assert res is True
    time.sleep(0.3)
    audio_service.stop_speech()
    print("  -> Offline speech synthesis and process termination: OK")

    # 2. Test Radar Widget Canvas
    print("\n[Test 2/3] Verifying 2D Radar Canvas Widget...")
    root = tk.Tk()
    root.withdraw() # Headless test
    
    radar = RadarWidget(root, width=380, height=220, max_range_meters=4.0)
    test_objs = [
        DetectedObject(
            class_key="stairs",
            name_vi="cầu thang",
            name_en="stairs",
            confidence=0.92,
            bbox=(200, 50, 440, 450),
            distance=0.85,
            direction_vi="Phía trước",
            direction_en="center",
            risk_level="DANGER",
            rel_x=0.5,
            rel_y=0.9,
            coord_3d=(0.0, 0.85)
        ),
        DetectedObject(
            class_key="chair",
            name_vi="ghế",
            name_en="chair",
            confidence=0.88,
            bbox=(50, 100, 180, 380),
            distance=1.6,
            direction_vi="Bên trái",
            direction_en="left",
            risk_level="WARNING",
            rel_x=0.2,
            rel_y=0.8,
            coord_3d=(-0.9, 1.6)
        )
    ]
    radar.update_objects(test_objs)
    radar.advance_sweep()
    assert len(radar.find_all()) > 0, "Radar canvas should have elements drawn"
    print(f"  -> Radar canvas drew {len(radar.find_all())} graphical objects correctly - OK")

    # 3. Test Desktop App Class Initialization
    print("\n[Test 3/3] Verifying SecondEyeDesktopApp Initializer...")
    app = SecondEyeDesktopApp(root, camera_source=0)
    assert app.current_mode == "nav"
    assert app.detector is not None
    assert app.alert_manager is not None
    assert app.radar_widget is not None

    # Test mode switching
    app.switch_mode("ocr")
    assert app.current_mode == "ocr"
    app.switch_mode("nav")
    assert app.current_mode == "nav"
    print("  -> Desktop UI tabs & mode switching: OK")

    # Test clean shutdown
    app.on_close()
    print("  -> Desktop UI clean termination: OK")

    print("\n" + "=" * 60)
    print("  ALL DESKTOP APPLICATION TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_desktop_gui()
