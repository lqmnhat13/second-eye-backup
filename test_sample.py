"""
Comprehensive Unit & Integration Test Suite for Second Eye.
Verifies 15 indoor classes, distance calculation mathematics, spatial zone mapping,
YOLO model inference, HUD rendering, and Vietnamese voice alert generation.
"""

import os
import sys
import numpy as np
import cv2

def run_tests():
    print("=" * 60)
    print("  RUNNING SECOND EYE SYSTEM VERIFICATION TESTS")
    print("=" * 60)

    # 1. Test Config & 15 Indoor Classes
    print("\n[Test 1/5] Verifying 15 Indoor Classes & Config...")
    from config import INDOOR_CLASSES, COCO_TO_INDOOR_MAP, DEFAULT_FOCAL_LENGTH
    assert len(INDOOR_CLASSES) == 15, f"Expected 15 indoor classes, found {len(INDOOR_CLASSES)}"
    for key, info in INDOOR_CLASSES.items():
        assert info.real_height > 0, f"Class {key} has invalid real_height"
        assert info.min_safe_dist > 0, f"Class {key} has invalid min_safe_dist"
        assert len(info.name_vi) > 0, f"Class {key} missing Vietnamese name"
    print("  -> Passed! 15 indoor classes validated successfully:")
    for i, (k, v) in enumerate(INDOOR_CLASSES.items(), 1):
        print(f"     {i:2d}. {v.name_vi.capitalize():<15} ({v.name_en:<12}) | Height: {v.real_height:.2f}m | Priority: {v.priority}")

    # 2. Test Distance Estimator & Spatial Math
    print("\n[Test 2/5] Verifying Distance Estimator & Spatial Zones...")
    from distance_estimator import DistanceEstimator
    estimator = DistanceEstimator(focal_length=650.0)

    # Test Distance calculation:
    # A chair (height 0.85m) taking 276px on a 480p frame with f=650 -> d = (650 * 0.85) / 276 = 2.0m
    dist_calc = estimator.estimate_distance("chair", (100, 100, 200, 376), (480, 640))
    expected_dist = (650.0 * 0.85) / 276.0
    assert abs(dist_calc - expected_dist) < 0.1, f"Expected ~{expected_dist:.2f}m, got {dist_calc:.2f}m"
    print(f"  -> Chair bbox (h=276px): Calculated Distance = {dist_calc:.2f}m (Expected: {expected_dist:.2f}m)")

    # Test Spatial Zones (Left, Center, Right)
    dir_vi_l, dir_en_l, _ = estimator.determine_direction((20, 100, 100, 300), 640)
    dir_vi_c, dir_en_c, _ = estimator.determine_direction((250, 100, 390, 300), 640)
    dir_vi_r, dir_en_r, _ = estimator.determine_direction((500, 100, 620, 300), 640)
    assert dir_en_l == "left", f"Expected 'left', got {dir_en_l}"
    assert dir_en_c == "center", f"Expected 'center', got {dir_en_c}"
    assert dir_en_r == "right", f"Expected 'right', got {dir_en_r}"
    print(f"  -> Direction Partitioning: Left='{dir_vi_l}', Center='{dir_vi_c}', Right='{dir_vi_r}' - OK")

    # Test Risk Levels
    risk_danger = estimator.determine_risk(0.7, "chair", "center")
    risk_warning = estimator.determine_risk(1.5, "chair", "center")
    risk_safe = estimator.determine_risk(3.0, "chair", "center")
    assert risk_danger == "DANGER"
    assert risk_warning == "WARNING"
    assert risk_safe == "SAFE"
    print(f"  -> Risk Levels: 0.7m={risk_danger}, 1.5m={risk_warning}, 3.0m={risk_safe} - OK")

    # 3. Test Alert Manager & Vietnamese Speech Formatting
    print("\n[Test 3/5] Verifying Alert Manager & Vietnamese Speech Engine...")
    from alert_system import AlertManager, format_distance_vi
    alert_mgr = AlertManager(enable_local_audio=False) # audio off for headless test

    assert format_distance_vi(0.8) == "không phẩy tám mét"
    assert format_distance_vi(1.0) == "1 mét"
    assert format_distance_vi(1.5) == "1 phẩy năm mét"
    assert format_distance_vi(2.3) == "2 phẩy ba mét"
    print("  -> Vietnamese Distance Speech formatting validated - OK")

    # Test DANGER Alert Phrase (critical close proximity: bbox height 450px on 480p -> ~0.94m)
    danger_obj = estimator.process_detection("stairs", 0.9, (200, 20, 440, 470), (480, 640))
    phrase_danger = alert_mgr.build_alert_phrase(danger_obj)
    print(f"  -> Generated DANGER Alert Phrase: \"{phrase_danger}\"")
    assert "nguy hiểm" in phrase_danger.lower() or "cảnh báo" in phrase_danger.lower()

    # Test WARNING Alert Phrase
    warning_obj = estimator.process_detection("chair", 0.85, (220, 100, 380, 380), (480, 640))
    phrase_warning = alert_mgr.build_alert_phrase(warning_obj)
    print(f"  -> Generated WARNING Alert Phrase: \"{phrase_warning}\"")
    assert "ghế" in phrase_warning.lower()
    
    alerts = alert_mgr.process_detections([danger_obj, warning_obj])
    assert len(alerts) >= 1
    print(f"  -> Alert Queue Processed: {len(alerts)} alerts triggered")

    # 4. Test YOLO Detector on Synthetic Indoor Scene
    print("\n[Test 4/5] Verifying IndoorDetector & YOLO Inference...")
    from detector import IndoorDetector
    detector = IndoorDetector(model_name="yolov8n.pt", conf_threshold=0.25)
    
    # Create synthetic test frame with simulated indoor shapes
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background floor & wall
    synthetic_frame[:240, :] = [40, 40, 45]
    synthetic_frame[240:, :] = [70, 70, 75]
    
    # Run detector
    detections = detector.detect(synthetic_frame)
    print(f"  -> Detection pipeline ran cleanly on frame without errors (Detections count: {len(detections)})")

    # Test HUD drawing
    annotated = detector.draw_hud(synthetic_frame, [danger_obj, warning_obj], fps=45.2)
    assert annotated.shape == synthetic_frame.shape
    os.makedirs("test_outputs", exist_ok=True)
    cv2.imwrite("test_outputs/annotated_hud_test.jpg", annotated)
    print("  -> HUD Annotation rendered and saved to 'test_outputs/annotated_hud_test.jpg'")

    # 5. Test Web App Endpoint Readiness
    print("\n[Test 5/5] Verifying Web App & API Endpoints...")
    from web_app import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    res_home = client.get("/")
    assert res_home.status_code == 200, f"GET / failed with {res_home.status_code}"
    print("  -> GET / (Dashboard UI HTML) status: 200 OK")

    res_classes = client.get("/api/classes")
    assert res_classes.status_code == 200
    classes_json = res_classes.json()
    assert len(classes_json["classes"]) == 15
    print(f"  -> GET /api/classes returned {len(classes_json['classes'])} classes: OK")

    # Test /api/detect_frame with a blank image
    import base64
    _, buffer = cv2.imencode('.jpg', synthetic_frame)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    res_detect = client.post("/api/detect_frame", json={"image": b64_str, "focal_length": 650.0})
    assert res_detect.status_code == 200
    detect_json = res_detect.json()
    assert detect_json["status"] == "success"
    print(f"  -> POST /api/detect_frame status: 200 OK | Inference Latency: {detect_json['inference_ms']}ms | FPS: {detect_json['fps']}")

    alert_mgr.stop()
    print("\n" + "=" * 60)
    print("  ALL 5 SYSTEM TESTS PASSED PERFECTLY! 100% SUCCESS")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
