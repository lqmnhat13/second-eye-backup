"""
Second Eye - Standalone Desktop Application (OpenCV HUD).
Runs real-time indoor obstacle detection, distance estimation, and voice warnings on local webcam or video file.

Controls:
- 'q' or 'ESC': Quit
- 'm': Toggle Mute / Unmute voice alerts
- 's': Save screenshot snapshot to data/outputs/
- '+': Increase Focal Length (Calibrate distance)
- '-': Decrease Focal Length
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_FOCAL_LENGTH, DEFAULT_MODEL_PATH, DATA_DIR
from src.core.detector import IndoorDetector
from src.services.alert_manager import AlertManager

def main():
    parser = argparse.ArgumentParser(description="Second Eye - Desktop CV Application")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path/URL to video stream")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="YOLO model path/name")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--focal", type=float, default=DEFAULT_FOCAL_LENGTH, help="Initial camera focal length")
    parser.add_argument("--no-audio", action="store_true", help="Disable local speech synthesis")
    args = parser.parse_args()

    # Parse camera source
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 65)
    print("      SECOND EYE - HỆ THỐNG HỖ TRỢ NGƯỜI KHIẾM THỊ (DESKTOP)")
    print("=" * 65)
    print(f"Source: {source}")
    print(f"Model: {args.model}")
    print(f"Initial Focal Length: {args.focal}")
    print("Controls:")
    print("  [Q/ESC] : Thoát ứng dụng")
    print("  [M]     : Bật / Tắt âm thanh cảnh báo (Mute/Unmute)")
    print("  [S]     : Chụp ảnh lưu trữ (Snapshot)")
    print("  [+] / [-]: Tăng / Giảm tiêu cự hiệu chuẩn khoảng cách")
    print("=" * 65)

    # Initialize modules
    detector = IndoorDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        focal_length=args.focal
    )
    alert_mgr = AlertManager(enable_local_audio=not args.no_audio)

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Không thể mở nguồn video/camera: {source}")
        print("\n💡 Gợi ý:")
        print("  - Nếu dùng camera MacBook: Hãy kiểm tra quyền Camera trong System Settings -> Privacy & Security -> Camera.")
        print("  - Nếu dùng iPhone qua Continuity: Thử chạy với --source 1")
        print("  - Nếu dùng IP Webcam: Nhập URL dạng: --source http://<IP>:8080/video")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    fps = 30.0
    is_muted = False
    current_focal = args.focal
    outputs_dir = DATA_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] Đã xem hết video hoặc mất tín hiệu camera.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # 1. Detect 15 indoor classes with metric distance
            detected_objects = detector.detect(frame)

            # 2. Process audio alerts
            alert_mgr.process_detections(detected_objects)

            # 3. Draw HUD overlay
            annotated_frame = detector.draw_hud(frame, detected_objects, fps=fps)

            # Display Mute status & Focal length on frame
            mute_str = "MUTED" if is_muted else "AUDIO ON"
            color_mute = (0, 0, 255) if is_muted else (0, 255, 0)
            cv2.putText(annotated_frame, f"Audio: {mute_str} | Focal: {current_focal:.0f}px", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_mute, 1, cv2.LINE_AA)

            # Show window
            cv2.imshow("Second Eye - Indoor Vision Assistant", annotated_frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
            elif key == ord('m') or key == ord('M'):
                is_muted = not is_muted
                alert_mgr.mute(is_muted)
                print(f"[Audio] Trạng thái âm thanh: {'MUTED (Tắt tiếng)' if is_muted else 'UNMUTED (Bật tiếng)'}")
            elif key == ord('s') or key == ord('S'):
                filename = str(outputs_dir / f"snapshot_{int(time.time())}.jpg")
                cv2.imwrite(filename, annotated_frame)
                print(f"[Saved] Đã lưu ảnh chụp: {filename}")
            elif key == ord('t') or key == ord('T'):
                print("\n[OCR] Đang quét và phân tích văn bản trong khung hình...")
                from src.core.ocr_reader import ocr_reader
                ocr_res = ocr_reader.extract_text(frame, render_annotated=True)
                if ocr_res.full_text.strip():
                    print(f"[OCR Thành Công] Nhận diện được {ocr_res.word_count} từ:")
                    print("-" * 50)
                    for idx, p in enumerate(ocr_res.paragraphs, 1):
                        print(f"  {idx}. {p}")
                    print("-" * 50)
                    # Speak out loud using local voice
                    from src.services.audio_service import audio_service
                    audio_service.speak_local(ocr_res.full_text)
                else:
                    print("[OCR] Không tìm thấy văn bản rõ ràng trong khung hình.")
                    from src.services.audio_service import audio_service
                    audio_service.speak_local("Không tìm thấy văn bản rõ ràng.")
            elif key == ord('+') or key == ord('='):
                current_focal += 25.0
                detector.set_focal_length(current_focal)
                print(f"[Calib] Tăng tiêu cự: {current_focal:.1f}px")
            elif key == ord('-') or key == ord('_'):
                current_focal = max(100.0, current_focal - 25.0)
                detector.set_focal_length(current_focal)
                print(f"[Calib] Giảm tiêu cự: {current_focal:.1f}px")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_mgr.stop()
        print("[Second Eye] Ứng dụng đã kết thúc an toàn.")

if __name__ == "__main__":
    main()
