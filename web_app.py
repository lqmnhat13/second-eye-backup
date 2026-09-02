"""
FastAPI Web Application & Real-time API Server for Second Eye.
Provides low-latency WebSocket / REST streaming endpoints, live browser webcam inference,
2D spatial radar telemetry, and multi-modal audio alerts.
"""

import os
import io
import time
import base64
import cv2
import numpy as np
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from config import (
    INDOOR_CLASSES,
    DEFAULT_FOCAL_LENGTH,
    DIST_DANGER_THRESHOLD,
    DIST_WARNING_THRESHOLD
)
from detector import IndoorDetector
from alert_system import AlertManager
from audio_service import audio_service

app = FastAPI(title="Second Eye - Vision Assistant API", version="2.0.0")

# Enable CORS for local network and mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static file directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global AI instances
detector = IndoorDetector(model_name="yolov8n.pt", conf_threshold=0.35, focal_length=DEFAULT_FOCAL_LENGTH)
alert_manager = AlertManager(enable_local_audio=False)

class FramePayload(BaseModel):
    image: str  # Base64 data URL
    focal_length: Optional[float] = DEFAULT_FOCAL_LENGTH
    conf_threshold: Optional[float] = 0.35

class SettingsPayload(BaseModel):
    focal_length: Optional[float] = None
    conf_threshold: Optional[float] = None
    muted: Optional[bool] = None
    disabled_classes: Optional[List[str]] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main dashboard."""
    classes_list = [
        {
            "key": k,
            "name_vi": v.name_vi,
            "name_en": v.name_en,
            "real_height": v.real_height,
            "priority": v.priority,
            "min_safe_dist": v.min_safe_dist
        }
        for k, v in INDOOR_CLASSES.items()
    ]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "indoor_classes": classes_list,
            "default_focal": DEFAULT_FOCAL_LENGTH,
            "danger_thresh": DIST_DANGER_THRESHOLD,
            "warning_thresh": DIST_WARNING_THRESHOLD
        }
    )

@app.post("/api/detect_frame")
async def detect_frame(payload: FramePayload):
    """
    Process incoming frame from client webcam (Base64), run detection + distance estimation,
    and return telemetry, objects, alerts, and 3D radar coordinates.
    """
    t_start = time.time()
    
    # Update runtime parameters if provided
    if payload.focal_length and payload.focal_length != detector.distance_estimator.focal_length:
        detector.set_focal_length(payload.focal_length)
    if payload.conf_threshold:
        detector.conf_threshold = payload.conf_threshold

    # Decode base64 image
    try:
        data_str = payload.image
        if "," in data_str:
            data_str = data_str.split(",")[1]
        img_bytes = base64.b64decode(data_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"status": "error", "message": "Failed to decode image"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    # Run Detection & Distance Pipeline
    detected_objects = detector.detect(frame)

    # Process Voice Alerts Queue
    triggered_alerts = alert_manager.process_detections(detected_objects)

    inference_time = (time.time() - t_start) * 1000.0  # ms
    fps = 1000.0 / max(1.0, inference_time)

    # Prepare response object data
    objects_data = [obj.to_dict() for obj in detected_objects]

    return {
        "status": "success",
        "objects": objects_data,
        "alerts": triggered_alerts,
        "inference_ms": round(inference_time, 1),
        "fps": round(fps, 1),
        "danger_count": sum(1 for o in detected_objects if o.risk_level == "DANGER"),
        "warning_count": sum(1 for o in detected_objects if o.risk_level == "WARNING")
    }

@app.get("/api/classes")
async def get_classes():
    """Get metadata for all 15 indoor classes."""
    return {
        "classes": {
            k: {
                "id": v.id,
                "name_vi": v.name_vi,
                "name_en": v.name_en,
                "real_height": v.real_height,
                "priority": v.priority,
                "enabled": k in detector.enabled_classes
            }
            for k, v in INDOOR_CLASSES.items()
        }
    }

@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    """Update system settings (focal length, confidence, mute, enabled classes)."""
    if payload.focal_length is not None:
        detector.set_focal_length(payload.focal_length)
    if payload.conf_threshold is not None:
        detector.conf_threshold = payload.conf_threshold
    if payload.muted is not None:
        alert_manager.mute(payload.muted)
    if payload.disabled_classes is not None:
        for k in INDOOR_CLASSES.keys():
            detector.toggle_class(k, k not in payload.disabled_classes)

    return {"status": "ok", "message": "Settings updated successfully"}

@app.get("/api/tts")
async def text_to_speech(text: str):
    """Generate or retrieve pre-cached native Vietnamese MP3 speech."""
    from audio_service import audio_service
    audio_b64 = audio_service.get_audio_base64(text)
    if audio_b64:
        return {"status": "ok", "audio_base64": audio_b64}
    return JSONResponse({"status": "error", "message": "Failed to synthesize speech"}, status_code=500)

@app.get("/api/alerts_log")
async def get_alerts_log():
    """Fetch recent alert history."""
    return {"log": alert_manager.alert_log}

def get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def ensure_ssl_certs():
    if not (os.path.exists("cert.pem") and os.path.exists("key.pem")):
        import subprocess
        try:
            print("[SSL] Đang tạo chứng chỉ SSL bảo mật cho kết nối Camera di động...")
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", "key.pem", "-out", "cert.pem",
                "-days", "365", "-nodes", "-subj", "/CN=SecondEye"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[SSL] Đã tạo chứng chỉ SSL cert.pem & key.pem thành công.")
        except Exception as e:
            print(f"[SSL Error] Không thể tạo chứng chỉ tự động: {e}")

def find_free_port(start_port: int = 8000) -> int:
    import socket
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('0.0.0.0', port)) != 0:
                return port
        port += 1
    return start_port

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Second Eye Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--auto-port", action="store_true", help="Auto-select free port if occupied")
    parser.add_argument("--ssl", action="store_true", default=True, help="Enable HTTPS (Required for phone camera)")
    parser.add_argument("--no-ssl", action="store_false", dest="ssl", help="Disable HTTPS and run plain HTTP")
    args = parser.parse_args()

    target_port = args.port
    if args.auto_port:
        target_port = find_free_port(args.port)

    local_ip = get_local_ip()
    use_ssl = args.ssl

    if use_ssl:
        ensure_ssl_certs()
        if not (os.path.exists("cert.pem") and os.path.exists("key.pem")):
            print("[Warning] Không tìm thấy cert.pem / key.pem, chuyển về chế độ HTTP thông thường.")
            use_ssl = False

    proto = "https" if use_ssl else "http"
    print("\n" + "=" * 70)
    print("      SECOND EYE - HỆ THỐNG TRỢ LÝ THỊ GIÁC CHO NGƯỜI KHIẾM THỊ")
    print("=" * 70)
    print(f"💻 MÁY TÍNH:   {proto}://localhost:{target_port}")
    print(f"📱 ĐIỆN THOẠI: {proto}://{local_ip}:{target_port}")
    print("-" * 70)
    if use_ssl:
        print("📌 LƯU Ý KHI MỞ TRÊN ĐIỆN THOẠI (iOS Safari / Android Chrome):")
        print("   Vì dùng chứng chỉ bảo mật nội bộ, trình duyệt sẽ hỏi xác nhận 1 lần:")
        print("   -> Bấm 'Nâng cao' (Advanced) -> Chọn 'Tiếp tục truy cập' (Proceed).")
        print("   -> Sau đó bấm 'Bật Camera' để cấp quyền sử dụng camera điện thoại.")
    print("=" * 70 + "\n")

    if use_ssl:
        uvicorn.run("web_app:app", host=args.host, port=target_port,
                    ssl_keyfile="key.pem", ssl_certfile="cert.pem", reload=True)
    else:
        uvicorn.run("web_app:app", host=args.host, port=target_port, reload=True)

