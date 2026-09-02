"""
FastAPI Web Application & Real-time API Server for Second Eye.
Provides low-latency WebSocket / REST streaming endpoints, live browser webcam inference,
2D spatial radar telemetry, and multi-modal audio alerts.
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from pydantic import BaseModel

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    INDOOR_CLASSES,
    DEFAULT_FOCAL_LENGTH,
    DIST_DANGER_THRESHOLD,
    DIST_WARNING_THRESHOLD,
    SRC_DIR
)
from src.core.detector import IndoorDetector
from src.core.ocr_reader import ocr_reader
from src.services.alert_manager import AlertManager
from src.services.audio_service import audio_service

# Paths
WEB_DIR = SRC_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

app = FastAPI(title="Second Eye - Vision Assistant & Document Reader API", version="2.5.0")

# Enable CORS for local network and mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & Jinja2 Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize Core AI Detector & Alert Manager
detector = IndoorDetector(conf_threshold=0.35, focal_length=DEFAULT_FOCAL_LENGTH)
alert_manager = AlertManager(enable_local_audio=False)  # Web mode uses browser audio player

class DetectionRequest(BaseModel):
    image: str  # Base64 encoded JPEG/PNG
    focal_length: Optional[float] = None
    confidence_threshold: Optional[float] = None

class SettingsPayload(BaseModel):
    focal_length: Optional[float] = None
    conf_threshold: Optional[float] = None
    muted: Optional[bool] = None
    disabled_classes: Optional[List[str]] = None

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the main Second Eye Web Dashboard UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"classes": INDOOR_CLASSES}
    )

@app.get("/api/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "Second Eye Vision Assistant",
        "version": "2.0.0",
        "device": detector.device
    }

@app.post("/api/detect_frame")
async def detect_frame(payload: DetectionRequest):
    """
    Process incoming client webcam frame (base64).
    Runs YOLO inference, calculates 3D distances, evaluates risks,
    and returns detected objects + any active voice alert with MP3 audio.
    """
    start_time = time.time()

    if payload.focal_length is not None:
        detector.set_focal_length(payload.focal_length)
    if payload.confidence_threshold is not None:
        detector.conf_threshold = payload.confidence_threshold

    # Decode base64 image
    try:
        if "," in payload.image:
            image_data = payload.image.split(",")[1]
        else:
            image_data = payload.image
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"status": "error", "message": "Invalid image data"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Base64 decode failed: {str(e)}"}, status_code=400)

    # 1. Detect indoor objects & compute distances
    detected_objects = detector.detect(frame)

    # 2. Process non-blocking audio alerts
    triggered_alerts = alert_manager.process_detections(detected_objects)
    active_alert = triggered_alerts[0] if triggered_alerts else None

    # Calculate latency
    latency_ms = round((time.time() - start_time) * 1000, 1)
    fps = round(1000.0 / max(1.0, latency_ms), 1)

    return {
        "status": "success",
        "inference_ms": latency_ms,
        "fps": fps,
        "objects_count": len(detected_objects),
        "objects": [obj.to_dict() for obj in detected_objects],
        "active_alert": active_alert
    }

@app.post("/api/detect_file")
async def detect_uploaded_file(file: UploadFile = File(...), focal_length: float = Form(DEFAULT_FOCAL_LENGTH)):
    """Process an uploaded static indoor photo."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"status": "error", "message": "Failed to decode image file"}, status_code=400)

    detector.set_focal_length(focal_length)
    detected_objects = detector.detect(frame)
    annotated = detector.draw_hud(frame, detected_objects, fps=0.0)

    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    triggered_alerts = alert_manager.process_detections(detected_objects)

    return {
        "status": "success",
        "objects_count": len(detected_objects),
        "objects": [obj.to_dict() for obj in detected_objects],
        "annotated_image": annotated_b64,
        "active_alert": triggered_alerts[0] if triggered_alerts else None
    }

@app.get("/api/classes")
async def get_classes():
    """Retrieve 15 indoor class configurations & descriptions."""
    return {
        "classes": {
            k: {
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
    audio_b64 = audio_service.get_audio_base64(text)
    if audio_b64:
        return {"status": "ok", "audio_base64": audio_b64}
    return JSONResponse({"status": "error", "message": "Failed to synthesize speech"}, status_code=500)

@app.get("/api/alerts_log")
async def get_alerts_log():
    """Fetch recent alert history."""
    return {"log": alert_manager.alert_log}

class OCRRequest(BaseModel):
    image: str  # Base64 encoded image
    synthesize_audio: Optional[bool] = True

@app.post("/api/ocr/read_frame")
async def ocr_read_frame(payload: OCRRequest):
    """
    Perform high-accuracy OCR on camera frame and synthesize Vietnamese voice reading.
    Returns extracted text, lines, paragraphs, audio base64 clips, and annotated image.
    """
    start_time = time.time()
    try:
        if "," in payload.image:
            image_data = payload.image.split(",")[1]
        else:
            image_data = payload.image

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"status": "error", "message": "Invalid image format"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Decode error: {str(e)}"}, status_code=400)

    # Run OCR text extraction
    ocr_res = ocr_reader.extract_text(frame, render_annotated=True)

    # Synthesize audio for paragraphs
    audio_paragraphs = []
    full_audio_b64 = None
    if payload.synthesize_audio and ocr_res.full_text:
        # Full text audio
        full_audio_b64 = audio_service.get_audio_base64(ocr_res.full_text)
        # Paragraphs audio for synchronized highlighting
        audio_paragraphs = audio_service.synthesize_document_paragraphs(ocr_res.paragraphs)

    # Encode annotated image
    annotated_b64 = None
    if ocr_res.annotated_image is not None:
        _, buffer = cv2.imencode('.jpg', ocr_res.annotated_image)
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "status": "success",
        "latency_ms": latency_ms,
        "full_text": ocr_res.full_text,
        "paragraphs": ocr_res.paragraphs,
        "word_count": ocr_res.word_count,
        "avg_confidence": round(ocr_res.avg_confidence, 2),
        "lines": [
            {
                "text": l.text,
                "confidence": round(l.confidence, 2),
                "rect": list(l.rect),
                "bbox": l.bbox
            }
            for l in ocr_res.lines
        ],
        "full_audio_base64": full_audio_b64,
        "audio_paragraphs": audio_paragraphs,
        "annotated_image": annotated_b64
    }

@app.post("/api/ocr/read_file")
async def ocr_read_file(file: UploadFile = File(...), synthesize_audio: bool = Form(True)):
    """Upload a document photo and extract & read text with Vietnamese voice."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"status": "error", "message": "Failed to decode image file"}, status_code=400)

    ocr_res = ocr_reader.extract_text(frame, render_annotated=True)

    full_audio_b64 = None
    audio_paragraphs = []
    if synthesize_audio and ocr_res.full_text:
        full_audio_b64 = audio_service.get_audio_base64(ocr_res.full_text)
        audio_paragraphs = audio_service.synthesize_document_paragraphs(ocr_res.paragraphs)

    annotated_b64 = None
    if ocr_res.annotated_image is not None:
        _, buffer = cv2.imencode('.jpg', ocr_res.annotated_image)
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "full_text": ocr_res.full_text,
        "paragraphs": ocr_res.paragraphs,
        "word_count": ocr_res.word_count,
        "avg_confidence": round(ocr_res.avg_confidence, 2),
        "lines": [
            {
                "text": l.text,
                "confidence": round(l.confidence, 2),
                "rect": list(l.rect),
                "bbox": l.bbox
            }
            for l in ocr_res.lines
        ],
        "full_audio_base64": full_audio_b64,
        "audio_paragraphs": audio_paragraphs,
        "annotated_image": annotated_b64
    }

