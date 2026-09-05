"""
Second Eye - Standalone Native Desktop Application.
100% Offline, Local Computer Vision, 2D Radar, and OCR Reader with macOS Vietnamese speech.
High-Contrast, Accessible Dark Theme with Custom Styled Widgets.
"""

import sys
import os
import time
import math
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_FOCAL_LENGTH,
    DEFAULT_MODEL_PATH,
    DATA_DIR,
    INDOOR_CLASSES
)
from src.core.detector import IndoorDetector
from src.core.distance_estimator import DetectedObject
from src.core.ocr_reader import OCRReader, OCRResult
from src.services.audio_service import audio_service
from src.services.alert_manager import AlertManager
from src.desktop.radar_canvas import RadarWidget

# Class Icon Mapping for Objects Card List
CLASS_ICONS = {
    "stairs": "🪜",
    "person": "🚶",
    "door": "🚪",
    "chair": "🪑",
    "couch": "🛋️",
    "table": "🪵",
    "bed": "🛏️",
    "tv": "📺",
    "refrigerator": "🧊",
    "toilet": "🚽",
    "sink": "🚰",
    "trash_can": "🗑️",
    "fan": "🪭",
    "bottle_cup": "🥤",
    "obstacle": "⚠️"
}

class PillButton(tk.Label):
    """
    Custom Modern Pill Button built on tk.Label to ensure consistent,
    vibrant dark-theme styling on macOS without Aqua native button glitches.
    """
    def __init__(
        self,
        master,
        text: str,
        command=None,
        bg: str = "#1e293b",
        fg: str = "#f8fafc",
        hover_bg: str = "#334155",
        font=("Arial", 10, "bold"),
        padx: int = 12,
        pady: int = 6,
        **kwargs
    ):
        super().__init__(
            master,
            text=text,
            bg=bg,
            fg=fg,
            font=font,
            padx=padx,
            pady=pady,
            cursor="hand2",
            relief="flat",
            **kwargs
        )
        self.command = command
        self.default_bg = bg
        self.hover_bg = hover_bg
        self.default_fg = fg
        self.is_active = False

        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_click(self, event=None):
        if self.command:
            self.command()

    def _on_enter(self, event=None):
        if not self.is_active:
            self.config(bg=self.hover_bg)

    def _on_leave(self, event=None):
        if not self.is_active:
            self.config(bg=self.default_bg)

    def set_active(self, active: bool, active_bg: str = "#0284c7", active_fg: str = "#ffffff"):
        self.is_active = active
        if active:
            self.config(bg=active_bg, fg=active_fg)
        else:
            self.config(bg=self.default_bg, fg=self.default_fg)

    def update_style(self, bg: str, fg: str, hover_bg: str):
        self.default_bg = bg
        self.default_fg = fg
        self.hover_bg = hover_bg
        if not self.is_active:
            self.config(bg=bg, fg=fg)


class ThreadedCamera:
    """Threaded camera grabber to eliminate I/O blocking from cv2.VideoCapture."""
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.open(source)

    def open(self, source=0):
        self.stop()
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened() and source != 0:
            self.cap = cv2.VideoCapture(0)
            self.source = 0

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        return False

    def _capture_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.2)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.frame = None


class InferenceWorker:
    """Decoupled background worker running YOLO inference and audio alert processing asynchronously."""
    def __init__(self, detector: IndoorDetector, alert_manager: AlertManager):
        self.detector = detector
        self.alert_manager = alert_manager
        self.running = True
        self.frame_to_process: Optional[np.ndarray] = None
        self.latest_detections: List[DetectedObject] = []
        self.last_detection_time: float = time.time()
        self.lock = threading.Lock()
        self.has_new_frame = threading.Event()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame: np.ndarray):
        with self.lock:
            # Independent copy to eliminate frame buffer race conditions
            self.frame_to_process = frame.copy()
        self.has_new_frame.set()

    def get_latest_detections(self) -> List[DetectedObject]:
        with self.lock:
            # If no inference has completed within 0.8s, discard stale detections
            # to guarantee bounding boxes and radar never freeze on screen
            if time.time() - self.last_detection_time > 0.8:
                return []
            return list(self.latest_detections)

    def _worker_loop(self):
        while self.running:
            got_signal = self.has_new_frame.wait(timeout=0.05)
            if not self.running:
                break
            if not got_signal:
                continue

            self.has_new_frame.clear()
            with self.lock:
                frame = self.frame_to_process
                self.frame_to_process = None

            if frame is None:
                continue

            try:
                # Fast inference with imgsz=480 (~18-25ms on Apple Silicon MPS)
                detections = self.detector.detect(frame)
                with self.lock:
                    self.latest_detections = detections
                    self.last_detection_time = time.time()

                # Fast offline alert processing & sound debouncing
                self.alert_manager.process_detections(detections)
            except Exception as e:
                import traceback
                print(f"[InferenceWorker] Detection error: {e}")
                traceback.print_exc()

    def stop(self):
        self.running = False
        self.has_new_frame.set()


class SecondEyeDesktopApp:
    def __init__(self, root: tk.Tk, camera_source: int = 0):
        self.root = root
        self.root.title("Second Eye - Trợ Lý Thị Giác & Đọc Văn Bản (Desktop 100% Offline)")

        # Screen-aware responsive window sizing
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1240, screen_w - 40)
        win_h = min(730, screen_h - 80)
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(1040, 640)
        self.root.configure(bg="#090d16")

        # Application State
        self.camera_source = camera_source
        self.threaded_cam: Optional[ThreadedCamera] = None
        self.camera_running = False
        self.current_mode = "nav"  # "nav" (Obstacle & Radar) or "ocr" (Document Reader)
        self.focal_length = DEFAULT_FOCAL_LENGTH
        self.voice_enabled = True
        self.font_size = 18
        self.fps = 0.0
        self._prev_frame_time = time.time()
        self._last_card_update = 0.0
        self.last_frame: Optional[np.ndarray] = None
        self.current_detections: List[DetectedObject] = []
        self.ocr_result: Optional[OCRResult] = None
        self.is_ocr_reading_paused = False

        # Core Engines (100% Local)
        print("[DesktopApp] Khởi tạo mô hình AI và dịch vụ âm thanh local...")
        self.detector = IndoorDetector(
            model_name=DEFAULT_MODEL_PATH,
            conf_threshold=0.35,
            focal_length=None
        )
        # Synchronize focal length with saved calibration if present
        self.focal_length = self.detector.distance_estimator.focal_length

        self.alert_manager = AlertManager(enable_local_audio=True)
        self.ocr_reader = OCRReader(gpu=False)
        self.inference_worker = InferenceWorker(self.detector, self.alert_manager)

        # Build GUI
        self._build_header()
        self._build_main_workspace()
        self._build_footer()
        self._bind_shortcuts()

        # Start Camera & Loops
        self._start_camera()
        self._schedule_sweep_animation()

        # Announce app ready
        if self.voice_enabled:
            audio_service.speak_local("Hệ thống trợ lý thị giác Second Eye đã sẵn sàng.", voice_rate=180)

        # Handle clean window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # HEADER BAR
    # ------------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self.root, bg="#0d1424", height=48, padx=14, pady=8)
        header.pack(fill="x", side="top")

        # Left: Brand Icon & Title
        brand_box = tk.Frame(header, bg="#0d1424")
        brand_box.pack(side="left")

        tk.Label(brand_box, text="👁️", font=("Arial", 18), bg="#0d1424").pack(side="left", padx=(0, 8))

        lbl_title = tk.Label(
            brand_box,
            text="SECOND EYE",
            font=("Arial", 13, "bold"),
            fg="#38bdf8",
            bg="#0d1424"
        )
        lbl_title.pack(side="left", padx=(0, 6))

        lbl_badge = tk.Label(
            brand_box,
            text="100% OFFLINE LOCAL",
            font=("Arial", 8, "bold"),
            fg="#10b981",
            bg="#162032",
            padx=6,
            pady=2
        )
        lbl_badge.pack(side="left")

        # Center: Mode Switcher Tabs (Pill Buttons)
        mode_box = tk.Frame(header, bg="#0d1424")
        mode_box.pack(side="left", expand=True)

        self.btn_nav_mode = PillButton(
            mode_box,
            text="🧭  1. Tránh Vật Cản & Radar",
            command=lambda: self.switch_mode("nav"),
            bg="#0284c7",
            fg="#ffffff",
            hover_bg="#0369a1",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=5
        )
        self.btn_nav_mode.is_active = True
        self.btn_nav_mode.pack(side="left", padx=5)

        self.btn_ocr_mode = PillButton(
            mode_box,
            text="📖  2. Đọc Văn Bản OCR",
            command=lambda: self.switch_mode("ocr"),
            bg="#162032",
            fg="#94a3b8",
            hover_bg="#1e293b",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=5
        )
        self.btn_ocr_mode.pack(side="left", padx=5)

        # Right: FPS Counter & Audio Toggle
        right_box = tk.Frame(header, bg="#0d1424")
        right_box.pack(side="right")

        self.lbl_fps = tk.Label(
            right_box,
            text="FPS: 0.0",
            font=("Courier", 10, "bold"),
            fg="#38bdf8",
            bg="#162032",
            padx=8,
            pady=4
        )
        self.lbl_fps.pack(side="left", padx=(0, 8))

        self.btn_voice = PillButton(
            right_box,
            text="🔊 Giọng nói: BẬT (V)",
            command=self.toggle_voice,
            bg="#059669",
            fg="#ffffff",
            hover_bg="#047857",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=4
        )
        self.btn_voice.pack(side="left")

    # ------------------------------------------------------------------
    # MAIN WORKSPACE LAYOUT
    # ------------------------------------------------------------------
    def _build_main_workspace(self):
        self.workspace = tk.Frame(self.root, bg="#090d16", padx=10, pady=8)
        self.workspace.pack(fill="both", expand=True)

        # LEFT COLUMN: Camera Feed & Video Controls
        self.left_col = tk.Frame(self.workspace, bg="#0d1424", highlightthickness=1, highlightbackground="#1e293b")
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Viewport Header at TOP
        vp_header = tk.Frame(self.left_col, bg="#111a2e", padx=10, pady=6)
        vp_header.pack(fill="x", side="top")

        self.lbl_viewport_title = tk.Label(
            vp_header,
            text="Camera Trực Tiếp & Phân Tích AI 15 Lớp Vật Thể",
            font=("Arial", 10, "bold"),
            fg="#f8fafc",
            bg="#111a2e"
        )
        self.lbl_viewport_title.pack(side="left")

        self.lbl_camera_status = tk.Label(
            vp_header,
            text="● Camera Online",
            font=("Arial", 9, "bold"),
            fg="#10b981",
            bg="#111a2e"
        )
        self.lbl_camera_status.pack(side="right")

        # Toolbar at BOTTOM (Packed FIRST to guarantee it is NEVER clipped!)
        toolbar = tk.Frame(self.left_col, bg="#111a2e", padx=8, pady=6)
        toolbar.pack(fill="x", side="bottom")

        self.btn_cam_toggle = PillButton(
            toolbar,
            text="⏹ Dừng Cam (Space)",
            command=self.toggle_camera,
            bg="#1e293b",
            fg="#ffffff",
            hover_bg="#334155",
            font=("Arial", 9, "bold"),
            padx=9,
            pady=4
        )
        self.btn_cam_toggle.pack(side="left", padx=3)

        btn_cam_switch = PillButton(
            toolbar,
            text="🔄 Đổi Cam (M)",
            command=self.switch_camera_source,
            bg="#1e293b",
            fg="#ffffff",
            hover_bg="#334155",
            font=("Arial", 9, "bold"),
            padx=9,
            pady=4
        )
        btn_cam_switch.pack(side="left", padx=3)

        btn_open_file = PillButton(
            toolbar,
            text="📁 Mở File (O)",
            command=self.open_image_file,
            bg="#1e293b",
            fg="#ffffff",
            hover_bg="#334155",
            font=("Arial", 9, "bold"),
            padx=9,
            pady=4
        )
        btn_open_file.pack(side="left", padx=3)

        btn_snapshot = PillButton(
            toolbar,
            text="📸 Chụp (S)",
            command=self.save_snapshot,
            bg="#1e293b",
            fg="#ffffff",
            hover_bg="#334155",
            font=("Arial", 9, "bold"),
            padx=9,
            pady=4
        )
        btn_snapshot.pack(side="left", padx=3)

        # Focal Length controls on Right of Toolbar
        focal_box = tk.Frame(toolbar, bg="#111a2e")
        focal_box.pack(side="right")

        PillButton(
            focal_box, text="🎯 Hiệu Chuẩn", command=self.open_calibration_dialog,
            bg="#0284c7", fg="#ffffff", hover_bg="#0369a1", font=("Arial", 8, "bold"), padx=8, pady=2
        ).pack(side="left", padx=(0, 6))

        tk.Label(focal_box, text="Tiêu cự:", font=("Arial", 9), fg="#94a3b8", bg="#111a2e").pack(side="left", padx=2)

        PillButton(
            focal_box, text="-", command=lambda: self.adjust_focal_length(-25),
            bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 9, "bold"), padx=7, pady=2
        ).pack(side="left")

        self.lbl_focal_val = tk.Label(
            focal_box, text=f"{self.focal_length:.0f}px", font=("Courier", 9, "bold"), fg="#38bdf8", bg="#111a2e", padx=4
        )
        self.lbl_focal_val.pack(side="left")

        PillButton(
            focal_box, text="+", command=lambda: self.adjust_focal_length(25),
            bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 9, "bold"), padx=7, pady=2
        ).pack(side="left")

        # Video Canvas in MIDDLE (Fills available space)
        self.video_label = tk.Label(self.left_col, bg="#050811")
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

        # RIGHT COLUMN: Dynamic Panel (Width 440px)
        self.right_col = tk.Frame(self.workspace, bg="#090d16", width=440)
        self.right_col.pack(side="right", fill="both")
        self.right_col.pack_propagate(False)

        # Build Sub-panels
        self._build_nav_panel()
        self._build_ocr_panel()

        # Show Nav Panel initially
        self.panel_nav.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # MODE 1: NAVIGATION & RADAR PANEL
    # ------------------------------------------------------------------
    def _build_nav_panel(self):
        self.panel_nav = tk.Frame(self.right_col, bg="#090d16")

        # 1. Active Threat Alert Banner (Large, High-Contrast Card)
        self.banner_frame = tk.Frame(
            self.panel_nav,
            bg="#0f172a",
            highlightthickness=2,
            highlightbackground="#10b981",
            padx=12,
            pady=8
        )
        self.banner_frame.pack(fill="x", pady=(0, 8))

        self.lbl_banner_icon = tk.Label(self.banner_frame, text="🛡️", font=("Arial", 22), bg="#0f172a")
        self.lbl_banner_icon.pack(side="left", padx=(0, 10))

        banner_text_box = tk.Frame(self.banner_frame, bg="#0f172a")
        banner_text_box.pack(side="left", fill="x", expand=True)

        self.lbl_banner_title = tk.Label(
            banner_text_box,
            text="LỐI ĐI AN TOÀN",
            font=("Arial", 11, "bold"),
            fg="#10b981",
            bg="#0f172a",
            anchor="w",
            wraplength=340
        )
        self.lbl_banner_title.pack(fill="x")

        self.lbl_banner_desc = tk.Label(
            banner_text_box,
            text="Không phát hiện vật cản nguy hiểm ở cự ly gần.",
            font=("Arial", 9),
            fg="#cbd5e1",
            bg="#0f172a",
            anchor="w",
            wraplength=340
        )
        self.lbl_banner_desc.pack(fill="x")

        # 2. 2D Spatial Top-Down Radar Container
        radar_container = tk.Frame(self.panel_nav, bg="#0d1424", highlightthickness=1, highlightbackground="#1e293b", padx=6, pady=6)
        radar_container.pack(fill="x", pady=(0, 8))

        radar_title_box = tk.Frame(radar_container, bg="#0d1424")
        radar_title_box.pack(fill="x", pady=(0, 4))
        tk.Label(radar_title_box, text="🧭 RADAR KHÔNG GIAN 2D (TOP-DOWN 4.0M)", font=("Arial", 9, "bold"), fg="#38bdf8", bg="#0d1424").pack(side="left")
        tk.Label(radar_title_box, text="Thời gian thực", font=("Arial", 8), fg="#64748b", bg="#0d1424").pack(side="right")

        self.radar_widget = RadarWidget(radar_container, width=420, height=210, max_range_meters=4.0)
        self.radar_widget.pack(fill="x")

        # 3. Detected Objects List (Custom Dark Card Container - NO WHITE TREEVIEW!)
        self.det_container = tk.Frame(self.panel_nav, bg="#0d1424", highlightthickness=1, highlightbackground="#1e293b", padx=8, pady=6)
        self.det_container.pack(fill="both", expand=True)

        tk.Label(
            self.det_container,
            text="📦 VẬT THỂ TRONG PHÒNG",
            font=("Arial", 9, "bold"),
            fg="#e2e8f0",
            bg="#0d1424"
        ).pack(anchor="w", pady=(0, 4))

        # Scrollable container for object cards
        self.cards_scroll_frame = tk.Frame(self.det_container, bg="#0d1424")
        self.cards_scroll_frame.pack(fill="both", expand=True)

        # Initial Empty State Card
        self._render_empty_cards_state()

    def _render_empty_cards_state(self):
        for widget in self.cards_scroll_frame.winfo_children():
            widget.destroy()

        empty_box = tk.Frame(self.cards_scroll_frame, bg="#111a2e", padx=12, pady=16)
        empty_box.pack(fill="x", pady=4)
        tk.Label(
            empty_box,
            text="🛡️  Không gian quang đãng",
            font=("Arial", 10, "bold"),
            fg="#10b981",
            bg="#111a2e"
        ).pack(anchor="w")
        tk.Label(
            empty_box,
            text="Chưa phát hiện vật cản trong cự ly 4 mét.",
            font=("Arial", 8),
            fg="#64748b",
            bg="#111a2e"
        ).pack(anchor="w")

    def _update_objects_table(self, objects: List[DetectedObject]):
        """Render detected objects as modern, high-contrast dark cards."""
        for widget in self.cards_scroll_frame.winfo_children():
            widget.destroy()

        if not objects:
            self._render_empty_cards_state()
            return

        for obj in objects[:5]: # Show top 5 closest/hazardous obstacles
            icon = CLASS_ICONS.get(obj.class_key, "📦")
            is_danger = obj.risk_level == "DANGER"
            is_warning = obj.risk_level == "WARNING"

            card_border = "#ef4444" if is_danger else ("#f59e0b" if is_warning else "#1e293b")
            risk_badge_bg = "#dc2626" if is_danger else ("#d97706" if is_warning else "#16a34a")
            risk_text = "NGUY HIỂM" if is_danger else ("CẢNH BÁO" if is_warning else "AN TOÀN")

            card = tk.Frame(
                self.cards_scroll_frame,
                bg="#111a2e",
                highlightthickness=1,
                highlightbackground=card_border,
                padx=8,
                pady=5
            )
            card.pack(fill="x", pady=2)

            # Left: Icon & Object Name
            left_info = tk.Frame(card, bg="#111a2e")
            left_info.pack(side="left", fill="x", expand=True)

            tk.Label(
                left_info,
                text=f"{icon} {obj.name_vi.capitalize()}",
                font=("Arial", 10, "bold"),
                fg="#f8fafc",
                bg="#111a2e"
            ).pack(anchor="w")

            tk.Label(
                left_info,
                text=f"Hướng: {obj.direction_vi}",
                font=("Arial", 8),
                fg="#94a3b8",
                bg="#111a2e"
            ).pack(anchor="w")

            # Right: Distance & Risk Badge
            right_info = tk.Frame(card, bg="#111a2e")
            right_info.pack(side="right")

            tk.Label(
                right_info,
                text=f"{obj.distance:.2f} m",
                font=("Courier", 11, "bold"),
                fg="#38bdf8",
                bg="#111a2e"
            ).pack(anchor="e")

            badge = tk.Label(
                right_info,
                text=risk_text,
                font=("Arial", 7, "bold"),
                fg="#ffffff",
                bg=risk_badge_bg,
                padx=5,
                pady=1
            )
            badge.pack(anchor="e", pady=(1, 0))

    # ------------------------------------------------------------------
    # MODE 2: OCR DOCUMENT READER PANEL
    # ------------------------------------------------------------------
    def _build_ocr_panel(self):
        self.panel_ocr = tk.Frame(self.right_col, bg="#090d16")

        # 1. OCR Action Card
        scan_card = tk.Frame(self.panel_ocr, bg="#0d1424", highlightthickness=1, highlightbackground="#1e293b", padx=10, pady=8)
        scan_card.pack(fill="x", pady=(0, 8))

        self.btn_ocr_scan = PillButton(
            scan_card,
            text="📷  CHỤP & QUÉT VĂN BẢN (Phím T)",
            command=self.scan_and_read_ocr,
            bg="#0284c7",
            fg="#ffffff",
            hover_bg="#0369a1",
            font=("Arial", 11, "bold"),
            padx=14,
            pady=8
        )
        self.btn_ocr_scan.pack(fill="x", pady=(0, 4))

        self.lbl_ocr_status = tk.Label(
            scan_card,
            text="Hướng camera vào tài liệu/sách báo rồi bấm Quét.",
            font=("Arial", 9),
            fg="#94a3b8",
            bg="#0d1424"
        )
        self.lbl_ocr_status.pack(anchor="w")

        # 2. Reading Controls & Font Sizer
        ctrl_card = tk.Frame(self.panel_ocr, bg="#111a2e", padx=8, pady=6)
        ctrl_card.pack(fill="x", pady=(0, 8))

        self.btn_ocr_replay = PillButton(
            ctrl_card, text="▶ Đọc Lại (R)", command=self.replay_ocr_text,
            bg="#059669", fg="#ffffff", hover_bg="#047857", font=("Arial", 9, "bold"), padx=8, pady=3
        )
        self.btn_ocr_replay.pack(side="left", padx=2)

        self.btn_ocr_pause = PillButton(
            ctrl_card, text="⏸ Tạm Dừng (P)", command=self.pause_resume_ocr_text,
            bg="#d97706", fg="#ffffff", hover_bg="#b45309", font=("Arial", 9, "bold"), padx=8, pady=3
        )
        self.btn_ocr_pause.pack(side="left", padx=2)

        self.btn_ocr_stop = PillButton(
            ctrl_card, text="⏹ Dừng", command=self.stop_ocr_text,
            bg="#dc2626", fg="#ffffff", hover_bg="#b91c1c", font=("Arial", 9, "bold"), padx=8, pady=3
        )
        self.btn_ocr_stop.pack(side="left", padx=2)

        # Font adjuster
        font_box = tk.Frame(ctrl_card, bg="#111a2e")
        font_box.pack(side="right")

        tk.Label(font_box, text="Cỡ chữ:", font=("Arial", 8), fg="#94a3b8", bg="#111a2e").pack(side="left", padx=2)

        PillButton(
            font_box, text="A-", command=lambda: self.adjust_font_size(-2),
            bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 8, "bold"), padx=6, pady=2
        ).pack(side="left", padx=1)

        PillButton(
            font_box, text="A+", command=lambda: self.adjust_font_size(2),
            bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 8, "bold"), padx=6, pady=2
        ).pack(side="left", padx=1)

        # 3. Document Text Box with Sentence Highlighting
        text_card = tk.Frame(self.panel_ocr, bg="#0d1424", highlightthickness=1, highlightbackground="#1e293b", padx=8, pady=8)
        text_card.pack(fill="both", expand=True)

        tk.Label(
            text_card,
            text="📖 NỘI DUNG VĂN BẢN",
            font=("Arial", 9, "bold"),
            fg="#38bdf8",
            bg="#0d1424"
        ).pack(anchor="w", pady=(0, 4))

        scroll_y = tk.Scrollbar(text_card)
        scroll_y.pack(side="right", fill="y")

        self.text_display = tk.Text(
            text_card,
            wrap="word",
            bg="#080d1a",
            fg="#f8fafc",
            font=("Arial", self.font_size),
            padx=10,
            pady=10,
            relief="flat",
            yscrollcommand=scroll_y.set
        )
        self.text_display.pack(fill="both", expand=True)
        scroll_y.config(command=self.text_display.yview)

        # High-contrast sentence highlight tag: glowing amber background with black text
        self.text_display.tag_config(
            "active_reading",
            background="#fbbf24",
            foreground="#000000",
            font=("Arial", self.font_size, "bold")
        )

        self.text_display.insert("1.0", "Chưa có văn bản nào được quét.\n\nHãy hướng camera vào trang sách, nhãn hàng hoặc tài liệu rồi bấm phím T.")
        self.text_display.config(state="disabled")

    # ------------------------------------------------------------------
    # FOOTER BAR
    # ------------------------------------------------------------------
    def _build_footer(self):
        footer = tk.Frame(self.root, bg="#0d1424", height=28, padx=14, pady=4)
        footer.pack(fill="x", side="bottom")

        lbl_shortcuts = tk.Label(
            footer,
            text="Phím tắt: [1] Radar | [2] OCR | [Space] Cam/Quét | [T] Đọc chữ | [P] Tạm dừng | [R] Đọc lại | [V] Giọng nói | [M] Đổi Cam | [+/-] Tiêu cự | [Esc] Thoát",
            font=("Arial", 8),
            fg="#94a3b8",
            bg="#0d1424"
        )
        lbl_shortcuts.pack(side="left")

        lbl_status = tk.Label(
            footer,
            text="⚡ Thuần Local 100% (YOLOv8 + EasyOCR + macOS Linh)",
            font=("Arial", 8, "bold"),
            fg="#10b981",
            bg="#0d1424"
        )
        lbl_status.pack(side="right")

    # ------------------------------------------------------------------
    # KEYBOARD SHORTCUTS
    # ------------------------------------------------------------------
    def _bind_shortcuts(self):
        self.root.bind("<Key-1>", lambda e: self.switch_mode("nav"))
        self.root.bind("<Key-2>", lambda e: self.switch_mode("ocr"))
        self.root.bind("<space>", lambda e: self.on_space_key())
        self.root.bind("<Key-t>", lambda e: self.scan_and_read_ocr())
        self.root.bind("<Key-T>", lambda e: self.scan_and_read_ocr())
        self.root.bind("<Key-p>", lambda e: self.pause_resume_ocr_text())
        self.root.bind("<Key-P>", lambda e: self.pause_resume_ocr_text())
        self.root.bind("<Key-r>", lambda e: self.replay_ocr_text())
        self.root.bind("<Key-R>", lambda e: self.replay_ocr_text())
        self.root.bind("<Key-v>", lambda e: self.toggle_voice())
        self.root.bind("<Key-V>", lambda e: self.toggle_voice())
        self.root.bind("<Key-m>", lambda e: self.switch_camera_source())
        self.root.bind("<Key-M>", lambda e: self.switch_camera_source())
        self.root.bind("<Key-o>", lambda e: self.open_image_file())
        self.root.bind("<Key-O>", lambda e: self.open_image_file())
        self.root.bind("<Key-s>", lambda e: self.save_snapshot())
        self.root.bind("<Key-S>", lambda e: self.save_snapshot())
        self.root.bind("<plus>", lambda e: self.adjust_focal_length(25))
        self.root.bind("<equal>", lambda e: self.adjust_focal_length(25))
        self.root.bind("<minus>", lambda e: self.adjust_focal_length(-25))
        self.root.bind("<underscore>", lambda e: self.adjust_focal_length(-25))
        self.root.bind("<Escape>", lambda e: self.on_close())

    # ------------------------------------------------------------------
    # MODE SWITCHING & CONTROLS
    # ------------------------------------------------------------------
    def switch_mode(self, mode: str):
        if mode == self.current_mode:
            return

        self.current_mode = mode
        if mode == "nav":
            self.btn_nav_mode.set_active(True, active_bg="#0284c7")
            self.btn_ocr_mode.set_active(False)
            self.panel_ocr.pack_forget()
            self.panel_nav.pack(fill="both", expand=True)
            self.lbl_viewport_title.config(text="Camera Trực Tiếp & Phân Tích AI 15 Lớp Vật Thể")
            self.alert_manager.set_local_audio(self.voice_enabled)
            if self.voice_enabled:
                audio_service.speak_local("Chế độ tránh vật cản và radar.", interrupt=True)
        else:
            self.btn_ocr_mode.set_active(True, active_bg="#0284c7")
            self.btn_nav_mode.set_active(False)
            self.panel_nav.pack_forget()
            self.panel_ocr.pack(fill="both", expand=True)
            self.lbl_viewport_title.config(text="Chế Độ Đọc Văn Bản & Nhãn Hàng (OCR)")
            self.alert_manager.set_local_audio(False)
            if self.voice_enabled:
                audio_service.speak_local("Chế độ đọc văn bản. Nhấn phím T để chụp và đọc.", interrupt=True)

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        self.alert_manager.mute(not self.voice_enabled)
        if not self.voice_enabled:
            audio_service.stop_speech()
            self.btn_voice.config(text="🔇 Giọng nói: TẮT (V)")
            self.btn_voice.update_style(bg="#475569", fg="#ffffff", hover_bg="#334155")
        else:
            self.btn_voice.config(text="🔊 Giọng nói: BẬT (V)")
            self.btn_voice.update_style(bg="#059669", fg="#ffffff", hover_bg="#047857")
            audio_service.speak_local("Đã bật âm thanh giọng nói.", interrupt=True)

    def on_space_key(self):
        if self.current_mode == "ocr":
            self.scan_and_read_ocr()
        else:
            self.toggle_camera()

    def open_calibration_dialog(self):
        """Open an interactive camera calibration wizard dialog."""
        cal_win = tk.Toplevel(self.root)
        cal_win.title("🎯 Hiệu Chuẩn Khoảng Cách & Camera")
        cal_win.geometry("520x460")
        cal_win.configure(bg="#0d1424")
        cal_win.transient(self.root)
        cal_win.grab_set()

        # Header
        hdr = tk.Frame(cal_win, bg="#111a2e", padx=16, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎯 HIỆU CHUẨN ĐO KHOẢNG CÁCH CHÍNH XÁC", font=("Arial", 11, "bold"), fg="#38bdf8", bg="#111a2e").pack(anchor="w")
        tk.Label(hdr, text="Tự động đồng bộ cự ly mét chuẩn theo từng loại camera và góc nhìn thực tế", font=("Arial", 8), fg="#94a3b8", bg="#111a2e").pack(anchor="w")

        body = tk.Frame(cal_win, bg="#0d1424", padx=16, pady=12)
        body.pack(fill="both", expand=True)

        # 1. Preset Cameras
        tk.Label(body, text="1. Chọn Cấu Hình Camera Có Sẵn:", font=("Arial", 9, "bold"), fg="#e2e8f0", bg="#0d1424").pack(anchor="w", pady=(0, 4))
        preset_box = tk.Frame(body, bg="#0d1424")
        preset_box.pack(fill="x", pady=(0, 12))

        lbl_status = tk.Label(body, text=f"Tiêu cự hiện tại: {self.focal_length:.0f}px", font=("Arial", 8, "italic"), fg="#38bdf8", bg="#0d1424")

        def apply_preset(val: float, name: str):
            diff = val - self.focal_length
            self.adjust_focal_length(diff)
            scale_focal.set(int(val))
            lbl_status.config(text=f"✓ Đã áp dụng cấu hình: {name} ({val:.0f}px)", fg="#10b981")
            if self.voice_enabled:
                audio_service.speak_local(f"Đã chọn cấu hình {name}", interrupt=True)

        PillButton(preset_box, text="💻 MacBook (480px)", command=lambda: apply_preset(480.0, "MacBook"),
                   bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 8, "bold"), padx=8, pady=4).pack(side="left", padx=2)
        PillButton(preset_box, text="📷 USB Webcam (600px)", command=lambda: apply_preset(600.0, "Webcam rời"),
                   bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 8, "bold"), padx=8, pady=4).pack(side="left", padx=2)
        PillButton(preset_box, text="📱 iPhone Cam (420px)", command=lambda: apply_preset(420.0, "iPhone"),
                   bg="#1e293b", fg="#ffffff", hover_bg="#334155", font=("Arial", 8, "bold"), padx=8, pady=4).pack(side="left", padx=2)

        # 2. 1-Click Auto Calibration at 1.0m
        tk.Label(body, text="2. Tự Động Căn Chỉnh Theo Mốc 1.0 Mét (Khuyên dùng):", font=("Arial", 9, "bold"), fg="#e2e8f0", bg="#0d1424").pack(anchor="w", pady=(0, 4))
        tk.Label(body, text="Hãy đứng trước camera cách đúng 1.0 mét (hoặc nhờ 1 người đứng cách 1.0m), sau đó bấm nút:", font=("Arial", 8), fg="#94a3b8", bg="#0d1424", wraplength=480, justify="left").pack(anchor="w", pady=(0, 6))

        def auto_calibrate_1m():
            dets = self.current_detections
            if not dets:
                lbl_status.config(text="⚠️ Chưa thấy vật cản nào! Hãy đứng trước camera trong khung hình.", fg="#f59e0b")
                return

            target = dets[0]
            bbox_h = max(1, target.bbox[3] - target.bbox[1])
            info = INDOOR_CLASSES.get(target.class_key)
            real_h = info.real_height if info else 1.0
            if target.class_key == "person":
                ar = bbox_h / max(1, target.bbox[2] - target.bbox[0])
                real_h = 0.88 if ar < 1.9 else 1.65

            new_f = self.detector.distance_estimator.calibrate_with_known_distance(1.0, bbox_h, real_h, frame_height=self.last_frame.shape[0] if self.last_frame is not None else 480)
            if new_f:
                diff = new_f - self.focal_length
                self.adjust_focal_length(diff)
                scale_focal.set(int(new_f))
                lbl_status.config(text=f"✓ Đã hiệu chuẩn thành công theo {target.name_vi}: Tiêu cự = {new_f:.0f}px (Lưu vĩnh viễn)", fg="#10b981")
                if self.voice_enabled:
                    audio_service.speak_local("Đã hiệu chuẩn cự ly 1 mét thành công.", interrupt=True)

        PillButton(body, text="🎯 LẤY MỐC 1.0 MÉT & TỰ ĐỘNG TÍNH TIÊU CỰ", command=auto_calibrate_1m,
                   bg="#059669", fg="#ffffff", hover_bg="#047857", font=("Arial", 9, "bold"), padx=12, pady=6).pack(anchor="w", pady=(0, 10))

        # 3. Fine tuning slider
        tk.Label(body, text="3. Tinh Chỉnh Thủ Công Bằng Thanh Trượt:", font=("Arial", 9, "bold"), fg="#e2e8f0", bg="#0d1424").pack(anchor="w", pady=(0, 4))
        slider_frame = tk.Frame(body, bg="#0d1424")
        slider_frame.pack(fill="x", pady=(0, 6))

        scale_focal = tk.Scale(slider_frame, from_=250, to=900, orient="horizontal", bg="#111a2e", fg="#38bdf8",
                               highlightthickness=0, troughcolor="#1e293b")
        scale_focal.set(int(self.focal_length))
        scale_focal.pack(side="left", fill="x", expand=True, padx=(0, 8))

        def on_slider(val):
            v = float(val)
            diff = v - self.focal_length
            if abs(diff) >= 1.0:
                self.adjust_focal_length(diff)
                lbl_status.config(text=f"Tiêu cự hiện tại: {self.focal_length:.0f}px", fg="#38bdf8")

        scale_focal.config(command=on_slider)
        lbl_status.pack(anchor="w", pady=4)

        # Close button
        PillButton(cal_win, text="Đóng", command=cal_win.destroy,
                   bg="#334155", fg="#ffffff", hover_bg="#475569", font=("Arial", 9), padx=14, pady=4).pack(side="bottom", pady=8)

    def adjust_focal_length(self, delta: float):
        self.focal_length = max(100.0, self.focal_length + delta)
        self.detector.set_focal_length(self.focal_length)
        self.lbl_focal_val.config(text=f"{self.focal_length:.0f}px")

    def adjust_font_size(self, delta: int):
        self.font_size = max(12, min(36, self.font_size + delta))
        self.text_display.config(font=("Arial", self.font_size))
        self.text_display.tag_config(
            "active_reading",
            font=("Arial", self.font_size, "bold")
        )

    # ------------------------------------------------------------------
    # CAMERA STREAM & VIDEO PROCESSING LOOP
    # ------------------------------------------------------------------
    def _start_camera(self):
        try:
            if self.threaded_cam:
                self.threaded_cam.stop()

            self.threaded_cam = ThreadedCamera(self.camera_source)
            if not self.threaded_cam.is_opened():
                self.threaded_cam = ThreadedCamera(0)

            if self.threaded_cam.is_opened():
                self.camera_running = True
                self.lbl_camera_status.config(text="● Camera Online", fg="#10b981")
                self.btn_cam_toggle.config(text="⏹ Dừng Cam (Space)")
                self._update_frame()
            else:
                self.lbl_camera_status.config(text="○ Camera Lỗi", fg="#ef4444")
                self.btn_cam_toggle.config(text="▶ Bật Cam (Space)")
        except Exception as e:
            print(f"[DesktopApp] Lỗi camera: {e}")

    def toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            if self.threaded_cam:
                self.threaded_cam.stop()
            self.lbl_camera_status.config(text="○ Camera Đã Tắt", fg="#94a3b8")
            self.btn_cam_toggle.config(text="▶ Bật Cam (Space)")
        else:
            self._start_camera()

    def switch_camera_source(self):
        next_source = 1 if self.camera_source == 0 else 0
        self.camera_source = next_source
        self._start_camera()

    def _update_frame(self):
        if not self.camera_running or not self.threaded_cam or not self.threaded_cam.is_opened():
            return

        frame = self.threaded_cam.read()
        if frame is None:
            self.root.after(10, self._update_frame)
            return

        self.last_frame = frame

        # Mode Processing
        if self.current_mode == "nav":
            # Asynchronous background AI inference
            self.inference_worker.submit_frame(frame)
            detected_objects = self.inference_worker.get_latest_detections()
            self.current_detections = detected_objects

            # Update Warning Banner
            self._update_warning_banner(detected_objects)

            # Update Radar Widget
            self.radar_widget.update_objects(detected_objects)

            # Throttled update of objects cards list (every 140ms to prevent widget churn)
            now = time.time()
            if now - self._last_card_update > 0.14:
                self._update_objects_table(detected_objects)
                self._last_card_update = now

            # Draw sleek AR HUD on frame (~0.8ms)
            display_frame = self.detector.draw_hud(frame, detected_objects, fps=self.fps)
        else:
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (int(w * 0.1), int(h * 0.1)), (int(w * 0.9), int(h * 0.9)), (0, 220, 255), 2)
            cv2.putText(
                display_frame,
                "Dat tai lieu vao khung - Nhan T de Quet & Doc",
                (int(w * 0.12), int(h * 0.08)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 255),
                1,
                cv2.LINE_AA
            )

        # FPS calculation
        now = time.time()
        dt = max(1e-4, now - self._prev_frame_time)
        self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)
        self._prev_frame_time = now
        self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")

        # Convert to RGB & scale to fit video container
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        lbl_w = max(400, self.video_label.winfo_width())
        lbl_h = max(280, self.video_label.winfo_height())

        if lbl_w > 50 and lbl_h > 50:
            fh, fw = rgb_frame.shape[:2]
            scale = min(lbl_w / fw, lbl_h / fh)
            new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pil_img = Image.fromarray(rgb_frame)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=tk_img)
        self.video_label.image = tk_img

        # High-frequency polling (12ms) for silky smooth 30-60 FPS
        self.root.after(12, self._update_frame)

    def _schedule_sweep_animation(self):
        if self.current_mode == "nav":
            self.radar_widget.advance_sweep()
        self.root.after(60, self._schedule_sweep_animation)

    # ------------------------------------------------------------------
    # WARNING BANNER UPDATE
    # ------------------------------------------------------------------
    def _update_warning_banner(self, objects: List[DetectedObject]):
        dangers = [o for o in objects if o.risk_level == "DANGER"]
        warnings = [o for o in objects if o.risk_level == "WARNING"]

        if dangers:
            top = dangers[0]
            self.banner_frame.config(bg="#450a0a", highlightbackground="#ef4444")
            self.lbl_banner_icon.config(text="🚨", bg="#450a0a")
            self.lbl_banner_title.config(
                text=f"NGUY HIỂM! {top.name_vi.upper()} ({top.distance:.1f}M)",
                fg="#fecaca",
                bg="#450a0a"
            )
            self.lbl_banner_desc.config(
                text=f"Vật cản ở {top.direction_vi.lower()}. Hãy dừng lại hoặc chuyển hướng an toàn!",
                fg="#ffffff",
                bg="#450a0a"
            )
        elif warnings:
            top = warnings[0]
            self.banner_frame.config(bg="#451a03", highlightbackground="#f59e0b")
            self.lbl_banner_icon.config(text="⚠️", bg="#451a03")
            self.lbl_banner_title.config(
                text=f"CẢNH BÁO: {top.name_vi.upper()} ({top.distance:.1f}M)",
                fg="#fef3c7",
                bg="#451a03"
            )
            self.lbl_banner_desc.config(
                text=f"Có vật cản ở {top.direction_vi.lower()}.",
                fg="#ffffff",
                bg="#451a03"
            )
        else:
            self.banner_frame.config(bg="#0f172a", highlightbackground="#10b981")
            self.lbl_banner_icon.config(text="🛡️", bg="#0f172a")
            self.lbl_banner_title.config(
                text="LỐI ĐI AN TOÀN",
                fg="#10b981",
                bg="#0f172a"
            )
            self.lbl_banner_desc.config(
                text="Không phát hiện vật cản nguy hiểm ở cự ly gần.",
                fg="#cbd5e1",
                bg="#0f172a"
            )

    # ------------------------------------------------------------------
    # OCR SCANNING & SYNCHRONIZED READING
    # ------------------------------------------------------------------
    def scan_and_read_ocr(self):
        if self.last_frame is None:
            return

        self.lbl_ocr_status.config(text="Đang phân tích văn bản trong ảnh (EasyOCR)...", fg="#38bdf8")
        self.root.update_idletasks()

        frame_to_ocr = self.last_frame.copy()

        def _ocr_worker():
            try:
                res = self.ocr_reader.extract_text(frame_to_ocr, render_annotated=True)
                self.root.after(0, lambda: self._on_ocr_finished(res))
            except Exception as e:
                print(f"[DesktopApp] Lỗi OCR: {e}")
                self.root.after(0, lambda: self.lbl_ocr_status.config(text="Lỗi khi phân tích chữ.", fg="#ef4444"))

        threading.Thread(target=_ocr_worker, daemon=True).start()

    def _on_ocr_finished(self, res: OCRResult):
        self.ocr_result = res
        if not res.full_text.strip():
            self.lbl_ocr_status.config(text="Không tìm thấy chữ rõ ràng trong khung hình.", fg="#f59e0b")
            self.text_display.config(state="normal")
            self.text_display.delete("1.0", "end")
            self.text_display.insert("1.0", "Không tìm thấy văn bản rõ ràng trong khung hình.\n\nVui lòng đưa tài liệu lại gần hơn và giữ yên máy ảnh.")
            self.text_display.config(state="disabled")
            if self.voice_enabled:
                audio_service.speak_local("Không tìm thấy văn bản rõ ràng.", interrupt=True)
            return

        self.lbl_ocr_status.config(
            text=f"Đã trích xuất {len(res.paragraphs)} đoạn ({res.word_count} từ) • Độ tin cậy: {int(res.avg_confidence * 100)}%",
            fg="#10b981"
        )

        self.text_display.config(state="normal")
        self.text_display.delete("1.0", "end")
        for p in res.paragraphs:
            self.text_display.insert("end", p + "\n\n")
        self.text_display.config(state="disabled")

        if self.voice_enabled:
            self.replay_ocr_text()

    def replay_ocr_text(self):
        if not self.ocr_result or not self.ocr_result.paragraphs:
            return

        self.is_ocr_reading_paused = False
        self.btn_ocr_pause.config(text="⏸ Tạm Dừng (P)")

        paragraphs = self.ocr_result.paragraphs

        def on_para_start(idx: int, text: str):
            self.root.after(0, lambda: self._highlight_paragraph(idx, text))

        def on_done():
            self.root.after(0, lambda: self._clear_highlights())

        audio_service.speak_paragraphs_sequence(
            paragraphs,
            voice_rate=175,
            on_paragraph_start=on_para_start,
            on_complete=on_done
        )

    def _highlight_paragraph(self, index: int, text: str):
        self.text_display.tag_remove("active_reading", "1.0", "end")
        start_idx = "1.0"
        while True:
            pos = self.text_display.search(text[:30], start_idx, stopindex="end")
            if not pos:
                break
            line_num = pos.split(".")[0]
            end_pos = f"{line_num}.end"
            self.text_display.tag_add("active_reading", pos, end_pos)
            self.text_display.see(pos)
            break

    def _clear_highlights(self):
        self.text_display.tag_remove("active_reading", "1.0", "end")

    def pause_resume_ocr_text(self):
        if self.is_ocr_reading_paused:
            audio_service.resume_document_reading()
            self.is_ocr_reading_paused = False
            self.btn_ocr_pause.config(text="⏸ Tạm Dừng (P)")
        else:
            audio_service.pause_document_reading()
            self.is_ocr_reading_paused = True
            self.btn_ocr_pause.config(text="▶ Tiếp Tục (P)")

    def stop_ocr_text(self):
        audio_service.stop_document_reading()
        self._clear_highlights()
        self.is_ocr_reading_paused = False
        self.btn_ocr_pause.config(text="⏸ Tạm Dừng (P)")

    # ------------------------------------------------------------------
    # FILE OPERATIONS & SCREENSHOTS
    # ------------------------------------------------------------------
    def open_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file ảnh kiểm tra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path or not os.path.exists(file_path):
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Lỗi", "Không thể đọc file ảnh.")
            return

        self.last_frame = img
        if self.current_mode == "ocr":
            self.scan_and_read_ocr()
        else:
            detections = self.detector.detect(img)
            self.radar_widget.update_objects(detections)
            self._update_objects_table(detections)
            self._update_warning_banner(detections)

    def save_snapshot(self):
        if self.last_frame is None:
            return
        outputs_dir = DATA_DIR / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        filename = str(outputs_dir / f"desktop_snapshot_{int(time.time())}.jpg")
        cv2.imwrite(filename, self.last_frame)
        print(f"[DesktopApp] Đã lưu ảnh chụp: {filename}")
        if self.voice_enabled:
            audio_service.speak_local("Đã lưu ảnh chụp màn hình.", interrupt=True)

    def on_close(self):
        print("[DesktopApp] Đang tắt ứng dụng Second Eye...")
        self.camera_running = False
        if hasattr(self, "inference_worker") and self.inference_worker:
            self.inference_worker.stop()
        if hasattr(self, "threaded_cam") and self.threaded_cam:
            self.threaded_cam.stop()
        audio_service.stop_document_reading()
        audio_service.stop_speech()
        self.alert_manager.stop()
        self.root.destroy()

def run_desktop():
    import argparse
    parser = argparse.ArgumentParser(description="Second Eye - Desktop Offline Application")
    parser.add_argument("--source", type=str, default="0", help="Camera index or stream URL")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    root = tk.Tk()
    app = SecondEyeDesktopApp(root, camera_source=src)
    root.mainloop()

if __name__ == "__main__":
    run_desktop()
