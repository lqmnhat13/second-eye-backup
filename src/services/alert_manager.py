"""
Intelligent Audio & Speech Alert System for Second Eye.
Handles non-blocking Vietnamese speech synthesis, urgency-based alert queuing,
smart debouncing/cooldown, and sound generation.
"""

import time
import queue
import threading
import subprocess
import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.config import (
    ALERT_REPEAT_COOLDOWN,
    DANGER_REPEAT_COOLDOWN,
    GLOBAL_ALERT_COOLDOWN,
    INDOOR_CLASSES
)
from src.core.distance_estimator import DetectedObject
from src.services.audio_service import audio_service

@dataclass
class AlertMessage:
    text_vi: str
    risk_level: str
    class_key: str
    distance: float
    direction_vi: str
    priority: int  # 1 (Highest) to 3 (Lowest)
    timestamp: float
    audio_base64: Optional[str] = None

def format_distance_vi(dist: float) -> str:
    """Format floating point distance to natural Vietnamese spoken words."""
    rounded = round(dist, 1)
    integer_part = int(rounded)
    decimal_part = int(round((rounded - integer_part) * 10))

    numbers_vi = {
        0: "không", 1: "một", 2: "hai", 3: "ba", 4: "bốn",
        5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín"
    }

    if decimal_part == 0:
        return f"{integer_part} mét"
    elif integer_part == 0:
        return f"không phẩy {numbers_vi.get(decimal_part, str(decimal_part))} mét"
    else:
        return f"{integer_part} phẩy {numbers_vi.get(decimal_part, str(decimal_part))} mét"

class AlertManager:
    def __init__(self, enable_local_audio: bool = False):
        """
        Initialize Alert Manager.
        """
        self.enable_local_audio = enable_local_audio
        self.alert_queue = queue.PriorityQueue()
        self.history_cooldown: Dict[str, Dict[str, float]] = {}
        self.global_last_alert_time: float = 0.0
        self.running = True
        self.is_muted = False
        self.voice_speed = 175
        self.last_spoken_text = ""
        self.alert_log: List[dict] = []

        if self.enable_local_audio:
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()

    def mute(self, muted: bool = True):
        self.is_muted = muted

    def build_alert_phrase(self, obj: DetectedObject) -> str:
        """
        Generate natural Vietnamese warning sentence.
        """
        dist_str = format_distance_vi(obj.distance)
        name = obj.name_vi
        direction = obj.direction_vi.lower()

        if obj.risk_level == "DANGER":
            if obj.class_key == "stairs":
                return f"Nguy hiểm! Cầu thang ngay phía trước, cách {dist_str}!"
            elif obj.direction_en == "center":
                return f"Cảnh báo! Có {name} ngay phía trước, cách {dist_str}!"
            else:
                return f"Cảnh báo! Có {name} {direction}, cách {dist_str}!"
        elif obj.risk_level == "WARNING":
            return f"Có {name} {direction}, cách {dist_str}."
        else:
            return ""

    def should_trigger_alert(self, obj: DetectedObject) -> bool:
        """
        Strict cooldown check to eliminate any spam or repetitive speech.
        """
        if obj.risk_level == "SAFE":
            return False

        current_time = time.time()
        is_danger = (obj.risk_level == "DANGER")

        # 1. Global Cooldown (Minimum silence between any alerts)
        global_cooldown = DANGER_REPEAT_COOLDOWN if is_danger else GLOBAL_ALERT_COOLDOWN
        if (current_time - self.global_last_alert_time) < global_cooldown:
            return False

        # 2. Object Cooldown
        tracking_key = f"{obj.class_key}_{obj.direction_en}"
        if tracking_key not in self.history_cooldown:
            self.history_cooldown[tracking_key] = {"time": current_time, "dist": obj.distance}
            self.global_last_alert_time = current_time
            return True

        last_record = self.history_cooldown[tracking_key]
        last_time = last_record["time"]
        last_dist = last_record["dist"]

        cooldown = DANGER_REPEAT_COOLDOWN if is_danger else ALERT_REPEAT_COOLDOWN
        time_passed = (current_time - last_time) >= cooldown
        distance_decreased = (last_dist - obj.distance) >= 0.5

        if time_passed or distance_decreased:
            self.history_cooldown[tracking_key] = {"time": current_time, "dist": obj.distance}
            self.global_last_alert_time = current_time
            return True

        return False

    def process_detections(self, detected_objects: List[DetectedObject]) -> List[dict]:
        """
        Process detections, returning AT MOST 1 alert with pre-synthesized Vietnamese MP3.
        """
        if not detected_objects:
            return []

        hazardous = [o for o in detected_objects if o.risk_level in ("DANGER", "WARNING")]
        if not hazardous:
            return []

        hazardous.sort(key=lambda o: (0 if o.risk_level == "DANGER" else 1, o.distance))
        top_obj = hazardous[0]

        if not self.should_trigger_alert(top_obj):
            return []

        phrase = self.build_alert_phrase(top_obj)
        if not phrase:
            return []

        # Get native Vietnamese MP3 base64
        audio_b64 = audio_service.get_audio_base64(phrase)

        priority = 1 if top_obj.risk_level == "DANGER" else 2
        now = time.time()

        alert_msg = AlertMessage(
            text_vi=phrase,
            risk_level=top_obj.risk_level,
            class_key=top_obj.class_key,
            distance=top_obj.distance,
            direction_vi=top_obj.direction_vi,
            priority=priority,
            timestamp=now,
            audio_base64=audio_b64
        )

        # Enqueue for local offline speech synthesis
        if self.enable_local_audio and not self.is_muted:
            self.alert_queue.put((priority, now, alert_msg))

        alert_dict = {
            "text_vi": phrase,
            "risk_level": top_obj.risk_level,
            "class_key": top_obj.class_key,
            "name_vi": top_obj.name_vi,
            "distance": round(top_obj.distance, 2),
            "direction_vi": top_obj.direction_vi,
            "timestamp": round(now, 2),
            "audio_base64": audio_b64
        }

        self.alert_log.insert(0, alert_dict)
        if len(self.alert_log) > 50:
            self.alert_log.pop()

        return [alert_dict]

    def set_local_audio(self, enabled: bool):
        """Enable or disable local speech audio output."""
        self.enable_local_audio = enabled
        if enabled and (not hasattr(self, "worker_thread") or not self.worker_thread.is_alive()):
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()

    def _speech_worker(self):
        """Background thread executing speech alerts smoothly for Desktop / CLI."""
        while self.running:
            try:
                item = self.alert_queue.get(timeout=0.5)
                priority, timestamp, alert = item

                # Discard alerts older than 2.0s
                if time.time() - timestamp > 2.0:
                    self.alert_queue.task_done()
                    continue

                if not self.is_muted and self.enable_local_audio:
                    # Speak with Linh/macOS local voice
                    self._speak_local(alert.text_vi, is_danger=(priority == 1))

                self.last_spoken_text = alert.text_vi
                self.alert_queue.task_done()

            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.1)

    def _speak_local(self, text: str, is_danger: bool = False):
        """Invoke local OS speech engine with Vietnamese voice specifically."""
        try:
            audio_service.speak_local(text, voice_rate=self.voice_speed, interrupt=is_danger)
        except Exception:
            pass

    def stop(self):
        """Stop alert worker thread."""
        self.running = False
        audio_service.stop_speech()
