"""
Vietnamese Audio Service for Second Eye.
Supports 100% Offline Local Speech (macOS native 'say -v Linh' / pyttsx3)
and cached online TTS clips (gTTS) for web compatibility.
"""

import os
import io
import sys
import time
import hashlib
import base64
import threading
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Callable

from src.config import DATA_DIR

AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

class VietnameseAudioService:
    def __init__(self):
        self.memory_cache: Dict[str, str] = {}  # {hash: base64_audio}
        self._current_proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._doc_thread: Optional[threading.Thread] = None
        self._doc_stop_event = threading.Event()
        self._doc_pause_event = threading.Event()
        self._doc_pause_event.set() # Not paused initially
        self.is_speaking_document = False

    # ------------------------------------------------------------------
    # 1. 100% OFFLINE LOCAL SPEECH ENGINE (macOS 'say -v Linh' / pyttsx3)
    # ------------------------------------------------------------------
    def stop_speech(self):
        """Immediately terminate any currently active speech process."""
        with self._proc_lock:
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                    self._current_proc.wait(timeout=0.2)
                except Exception:
                    pass
                self._current_proc = None

    def speak_local(self, text: str, voice_rate: int = 175, interrupt: bool = True) -> bool:
        """
        Speak text using the local offline speech engine.
        On macOS: uses native high-quality Vietnamese voice 'Linh'.
        On Linux/Windows: falls back to pyttsx3.
        Non-blocking execution.
        """
        if not text or not text.strip():
            return False

        clean_text = text.strip()

        if interrupt:
            self.stop_speech()

        def _worker():
            try:
                if sys.platform == "darwin":
                    cmd = ["say", "-v", "Linh", "-r", str(voice_rate), clean_text]
                    with self._proc_lock:
                        self._current_proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    self._current_proc.wait()
                    with self._proc_lock:
                        self._current_proc = None
                else:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", voice_rate)
                    engine.say(clean_text)
                    engine.runAndWait()
            except Exception as e:
                print(f"[AudioService] Local TTS error: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True

    def speak_paragraphs_sequence(
        self,
        paragraphs: List[str],
        voice_rate: int = 175,
        on_paragraph_start: Optional[Callable[[int, str], None]] = None,
        on_complete: Optional[Callable[[], None]] = None
    ):
        """
        Read a sequence of paragraphs one by one with live progress callbacks.
        Allows pausing, resuming, and stopping.
        """
        self.stop_document_reading()

        self._doc_stop_event.clear()
        self._doc_pause_event.set()
        self.is_speaking_document = True

        def _doc_worker():
            try:
                for idx, para in enumerate(paragraphs):
                    if self._doc_stop_event.is_set():
                        break

                    # Wait if paused
                    self._doc_pause_event.wait()
                    if self._doc_stop_event.is_set():
                        break

                    clean_para = para.strip()
                    if not clean_para:
                        continue

                    if on_paragraph_start:
                        try:
                            on_paragraph_start(idx, clean_para)
                        except Exception:
                            pass

                    # Speak this paragraph synchronously on macOS
                    if sys.platform == "darwin":
                        cmd = ["say", "-v", "Linh", "-r", str(voice_rate), clean_para]
                        with self._proc_lock:
                            self._current_proc = subprocess.Popen(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                        
                        # Poll while process runs to allow responsive stop/pause
                        while self._current_proc and self._current_proc.poll() is None:
                            if self._doc_stop_event.is_set():
                                self.stop_speech()
                                break
                            time.sleep(0.05)

                        with self._proc_lock:
                            self._current_proc = None
                    else:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty("rate", voice_rate)
                        engine.say(clean_para)
                        engine.runAndWait()

                    # Brief pause between paragraphs
                    time.sleep(0.2)

            finally:
                self.is_speaking_document = False
                if on_complete:
                    try:
                        on_complete()
                    except Exception:
                        pass

        self._doc_thread = threading.Thread(target=_doc_worker, daemon=True)
        self._doc_thread.start()

    def pause_document_reading(self):
        """Pause the current document reading sequence."""
        self._doc_pause_event.clear()
        self.stop_speech()

    def resume_document_reading(self):
        """Resume the paused document reading sequence."""
        self._doc_pause_event.set()

    def stop_document_reading(self):
        """Stop and reset the document reading sequence."""
        self._doc_stop_event.set()
        self._doc_pause_event.set()
        self.stop_speech()
        self.is_speaking_document = False

    # ------------------------------------------------------------------
    # 2. CACHED MP3 / DATA URL GENERATION (100% Offline)
    # ------------------------------------------------------------------
    def get_audio_base64(self, text_vi: str) -> Optional[str]:
        """
        Retrieve cached native Vietnamese audio (MP3 Base64 Data URL) if available.
        100% offline - never makes blocking network requests.
        """
        if not text_vi or not text_vi.strip():
            return None

        clean_text = text_vi.strip()
        text_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()

        # Check in-memory cache
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]

        # Check on-disk cache
        disk_path = os.path.join(AUDIO_CACHE_DIR, f"{text_hash}.mp3")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    data_uri = f"data:audio/mp3;base64,{b64}"
                    self.memory_cache[text_hash] = data_uri
                    return data_uri
            except Exception as e:
                print(f"[AudioService] Error reading disk cache: {e}")

        return None

    def synthesize_document_paragraphs(self, paragraphs: list) -> list:
        """
        Synthesize audio for each paragraph in a document for synchronized reading.
        """
        results = []
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            audio_b64 = self.get_audio_base64(p_clean)
            results.append({
                "text": p_clean,
                "audio_base64": audio_b64
            })
        return results

audio_service = VietnameseAudioService()


