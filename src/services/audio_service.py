"""
Vietnamese Audio Service for Second Eye.
Pre-synthesizes and caches high-quality native Vietnamese voice clips using gTTS (Google Vietnamese TTS)
to ensure 100% authentic pronunciation across all web browsers without relying on client-side OS voices.
"""

import os
import io
import hashlib
import base64
from pathlib import Path
from typing import Dict, Optional
from gtts import gTTS

from src.config import SRC_DIR

AUDIO_CACHE_DIR = SRC_DIR / "web" / "static" / "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

class VietnameseAudioService:
    def __init__(self):
        self.memory_cache: Dict[str, str] = {}  # {hash: base64_audio}

    def get_audio_base64(self, text_vi: str) -> Optional[str]:
        """
        Generate or retrieve cached native Vietnamese audio (MP3 Base64 Data URL).
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

        # Synthesize with gTTS in pure Vietnamese ('vi')
        try:
            tts = gTTS(text=clean_text, lang="vi", slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_bytes = mp3_fp.getvalue()

            # Save to disk cache
            with open(disk_path, "wb") as f:
                f.write(mp3_bytes)

            b64 = base64.b64encode(mp3_bytes).decode("utf-8")
            data_uri = f"data:audio/mp3;base64,{b64}"
            self.memory_cache[text_hash] = data_uri
            return data_uri
        except Exception as e:
            print(f"[AudioService] gTTS generation error: {e}")
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

