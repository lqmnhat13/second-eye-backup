"""
Optical Character Recognition (OCR) & Document Reader Module for Second Eye.
Extracts Vietnamese and English text from camera frames, books, medication labels,
receipts, and documents for natural text-to-speech reading to visually impaired users.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    rect: Tuple[int, int, int, int]  # (x, y, w, h)

@dataclass
class OCRResult:
    full_text: str
    paragraphs: List[str]
    lines: List[OCRLine]
    avg_confidence: float
    word_count: int
    annotated_image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_text": self.full_text,
            "paragraphs": self.paragraphs,
            "word_count": self.word_count,
            "avg_confidence": round(self.avg_confidence, 2),
            "lines": [
                {
                    "text": line.text,
                    "confidence": round(line.confidence, 2),
                    "rect": list(line.rect),
                    "bbox": line.bbox
                }
                for line in self.lines
            ]
        }

class OCRReader:
    _instance: Optional['OCRReader'] = None
    _reader: Any = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OCRReader, cls).__new__(cls)
        return cls._instance

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        if self._reader is None:
            if languages is None:
                languages = ['vi', 'en']
            self.languages = languages
            self.gpu = gpu
            self._load_reader()

    def _load_reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            import easyocr
            print(f"[OCRReader] Khởi tạo EasyOCR Engine với ngôn ngữ: {self.languages} (GPU={self.gpu})...")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            print("[OCRReader] EasyOCR Engine đã sẵn sàng!")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for OCR (contrast enhancement, adaptive equalization, slight sharpening).
        """
        if image is None or image.size == 0:
            return image

        # Resize if image is excessively large for faster inference
        h, w = image.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Convert to LAB color space for luminance CLAHE enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_bgr
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def extract_text(self, image: np.ndarray, min_conf: float = 0.20, render_annotated: bool = True) -> OCRResult:
        """
        Perform OCR on input BGR image.
        Returns sorted paragraphs, lines, full text, and annotated image.
        """
        if image is None or image.size == 0:
            return OCRResult(
                full_text="",
                paragraphs=[],
                lines=[],
                avg_confidence=0.0,
                word_count=0,
                annotated_image=image
            )

        processed = self.preprocess_image(image)
        
        # EasyOCR readtext returns: [ (bbox, text, prob), ... ]
        raw_results = self._reader.readtext(processed)

        parsed_lines: List[OCRLine] = []
        total_conf = 0.0

        for item in raw_results:
            bbox, text, prob = item
            if prob < min_conf or not text.strip():
                continue

            # Convert bbox to int coordinates
            int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
            xs = [pt[0] for pt in int_bbox]
            ys = [pt[1] for pt in int_bbox]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            rect = (min_x, min_y, max_x - min_x, max_y - min_y)

            parsed_lines.append(OCRLine(
                text=text.strip(),
                confidence=float(prob),
                bbox=int_bbox,
                rect=rect
            ))
            total_conf += float(prob)

        # Sort lines naturally: Top-to-Bottom, Left-to-Right
        # Group lines with close Y coordinates into the same horizontal reading band
        def get_sort_key(line: OCRLine):
            # Quantize Y coordinate by line height / 2 to group lines on the same row
            y_band = line.rect[1] // max(15, line.rect[3] // 2)
            return (y_band, line.rect[0])

        parsed_lines.sort(key=get_sort_key)

        # Build full text and paragraphs
        text_lines = [l.text for l in parsed_lines]
        full_text = " ".join(text_lines)
        word_count = len(full_text.split()) if full_text else 0
        avg_conf = (total_conf / len(parsed_lines)) if parsed_lines else 0.0

        # Paragraph grouping: group lines that end with punctuation or have vertical gaps
        paragraphs: List[str] = []
        current_para: List[str] = []

        for line in parsed_lines:
            current_para.append(line.text)
            # If line ends with sentence terminator (. ! ? :), break paragraph
            if line.text.endswith(('.', '!', '?', ':')) or len(current_para) >= 4:
                paragraphs.append(" ".join(current_para))
                current_para = []

        if current_para:
            paragraphs.append(" ".join(current_para))

        # Render visual annotation
        annotated_image = None
        if render_annotated:
            annotated_image = self.draw_ocr_hud(image.copy(), parsed_lines)

        return OCRResult(
            full_text=full_text,
            paragraphs=paragraphs,
            lines=parsed_lines,
            avg_confidence=avg_conf,
            word_count=word_count,
            annotated_image=annotated_image
        )

    def draw_ocr_hud(self, frame: np.ndarray, lines: List[OCRLine]) -> np.ndarray:
        """
        Draw high-visibility text bounding boxes with line numbering and translucent badges.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent background overlay for text boxes
        overlay = annotated.copy()

        for idx, line in enumerate(lines, 1):
            x, y, bw, bh = line.rect
            pts = np.array(line.bbox, np.int32).reshape((-1, 1, 2))
            
            # Fill bounding box with translucent blue/cyan
            cv2.fillPoly(overlay, [pts], (255, 180, 0))
            # Border rectangle
            cv2.polylines(annotated, [pts], True, (0, 220, 255), 2, cv2.LINE_AA)

            # Badge number
            badge_x = max(5, x)
            badge_y = max(15, y - 5)
            cv2.circle(annotated, (badge_x, badge_y), 10, (0, 180, 255), -1)
            cv2.putText(annotated, str(idx), (badge_x - 4, badge_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        # Status header
        cv2.rectangle(annotated, (0, 0), (w, 38), (20, 20, 24), -1)
        status_str = f"SECOND EYE OCR | Lines: {len(lines)} | Words: {sum(len(l.text.split()) for l in lines)}"
        cv2.putText(annotated, status_str, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        return annotated

# Global singleton instance
ocr_reader = OCRReader()
