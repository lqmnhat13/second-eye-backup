"""
Second Eye - Trợ Lý Thị Giác & Đọc Văn Bản Cho Người Khiếm Thị
100% Offline Local Desktop Application.

Khởi chạy nhanh:
    python main.py
    python main.py --source 0
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.desktop.app import run_desktop

if __name__ == "__main__":
    run_desktop()
