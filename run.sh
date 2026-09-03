#!/bin/bash
# ==============================================================================
# SECOND EYE - HỆ THỐNG TRỢ LÝ THỊ GIÁC CHO NGƯỜI KHIẾM THỊ
# Khởi động ứng dụng Desktop App Thuần Local 100% (Offline)
# ==============================================================================

# Determine Python Binary
PYTHON_BIN="/opt/anaconda3/envs/ai-macbook/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

export PYTHONPATH="."

echo "=================================================================="
echo "    SECOND EYE - HỆ THỐNG HỖ TRỢ THỊ GIÁC & OCR (THUẦN LOCAL)"
echo "=================================================================="
echo "1. Khởi động Desktop GUI Hoàn Chỉnh (Mặc định: Camera + Radar 2D + OCR + Giọng Linh)"
echo "2. Khởi động Desktop HUD Tối Giản (Cửa sổ OpenCV trực tiếp trên máy tính)"
echo "3. Chạy kiểm thử tự động toàn diện (Unit & Desktop Tests)"
echo "4. Chạy demo suy luận AI trên ảnh mẫu"
echo "=================================================================="

MODE="${1:-1}"
PARAM="${2:-0}"

if [ "$MODE" == "1" ] || [ "$MODE" == "desktop" ] || [ "$MODE" == "app" ]; then
    SOURCE="${2:-0}"
    echo "Đang khởi động Desktop GUI Second Eye (Thuần Local 100%, Camera nguồn: $SOURCE)..."
    $PYTHON_BIN main.py --source "$SOURCE"
elif [ "$MODE" == "2" ] || [ "$MODE" == "hud" ] || [ "$MODE" == "cli" ]; then
    SOURCE="${2:-0}"
    echo "Đang khởi động Desktop HUD OpenCV tối giản với Camera nguồn: $SOURCE..."
    $PYTHON_BIN scripts/run_cli.py --source "$SOURCE"
elif [ "$MODE" == "3" ] || [ "$MODE" == "test" ]; then
    echo "Đang chạy kiểm thử hệ thống tự động..."
    $PYTHON_BIN tests/test_system.py
elif [ "$MODE" == "4" ] || [ "$MODE" == "demo" ]; then
    echo "Đang chạy demo suy luận AI trên ảnh mẫu..."
    $PYTHON_BIN scripts/run_demo.py
else
    echo "Lựa chọn không hợp lệ. Sử dụng: ./run.sh [1|2|3|4] [source]"
fi

