#!/bin/bash
# ==============================================================================
# SECOND EYE - HỆ THỐNG TRỢ LÝ THỊ GIÁC CHO NGƯỜI KHIẾM THỊ
# Khởi động ứng dụng Web Dashboard hoặc ứng dụng Desktop OpenCV HUD
# ==============================================================================

PYTHON_BIN="/opt/anaconda3/envs/ai-macbook/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=================================================================="
echo "    SECOND EYE - HỆ THỐNG CẢNH BÁO VẬT CẢN TRONG NHÀ CHO NGƯỜI KHIẾM THỊ"
echo "=================================================================="
echo "1. Khởi động Web Dashboard (Giao diện Web, Radar 2D & Giọng nói Tiếng Việt)"
echo "2. Khởi động Desktop HUD (Cửa sổ OpenCV trực tiếp trên máy tính)"
echo "3. Chạy kiểm thử tự động toàn diện (Unit & Integration Tests)"
echo "4. Chạy demo trên ảnh mẫu trong nhà"
echo "=================================================================="

MODE="${1:-1}"
PORT="${2:-8000}"

if [ "$MODE" == "1" ] || [ "$MODE" == "web" ]; then
    echo "Đang khởi động Web Dashboard (tự động chọn port trống nếu port $PORT bận)..."
    $PYTHON_BIN web_app.py --port "$PORT" --auto-port
elif [ "$MODE" == "2" ] || [ "$MODE" == "desktop" ]; then
    echo "Đang khởi động Desktop HUD với Camera máy tính..."
    $PYTHON_BIN main_cli.py
elif [ "$MODE" == "3" ] || [ "$MODE" == "test" ]; then
    $PYTHON_BIN test_sample.py
elif [ "$MODE" == "4" ] || [ "$MODE" == "demo" ]; then
    $PYTHON_BIN demo_inference.py
else
    echo "Lựa chọn không hợp lệ. Sử dụng: ./run.sh [1|2|3|4] [port]"
fi
