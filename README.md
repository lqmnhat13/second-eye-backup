# 👁️ Second Eye - Hệ Thống Trợ Lý Thị Giác & Đọc Văn Bản Cho Người Khiếm Thị

**Second Eye** là hệ thống trợ lý thông minh đa năng dành cho người khiếm thị và người thị lực kém (Low Vision). Hệ thống tích hợp mô hình Deep Learning YOLOv8 nhận diện 15 lớp vật thể/vật cản, thuật toán ước lượng khoảng cách Pinhole Camera, radar 2D thời gian thực và công nghệ **OCR Đọc Văn Bản Tiếng Việt** giúp người khiếm thị đọc sách báo, nhãn thuốc, hóa đơn, bao bì thực phẩm hay biển báo bằng giọng nói chuẩn xác.

---

## 🌟 2 Chế Độ Hoạt Động (Dual-Mode System)

### 🧭 Chế độ 1: Định Vị & Tránh Vật Cản (Navigation & Radar Mode)
- Nhận diện 15 lớp vật thể trong nhà (Cầu thang, người, cửa, ghế, bàn, giường, quạt, thùng rác,...).
- Ước lượng khoảng cách mét thời gian thực và phân vùng hướng đi (Trái, Phía trước, Phải).
- Radar không gian 2D góc nhìn từ trên xuống (Top-down 4.0m).
- Cảnh báo âm thanh giọng nói Tiếng Việt ngắn gọn, thông minh và chống spam lặp từ.

### 📖 Chế độ 2: Đọc Văn Bản & Nhãn Hàng Thông Minh (OCR Document Reader)
- Chụp ảnh và trích xuất chữ viết tay/in ấn (Tiếng Việt có dấu & Tiếng Anh).
- Tự động sắp xếp đoạn văn theo thứ tự đọc tự nhiên.
- Đọc to văn bản bằng giọng nói Tiếng Việt truyền cảm, mượt mà.
- **Tô sáng từng câu/đoạn đang đọc (Sentence Highlighting)**, hỗ trợ nút Tạm dừng (`P`), Đọc lại (`R`) và tùy chỉnh cỡ chữ cực lớn cho người thị lực kém.

---

## 🏛️ Cấu trúc Dự án (Project Architecture)

```text
Second-Eye/
├── models/                           # Trọng số mô hình Deep Learning (YOLO weights)
│   └── yolov8n.pt
├── src/                              # Toàn bộ mã nguồn cốt lõi (Source Package)
│   ├── __init__.py
│   ├── config.py                     # Cấu hình 15 lớp vật thể, ngưỡng an toàn & thông số camera
│   ├── core/                         # Module AI, Hình học & OCR
│   │   ├── __init__.py
│   │   ├── detector.py               # Nhận diện vật thể YOLOv8 & HUD Overlay
│   │   ├── distance_estimator.py     # Ước lượng khoảng cách mét & Tọa độ Radar 2D
│   │   └── ocr_reader.py             # OCR trích xuất văn bản Tiếng Việt (EasyOCR)
│   ├── services/                     # Dịch vụ Âm thanh & Quản lý Cảnh báo
│   │   ├── __init__.py
│   │   ├── alert_manager.py          # Hàng đợi cảnh báo ưu tiên, chống spam lặp từ
│   │   └── audio_service.py          # Tổng hợp giọng đọc Tiếng Việt chuẩn (gTTS & Cache MP3)
│   └── web/                          # Ứng dụng Web Dashboard & REST API
│       ├── __init__.py
│       ├── app.py                    # Backend FastAPI & API endpoints (Nav + OCR)
│       ├── static/                   # Static assets (CSS, JS, Audio Cache)
│       │   ├── css/style.css
│       │   ├── js/app.js
│       │   └── audio_cache/          # Cache file âm thanh MP3
│       └── templates/                # Giao diện Web HTML
│           └── index.html
├── scripts/                          # Entry points thực thi hệ thống
│   ├── run_server.py                 # Khởi chạy Web Server (Hỗ trợ HTTPS cho camera di động)
│   ├── run_cli.py                    # Khởi chạy Desktop OpenCV HUD (Phím T để đọc chữ)
│   └── run_demo.py                   # Chạy suy luận AI trên ảnh mẫu
├── tests/                            # Bộ kiểm thử tự động toàn diện (Unit & Integration)
│   ├── __init__.py
│   └── test_system.py                # 6 bài kiểm thử toàn diện hệ thống & OCR
├── data/                             # Dữ liệu mẫu & kết quả đầu ra
│   ├── samples/                      # Ảnh mẫu trong nhà
│   │   └── indoor_demo.jpg
│   └── outputs/                      # Ảnh kết quả & snapshots
├── run.sh                            # Script khởi động tự động đa chế độ
├── requirements.txt                  # Danh sách dependencies
├── pyrightconfig.json                # Cấu hình Language Server / Pyright
└── .vscode/settings.json             # Cấu hình VS Code
```

---

## 🚀 Hướng dẫn Khởi chạy (Quick Start)

### 1. Cài đặt Môi trường & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy bằng Script `run.sh`

```bash
# [1] Khởi chạy Web Dashboard (Tránh vật cản + Đọc văn bản OCR)
./run.sh 1

# [2] Khởi chạy Desktop HUD trực tiếp trên máy tính (OpenCV Window)
./run.sh 2

# [2.1] Khởi chạy Desktop HUD với iPhone qua Continuity Camera (source 1)
./run.sh 2 1

# [2.2] Khởi chạy Desktop HUD với IP Camera / DroidCam
./run.sh 2 "http://192.168.1.15:8080/video"

# [3] Chạy toàn bộ 6 bài kiểm thử tự động (Unit, Integration & OCR)
./run.sh 3

# [4] Chạy demo trên ảnh mẫu trong nhà
./run.sh 4
```

---

## ⌨️ Phím Tắt Trợ Năng (Keyboard Accessibility)

| Phím | Chức năng |
|:---:|:---|
| <kbd>1</kbd> | Chuyển sang chế độ **Tránh Vật Cản & Radar** |
| <kbd>2</kbd> | Chuyển sang chế độ **Đọc Văn Bản & Nhãn Hàng (OCR)** |
| <kbd>T</kbd> | **Chụp & Đọc to văn bản** trước camera ngay lập tức |
| <kbd>Space</kbd> | Bật/Tắt Camera (hoặc Chụp & Đọc văn bản khi ở chế độ OCR) |
| <kbd>P</kbd> | **Tạm dừng / Tiếp tục** đọc đoạn văn |
| <kbd>R</kbd> | **Đọc lại** đoạn văn từ đầu |
| <kbd>V</kbd> | Bật / Tắt âm thanh giọng nói Tiếng Việt |
| <kbd>M</kbd> | Đổi camera trước / sau |
| <kbd>+</kbd> / <kbd>-</kbd> | Tăng / Giảm tiêu cự hiệu chuẩn khoảng cách |
| <kbd>Esc</kbd> | Dừng camera hoặc đóng hộp thoại |

---

## 📱 Hướng dẫn Dùng Camera Điện thoại (iPhone / Android)

Web Dashboard đã tích hợp sẵn **HTTPS** để truy cập camera di động qua Wi-Fi:

1. Chạy máy chủ: `./run.sh 1`
2. Mở trình duyệt trên điện thoại (Safari hoặc Chrome) và truy cập đường dẫn:
   ```text
   https://192.168.1.47:8000
   ```
3. Khi trình duyệt cảnh báo chứng chỉ SSL nội bộ:
   - Chọn **Nâng cao (Advanced)** -> Bấm **Tiếp tục truy cập (Proceed)**.
4. Bấm nút **"Bật Camera"** và cấp quyền để sử dụng trực tiếp trên điện thoại!
