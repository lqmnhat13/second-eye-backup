# 👁️ Second Eye - Hệ Thống Trợ Lý Thị Giác & Đọc Văn Bản Cho Người Khiếm Thị

**Second Eye** là hệ thống trợ lý thông minh đa năng dành cho người khiếm thị và người thị lực kém (Low Vision). Hệ thống tích hợp mô hình Deep Learning YOLOv8 nhận diện 15 lớp vật thể/vật cản, thuật toán ước lượng khoảng cách Pinhole Camera, radar 2D thời gian thực và công nghệ **OCR Đọc Văn Bản Tiếng Việt** giúp người khiếm thị đọc sách báo, nhãn thuốc, hóa đơn, bao bì thực phẩm hay biển báo bằng giọng nói chuẩn xác.

> **Tình trạng AI:** Cấu hình có 15 nhóm nhưng trọng số mặc định hiện chỉ hỗ trợ 11 nhóm; thiếu cầu thang, cửa, quạt, thùng rác. Khoảng cách mặc định là ước lượng hình học, chưa được xác thực độ chính xác. Xem [rà soát và hướng dẫn thử metric depth](docs/AI_REVIEW.md).

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
│   │   ├── detector.py               # Nhận diện vật thể YOLOv8 & AR HUD Overlay
│   │   ├── distance_estimator.py     # Ước lượng khoảng cách mét & Tọa độ Radar 2D
│   │   └── ocr_reader.py             # OCR trích xuất văn bản Tiếng Việt (EasyOCR)
│   ├── services/                     # Dịch vụ Âm thanh & Quản lý Cảnh báo
│   │   ├── __init__.py
│   │   ├── alert_manager.py          # Hàng đợi cảnh báo ưu tiên, chống spam lặp từ
│   │   └── audio_service.py          # Dịch vụ giọng đọc Tiếng Việt offline (macOS Linh / pyttsx3)
│   └── desktop/                      # Ứng dụng Desktop GUI Hoàn Chỉnh (Tkinter + Pillow)
│       ├── __init__.py
│       ├── app.py                    # Giao diện chính Desktop App, đa luồng & FPS cao
│       └── radar_canvas.py           # Widget Radar 2D không gian trực quan
├── scripts/                          # Entry points thực thi hệ thống
│   ├── run_cli.py                    # Khởi chạy Desktop OpenCV HUD tối giản
│   └── run_demo.py                   # Chạy suy luận AI trên ảnh mẫu
├── tests/                            # Bộ kiểm thử tự động toàn diện (Unit & Desktop)
│   ├── __init__.py
│   ├── test_system.py                # Bài kiểm thử toàn diện hệ thống AI & OCR
│   └── test_desktop.py               # Kiểm thử giao diện Desktop GUI & âm thanh offline
├── data/                             # Dữ liệu mẫu & kết quả đầu ra
│   ├── samples/                      # Ảnh mẫu trong nhà
│   │   └── indoor_demo.jpg
│   └── outputs/                      # Ảnh kết quả & snapshots
├── main.py                           # Điểm khởi chạy chính của ứng dụng Desktop GUI
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

### 2. Khởi chạy Ứng Dụng Desktop Thuần Local (100% Offline)

Hệ thống hoạt động trực tiếp trên máy tính dưới dạng ứng dụng Desktop GUI, không cần mở trình duyệt hay kết nối mạng internet:

```bash
# [Mặc định] Khởi chạy trực tiếp Desktop GUI hoàn chỉnh:
./run.sh
# Hoặc:
python main.py

# Khởi chạy Desktop GUI với Camera iPhone (Continuity Camera):
python main.py --source 1
# Hoặc:
./run.sh 1 1

# [Tùy chọn 2] Khởi chạy cửa sổ OpenCV HUD tối giản:
./run.sh 2

# [Tùy chọn 3] Chạy toàn bộ bài kiểm thử tự động hệ thống:
./run.sh 3

# [Tùy chọn 4] Chạy demo suy luận trên ảnh mẫu:
./run.sh 4

# [Tùy chọn 5] Khởi chạy Web Dashboard cũ (nếu cần truy cập từ xa):
./run.sh 5
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

## 📱 Hướng dẫn Dùng Camera Ngoài / iPhone (Continuity Camera)

Hệ thống hỗ trợ chuyển đổi linh hoạt nguồn camera trên Desktop:

1. **Dùng Camera iPhone (Continuity Camera trên macOS)**:
   - Đặt iPhone gần MacBook (cùng đăng nhập Apple ID).
   - Chạy lệnh:
     ```bash
     python main.py --source 1
     ```
   - Hoặc bấm phím <kbd>M</kbd> (hoặc nút **🔄 Đổi Cam**) trực tiếp trên giao diện Desktop.

2. **Dùng IP Webcam / DroidCam**:
   - Nhập URL luồng video trực tiếp:
     ```bash
     python main.py --source "http://192.168.1.15:8080/video"
     ```
