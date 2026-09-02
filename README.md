# 👁️ Second Eye - Hệ Thống Trợ Lý Thị Giác Cho Người Khiếm Thị

**Second Eye** là hệ thống thị giác máy tính và hỗ trợ định hướng thời gian thực dành cho người khiếm thị trong môi trường trong nhà (Indoor Navigation). Hệ thống tích hợp mô hình Deep Learning YOLOv8 nhận diện 15 lớp vật thể/vật cản, thuật toán ước lượng khoảng cách dựa trên hình học Pinhole Camera, radar định vị 2D và cảnh báo âm thanh giọng nói Tiếng Việt thông minh.

---

## 🏛️ Cấu trúc Dự án (Project Architecture)

Dự án được xây dựng theo kiến trúc module hóa tiêu chuẩn (**Modular Standard Architecture**):

```text
Second-Eye/
├── models/                           # Trọng số mô hình Deep Learning (YOLO weights)
│   └── yolov8n.pt
├── src/                              # Toàn bộ mã nguồn cốt lõi (Source Package)
│   ├── __init__.py
│   ├── config.py                     # Cấu hình 15 lớp vật thể, ngưỡng an toàn & thông số camera
│   ├── core/                         # Module Thị giác máy tính & Tính toán hình học
│   │   ├── __init__.py
│   │   ├── detector.py               # Nhận diện vật thể YOLOv8 & Vẽ giao diện AR HUD
│   │   └── distance_estimator.py     # Ước lượng khoảng cách mét & Tọa độ Radar 2D
│   ├── services/                     # Dịch vụ Âm thanh & Quản lý Cảnh báo
│   │   ├── __init__.py
│   │   ├── alert_manager.py          # Hàng đợi cảnh báo ưu tiên, chống spam lặp từ
│   │   └── audio_service.py          # Tổng hợp giọng nói Tiếng Việt chuẩn qua gTTS & Cache MP3
│   └── web/                          # Ứng dụng Web Dashboard & REST/WebSocket API
│       ├── __init__.py
│       ├── app.py                    # Backend FastAPI & API endpoints
│       ├── static/                   # Static assets (CSS, JS, Audio Cache)
│       │   ├── css/style.css
│       │   ├── js/app.js
│       │   └── audio_cache/          # Cache file âm thanh MP3
│       └── templates/                # Giao diện Web HTML
│           └── index.html
├── scripts/                          # Entry points thực thi hệ thống
│   ├── run_server.py                 # Khởi chạy Web Server (Hỗ trợ HTTPS cho camera di động)
│   ├── run_cli.py                    # Khởi chạy Desktop OpenCV HUD
│   └── run_demo.py                   # Chạy suy luận AI trên ảnh mẫu
├── tests/                            # Bộ kiểm thử tự động toàn diện (Unit & Integration)
│   ├── __init__.py
│   └── test_system.py                # Kiểm thử 5 thành phần cốt lõi của hệ thống
├── data/                             # Dữ liệu mẫu & kết quả đầu ra
│   ├── samples/                      # Ảnh mẫu trong nhà
│   │   └── indoor_demo.jpg
│   └── outputs/                      # Ảnh kết quả & snapshots
├── run.sh                            # Script khởi động đa chế độ (Web, Desktop, Test, Demo)
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
# [1] Khởi chạy Web Dashboard (Giao diện Web, Radar 2D & Giọng nói Tiếng Việt)
./run.sh 1

# [2] Khởi chạy Desktop HUD trực tiếp trên máy tính (OpenCV Window)
./run.sh 2

# [2.1] Khởi chạy Desktop HUD với iPhone / Continuity Camera (source index 1)
./run.sh 2 1

# [2.2] Khởi chạy Desktop HUD với Camera IP / DroidCam
./run.sh 2 "http://192.168.1.15:8080/video"

# [3] Chạy toàn bộ bài kiểm thử tự động (Unit & Integration Tests)
./run.sh 3

# [4] Chạy demo trên ảnh mẫu trong nhà
./run.sh 4
```

---

## 📱 Hướng dẫn Dùng Camera Điện thoại (iPhone / Android)

Hệ thống Web Dashboard đã được tích hợp sẵn **HTTPS** để hỗ trợ camera di động qua mạng Wi-Fi:

1. Chạy máy chủ: `./run.sh 1`
2. Mở trình duyệt trên điện thoại (Safari hoặc Chrome) và truy cập đường dẫn `https://<IP_MACBOOK>:8000` (ví dụ: `https://192.168.1.47:8000`).
3. Khi trình duyệt cảnh báo chứng chỉ SSL nội bộ:
   - Chọn **Nâng cao (Advanced)** -> Bấm **Tiếp tục truy cập (Proceed)**.
4. Bấm nút **"Bật Camera"** và cấp quyền để quét vật cản trực tiếp bằng camera sau của điện thoại.

---

## 🎯 15 Lớp Vật thể Hỗ trợ (Indoor Classes)

| STT | Tên Tiếng Việt | Tên Tiếng Anh | Chiều cao vật lý | Mức độ ưu tiên |
|:---:|:---|:---|:---:|:---:|
| 1 | Cầu thang | `stairs` | 1.00 m | Mức 1 (Nguy hiểm cao) |
| 2 | Người | `person` | 1.65 m | Mức 1 (Nguy hiểm cao) |
| 3 | Cửa | `door` | 2.00 m | Mức 2 (Định hướng) |
| 4 | Ghế | `chair` | 0.85 m | Mức 2 (Vật cản ngã) |
| 5 | Ghế sofa | `couch` | 0.85 m | Mức 2 |
| 6 | Bàn | `table` | 0.75 m | Mức 2 (Tầm bụng) |
| 7 | Giường | `bed` | 0.60 m | Mức 2 |
| 8 | Tivi / Màn hình | `tv` | 0.60 m | Mức 3 |
| 9 | Tủ lạnh | `refrigerator` | 1.70 m | Mức 2 |
| 10 | Bồn cầu | `toilet` | 0.75 m | Mức 2 (Khu vệ sinh) |
| 11 | Bồn rửa | `sink` | 0.85 m | Mức 2 |
| 12 | Thùng rác | `trash_can` | 0.45 m | Mức 1 (Vật cản sàn) |
| 13 | Quạt | `fan` | 1.10 m | Mức 1 (Cánh/Dây điện) |
| 14 | Chai / Ly | `bottle_cup` | 0.22 m | Mức 3 (Đổ vỡ) |
| 15 | Vật cản chung | `obstacle` | 1.40 m | Mức 1 (Cột, cây, vali) |

---

## 🧪 Kiểm thử Hệ thống

Chạy test tự động để đảm bảo toàn bộ pipeline hoạt động hoàn hảo:
```bash
./run.sh 3
# hoặc: PYTHONPATH=. python tests/test_system.py
```
- ✅ Kiểm tra định nghĩa 15 lớp và dữ liệu vật lý
- ✅ Kiểm tra công thức Pinhole Camera và phân vùng 3D (Trái, Phía trước, Phải)
- ✅ Kiểm tra cơ chế chống lặp từ và phát âm thanh tiếng Việt
- ✅ Kiểm tra mô hình YOLOv8n và vẽ AR HUD
- ✅ Kiểm tra toàn bộ REST API endpoints của FastAPI Web Server
