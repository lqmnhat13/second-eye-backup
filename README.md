# Second Eye - Indoor Navigation & Obstacle Warning System for the Visually Impaired
### Hệ Thống Hỗ Trợ Thị Giác Trong Nhà Cho Người Khiếm Thị

Second Eye là hệ thống thị giác máy tính AI kết hợp mô hình ước lượng khoảng cách không gian thực và công nghệ cảnh báo giọng nói Tiếng Việt, giúp người khiếm thị nhận diện vật thể xung quanh, đánh giá mức độ nguy hiểm theo thời gian thực và định hướng di chuyển an toàn trong nhà.

---

## 🌟 Tính Năng Nổi Bật

1. **Nhận Diện 15 Lớp Đối Tượng Trong Nhà Thiết Yếu**:
   - `stairs` (Cầu thang - Cảnh báo khẩn cấp)
   - `person` (Người)
   - `door` (Cửa ra vào / Cửa phòng)
   - `chair` (Ghế)
   - `couch` (Ghế sofa)
   - `table` (Bàn ăn / Bàn làm việc)
   - `bed` (Giường)
   - `tv` (Tivi / Màn hình)
   - `refrigerator` (Tủ lạnh)
   - `toilet` (Bồn cầu)
   - `sink` (Bồn rửa)
   - `trash_can` (Thùng rác)
   - `fan` (Quạt đứng / Quạt sàn)
   - `bottle_cup` (Chai / Ly nước)
   - `obstacle` (Cột / Vách ngăn / Cây cảnh / Vật cản chung)

2. **Ước Lượng Khoảng Cách Thời Gian Thực (Metric Distance Estimation)**:
   - Sử dụng mô hình hình học Pinhole kết hợp kích thước vật lý chuẩn của từng loại vật thể.
   - Ước lượng khoảng cách theo mét chính xác và ổn định ở tốc độ 30 - 60 FPS mà không cần phần cứng cảm biến LiDAR đắt tiền.

3. **Định Vị Vùng Không Gian (Spatial Zone Partitioning)**:
   - Phân tích tọa độ vật thể thành 3 hướng chính: **Bên trái**, **Phía trước / Ở giữa**, **Bên phải**.
   - Tự động ưu tiên cảnh báo các vật cản nằm trực diện quỹ đạo di chuyển của người dùng.

4. **Cảnh Báo Thông Minh Đa Tầng Bằng Tiếng Việt (Vietnamese Voice Alerts)**:
   - 🔴 **Vùng Nguy Hiểm ($< 1.0m$)**: Âm báo khẩn cấp + Giọng đọc tức thì (*"Nguy hiểm! Cầu thang ngay phía trước, cách không phẩy tám mét!"*).
   - 🟡 **Vùng Cảnh Giác ($1.0m - 2.0m$)**: Giọng đọc định hướng nhẹ nhàng (*"Có ghế bên phải, cách một phẩy hai mét."*).
   - 🟢 **Vùng An Toàn ($> 2.0m$)**: Giữ yên lặng để chống ô nhiễm tiếng ồn.
   - **Smart Debounce Cooldown**: Bộ đệm chống nói lặp lại gây khó chịu, tự động ngắt câu cũ khi có nguy cơ va chạm mới.

5. **Giao Diện Hiện Đại & Radar Không Gian 2D**:
   - Giao diện Web Dashboard Dark Glassmorphism cao cấp.
   - **Radar 2D Top-down**: Mô phỏng vị trí các vật thể xung quanh người dùng từ trên cao với tia quét radar thời gian thực.
   - Tích hợp phím tắt trợ năng hoàn chỉnh cho người khiếm thị.

---

## 🚀 Hướng Dẫn Khởi Động Nhanh

### 1. Khởi động Web Dashboard (Khuyên dùng)
```bash
./run.sh 1
# hoặc: python web_app.py
```
Truy cập trình duyệt tại: **`http://localhost:8000`**

### 2. Khởi động Desktop OpenCV HUD (Chạy trực tiếp từ Terminal)
```bash
./run.sh 2
# hoặc: python main_cli.py
```

### 3. Chạy Kiểm Thử Hệ Thống (Unit & Integration Tests)
```bash
./run.sh 3
# hoặc: python test_sample.py
```

### 4. Chạy Thử Nghiệm Trên Ảnh Mẫu
```bash
./run.sh 4
# hoặc: python demo_inference.py
```

---

## ⌨️ Phím Tắt Trợ Năng Cho Người Khiếm Thị

| Phím tắt | Chức năng |
|---|---|
| <kbd>Space</kbd> | Bật / Tạm dừng luồng Camera |
| <kbd>V</kbd> | Bật / Tắt giọng đọc cảnh báo Tiếng Việt |
| <kbd>R</kbd> | Yêu cầu đọc lại cảnh báo vật cản gần nhất ngay lập tức |
| <kbd>+</kbd> / <kbd>-</kbd> | Tăng / Giảm tiêu cự camera để hiệu chuẩn khoảng cách |
| <kbd>M</kbd> | Chuyển đổi giữa Camera trước và Camera sau |
| <kbd>Esc</kbd> | Dừng camera và đóng tất cả cửa sổ cài đặt |

---

## 📁 Cấu Trúc Dự Án

```
Second-Eye/
├── config.py              # Cấu hình 15 lớp, chiều cao vật lý, ngưỡng khoảng cách
├── detector.py            # AI Object Detection Engine (YOLOv8 + Apple MPS/CUDA/CPU)
├── distance_estimator.py  # Thuật toán Pinhole geometry & tọa độ không gian 3D
├── alert_system.py        # Quản lý hàng đợi âm thanh & giọng nói Tiếng Việt
├── web_app.py             # FastAPI backend API & WebSocket/Streaming server
├── main_cli.py            # Ứng dụng Desktop OpenCV HUD tương tác
├── test_sample.py         # Bộ kiểm thử tự động toàn diện
├── demo_inference.py      # Script demo nhận diện ảnh mẫu
├── run.sh                 # Script khởi động hệ thống tiện lợi
├── requirements.txt       # Danh sách thư viện phụ thuộc
├── templates/
│   └── index.html         # Giao diện Web Dashboard (Radar 2D & AR HUD)
└── static/
    ├── style.css          # CSS Glassmorphism Dark Mode
    └── app.js             # Client Web Speech API, Radar Canvas & Webcam Stream
```
