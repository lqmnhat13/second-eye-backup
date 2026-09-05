# Rà soát nhận diện và khoảng cách

## Kết luận từ mã và trọng số hiện có

- `yolov8n.pt` chỉ ánh xạ được 11/15 nhóm trong cấu hình. Không có cầu thang, cửa, quạt, thùng rác. Thêm tên lớp trong Python không huấn luyện được lớp mới. Nhóm “vật cản” cũng chỉ bao gồm những lớp đã ánh xạ, không phải mọi vật cản.
- Trước thay đổi, hệ thống không có mạng dự đoán độ sâu. Khoảng cách phụ thuộc kích thước trung bình; gộp sách, mèo, vali vào vật cản cao 1,4 m dẫn tới sai số lớn. Confidence YOLO không phải độ tin cậy của khoảng cách.
- Heuristic tư thế người từ tỷ lệ hộp chưa được kiểm chứng. Hộp bị che khuất, vật khác kích thước chuẩn, chuyển camera và camera bị crop vẫn gây sai số.
- Các kiểm thử cũ chủ yếu kiểm tra chạy được, không đo precision/recall, mAP hoặc sai số khoảng cách.

## Thay đổi đã thực hiện

- Báo rõ lớp mà trọng số không hỗ trợ; ánh xạ đầy đủ alias cấu hình. Độ phân giải suy luận mặc định tăng 480 → 640; cần đo FPS và recall trên phần cứng đích.
- Giữ calibration đã lưu khi khởi động desktop/CLI; quy đổi tiêu cự từ hệ tọa độ tham chiếu 640×480 khi ảnh được resize. Hiệu chuẩn từ chiều cao hộp nhận cả chiều cao ảnh. Không tự xử lý camera crop hoặc đổi camera; cần hiệu chuẩn riêng.
- Bỏ suy luận mặt sàn không có bằng chứng tiếp xúc sàn/pose camera.
- Track chỉ được dùng một lần trong mỗi frame, hết hạn được dọn trước khi ghép. Khoảng cách giảm được phản ánh ngay, giảm độ trễ cảnh báo; đổi lại có thể tăng cảnh báo giả khi dự đoán nhiễu. IoU vẫn có thể đổi danh tính khi hai vật giao nhau.
- Gắn nguồn ước lượng và reliability định tính `low`, không giả làm xác suất đã hiệu chuẩn.
- Backend metric depth tùy chọn, chỉ nạp local, từ chối model relative depth. Lấy median vùng giữa hộp, loại vùng thiếu dữ liệu hoặc độ sâu phân tán cao. Đây chưa phải segmentation: vẫn có thể lấy nhầm nền đồng nhất. ROI bị loại quay về size prior và được ghi nguồn tương ứng.

## Bật nhánh độ sâu để thử nghiệm

Cài `requirements-depth.txt` vào môi trường dự án. Tải trước model **depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf** vào một thư mục local theo hướng dẫn model card, rồi chạy:

```bash
SECOND_EYE_DEPTH_MODEL=/absolute/path/to/local-model python main.py
```

Không tự tải hoặc bật model này mặc định. Chưa chạy end-to-end nhánh này với trọng số thực; cần kiểm tra tương thích transformers, tốc độ và chất lượng trên camera đích. Model lỗi sẽ báo lỗi, không âm thầm coi relative depth là mét.

Nguồn: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf

## Thực nghiệm cần có cho đồ án

1. Thu thập ảnh/video tại hành lang, phòng học và cầu thang IUH với nhãn hộp cho 15 lớp. Chia train/validation/test theo phòng hoặc phiên quay; tránh chia ngẫu nhiên các frame liền nhau gây rò rỉ dữ liệu.
2. Fine-tune detector trên đúng lớp còn thiếu. So sánh baseline nano với model lớn hơn trên cùng test set; báo mAP50–95, precision/recall từng lớp, đặc biệt recall cầu thang và vật cản nhỏ. Không chọn chỉ theo mAP tổng.
3. Đo khoảng cách thật tới bề mặt vật thể, thống nhất độ sâu trục Z với khoảng cách Euclid. Lưu CSV `ground_truth_m,predicted_m`, đánh giá bằng `python scripts/evaluate_distance.py measurements.csv`. Báo MAE, RMSE, AbsRel, delta1 và tỷ lệ ước lượng xa hơn thực tế trên 0,5 m. Đếm riêng trường hợp bỏ sót/không dự đoán, không loại chúng khỏi báo cáo tổng.
4. So sánh size prior với metric depth trên cùng tập, chia theo khoảng cách, ánh sáng, che khuất và lớp vật. Báo số mẫu và bootstrap confidence interval theo phiên quay, không theo frame độc lập.
5. Đo độ trễ đầu-cuối tới cảnh báo, FPS và p95 latency trên thiết bị đích; đánh giá khi đi tới vật cản, đổi camera và mất frame. Kiểm tra cảnh báo âm thanh vì cooldown cũng ảnh hưởng độ trễ.

## Kiểm chứng trong lần sửa này

7 kiểm thử hồi quy headless đạt; YOLO thực trên CPU và vẽ HUD ảnh tổng hợp đạt. Chưa kiểm thử camera/GUI/OCR/âm thanh toàn hệ thống; chưa huấn luyện lại; chưa có số đo chứng minh mAP hoặc độ sâu tăng. Không diễn giải trạng thái SAFE của thuật toán là chứng nhận đường đi không có vật cản.
