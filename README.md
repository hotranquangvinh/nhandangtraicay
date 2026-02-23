# nhandangtraicay

# Nhận dạng trái cây

## Cấu trúc thư mục
- backend/: FastAPI phục vụ dự đoán
- frontend/: giao diện web (tĩnh)
- models/: fruit_model.h5, labels.txt
- dataset/train, dataset/valid: dữ liệu ảnh cho huấn luyện/validation (5 lớp Apple, Banana, Grape, Mango, Strawberry)

## Yêu cầu
- Python 3.9+ (thử với TensorFlow 2.x)
- Cài đặt thư viện backend:
  ```
  cd backend
  pip install -r requirements.txt
  ```

## Huấn luyện mô hình
1) Vào thư mục train:
	```
	cd train
	python train.py
	```
2) Kết quả:
	- models/fruit_model.h5
	- models/labels.txt

Lưu ý: tỷ lệ train/val do bạn đặt ảnh vào dataset/train và dataset/valid. Hiện tại loader không tạo tập test.

## Chạy backend (API)
Từ thư mục backend:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API: POST http://127.0.0.1:8000/predict (form-data: file)

Ngưỡng lọc không-phải-trái-cây: xem backend/predict.py
- CONF_THRESHOLD = 0.8 (độ tin cậy tối thiểu)
- MARGIN_THRESHOLD = 0.1 (khoảng cách top1 - top2)
Nếu không đạt, trả về class="unknown", confidence=0.

## Chạy frontend
Chạy server tĩnh (ví dụ từ thư mục frontend):
```
cd frontend
python -m http.server 5500
```
Mở http://127.0.0.1:5500/ và giữ backend chạy ở cổng 8000.

## Sử dụng
1) Mở trang web, chọn ảnh, bấm "Dự đoán".
2) Ảnh xem trước hiện ngay; kết quả hiển thị dưới khung.
3) Nếu backend không chạy, trang sẽ báo lỗi kết nối API.
