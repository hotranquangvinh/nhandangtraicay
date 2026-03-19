---
title: Nhan Dang Va Dem So Luong Trai Cay
emoji: 🍍
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# He thong nhan dang va dem so luong trai cay (YOLOv8 + Flask)

Du an xay dung he thong nhan dang trai cay tu anh hoac webcam, dong thoi dem so luong theo tung loai. He thong su dung YOLOv8 de phat hien doi tuong va Flask de trien khai giao dien web.

## 1) Muc tieu du an

- Nhan dang nhieu loai trai cay trong 1 anh/khung hinh.
- Dem so luong theo tung lop trai cay.
- Hien thi do tin cay nhan dang.
- Cung cap thong tin tham khao (dinh duong, cong dung, goi y mon an) cho trai cay da phat hien.
- Ho tro 2 che do su dung:
  - Tai anh len web.
  - Nhan dang truc tiep bang webcam tren trinh duyet.

## 2) Cau truc thu muc chinh

- `dataset/`: Du lieu train/valid/test theo dinh dang YOLO.
- `data.yaml`: Cau hinh duong dan dataset va danh sach class.
- `train.py`: Script huan luyen YOLOv8.
- `predict.py`: Thu nghiem nhan dang tren tap anh, hien thi ket qua bang OpenCV.
- `count.py`: Script dem so luong trai cay tren tap test.
- `webcam.py`: Nhan dang real-time bang webcam desktop (OpenCV window).
- `app.py`: Ung dung Flask (upload anh + webcam browser + tra JSON ket qua).
- `templates/`: Giao dien HTML (`index.html`, `webcam.html`).
- `runs/detect/`: Luu cac lan train (`train18`, `train19`, ...).

## 3) Cong nghe su dung

- Python 3.x
- Ultralytics YOLOv8
- OpenCV
- Flask
- NumPy

## 4) Cai dat moi truong

## 4.1) Chay nhanh bang Docker (khuyen nghi khi chia se GitHub)

Chi can cai Docker Desktop (hoac Docker Engine + Docker Compose). Khong can cai Python, pip, hay thu vien trong may.

### Cach 1 - Docker Compose (de nhat)

```bash
docker compose up --build
```

Mo trinh duyet tai:

- `http://127.0.0.1:7860/` (upload anh)
- `http://127.0.0.1:7860/webcam` (nhan dang webcam)

### Cach 2 - Docker thuong

```bash
docker build -t fruit-detector .
docker run --rm -p 7860:7860 --name fruit-detector fruit-detector
```

### Luu y model khi share repo

- Uu tien dat file model `best.pt` o thu muc goc repo va commit len GitHub.
- Hoac dat bien moi truong `MODEL_PATH` (xem `docker-compose.yml`).
- Neu khong co custom model, app se fallback sang `yolov8n.pt` khi `ALLOW_BASE_MODEL=1`.
- Voi file model lon, nen dung Git LFS de tranh gioi han dung luong tren GitHub.

### B1. Tao va kich hoat moi truong ao (khuyen nghi)

Windows (CMD):

```bash
python -m venv venv
venv\Scripts\activate
```

### B2. Cai thu vien

```bash
pip install --upgrade pip
pip install ultralytics flask opencv-python numpy
```

## 5) Quy trinh xay dung he thong tu dau den hien tai

### Giai doan 1 - Chuan bi du lieu

1. Thu thap/nhap bo du lieu trai cay (Roboflow metadata da co trong `data.yaml`).
2. To chuc theo cau truc YOLO:
	- `dataset/train/images`, `dataset/train/labels`
	- `dataset/valid/images`, `dataset/valid/labels`
	- `dataset/test/images`, `dataset/test/labels`
3. Cau hinh `data.yaml`:
	- `train`, `val`, `test`
	- `nc: 30`
	- `names`: danh sach 30 lop trai cay

### Giai doan 2 - Lam sach va chuan hoa nhan

1. Ra soat nhan sai trong dataset.
2. Da xu ly truong hop nham nhan Peach vao class coconut, va dua Peach ve class dung.
3. Dataset van co mot so file nhan tron giua detect/segment; Ultralytics van train detect duoc va bo qua phan segment (canh bao se xuat hien trong log).

### Giai doan 3 - Huan luyen mo hinh

1. Khoi tao model tu pretrain weight (`yolov8n.pt`).
2. Train voi cau hinh hien tai trong `train.py`:
	- `epochs=15`
	- `imgsz=512`
	- `batch=12`
	- `patience=5`
	- `cache="disk"`
	- `cos_lr=True`
	- `device="cpu"`
3. Sau moi lan train, ket qua duoc luu trong `runs/detect/trainXX/`.

### Giai doan 4 - Danh gia nhanh va dem so luong

1. Dung `predict.py` de test nhan dang tren `dataset/test/images` va xem anh da ve bounding box.
2. Dung `count.py` de tong hop so luong theo class tren tap test.
3. So sanh cac lan train (`train18`, `train19`, ...) de chon mo hinh tot nhat.

### Giai doan 5 - Trien khai web app

1. Xay dung Flask app trong `app.py`.
2. Tai model tot nhat de suy luan (hien tai dang tro den `runs/detect/train19/weights/best.pt`).
3. Ho tro API va giao dien:
	- Trang upload anh `/`
	- Trang webcam `/webcam`
	- API nhan frame webcam `/webcam-detect`
4. Bo sung logic tong hop ket qua:
	- Dem so luong tung loai
	- Tinh do tin cay trung binh/cao nhat
	- Dich ten class sang tieng Viet
	- Tra ve thong tin dinh duong + cong dung + goi y

### Giai doan 6 - Toi uu trai nghiem nguoi dung

1. Giao dien responsive cho desktop/mobile.
2. Hien thi anh ket qua co bbox + bang tong hop theo loai trai cay.
3. Webcam cap nhat ket qua theo chu ky (khoang 1.4s/frame gui len server).

## 6) Cac buoc chay he thong hien tai

### Cach A - Chay web app (khuyen nghi)

```bash
python app.py
```

Mo trinh duyet tai:

- `http://127.0.0.1:5000/` (upload anh)
- `http://127.0.0.1:5000/webcam` (nhan dang webcam)

### Cach B - Chay train

```bash
python train.py
```

### Cach C - Test nhan dang bang script

```bash
python predict.py
```

### Cach D - Dem so luong bang script

```bash
python count.py
```

### Cach E - Nhan dang webcam bang OpenCV window

```bash
python webcam.py
```

## 7) Luong xu ly trong he thong (pipeline)

1. Dau vao:
	- Anh upload hoac frame webcam.
2. Tien xu ly:
	- Doc anh thanh ma tran NumPy/OpenCV.
3. Suy luan:
	- YOLOv8 tra ve bbox, class, confidence.
4. Hau xu ly:
	- Gom nhom theo class.
	- Dem so luong.
	- Tinh thong ke do tin cay.
	- Anh xa ten class sang tieng Viet.
	- Gan thong tin tham khao theo loai trai cay.
5. Dau ra:
	- Anh da ve bbox.
	- Bang tong hop ket qua nhan dang.
	- Thong tin tham khao cho nguoi dung.

## 8) Tinh trang hien tai

- He thong da hoan thanh chu trinh co ban: du lieu -> train -> test -> app web.
- Da co nhieu lan train trong `runs/detect/`; app dang dung model tu `train19`.
- Da co tinh nang nhan dang qua upload anh va webcam browser.

## 9) Han che va huong nang cap tiep theo

- Du lieu can tiep tuc lam sach de giam nhan sai/nhieu.
- Nen bo sung metric danh gia ro rang (mAP50, mAP50-95, precision, recall) vao README moi lan train.
- Can tach file `requirements.txt` de cai dat dong nhat tren may khac.
- Co the nang cap:
  - Chon model qua bien moi truong (khong hard-code duong dan).
  - Them luu lich su nhan dang vao CSDL.
  - Dong goi bang Docker de trien khai de dang hon.

## 10) Ghi chu van hanh

- Neu doi model, cap nhat duong dan weight trong `app.py`, `count.py`, `predict.py`, `webcam.py`.
- Neu train tren GPU, doi `device` trong `train.py` thanh `0` (hoac GPU id phu hop).
- Dam bao quyen camera trinh duyet khi dung trang webcam.

## 11) Danh sach trai cay
30 loại
- Durian - sau rieng
- Mulberries - dau tam
- Raspberry - mam xoi
- Red pomegranate - luu do
- apple - tao
- avocado - bo
- banana - chuoi
- blueberry - viet quat
- cantaloupe - dua luoi
- carambola - khe
- cherry - anh dao
- coconut - dua
- grapefruit - buoi
- grapes - nho
- green apple - tao xanh
- green grapes - nho xanh
- guava - oi
- kiwi - kiwi
- lemon - chanh
- litchi - vai
- mango - xoai
- orange - cam
- papaya - du du
- passion fruit - chanh day
- pear - le
- pineapple - dua
- pitaya - thanh long
- strawberry - dau tay
- watermelon - dua hau
- peach - dao

chạy: d:/nhandangvademsoluongtraicay/venv/Scripts/python.exe app.py

yolov8n: n = nano
Công dụng: rất nhẹ, chạy nhanh, phù hợp máy yếu, realtime webcam, demo nhanh.
Đổi lại: độ chính xác thường thấp hơn các bản lớn hơn.

yolov8s: s = small
Công dụng: vẫn khá nhẹ nhưng mạnh hơn yolov8n, thường cho kết quả nhận diện tốt hơn.
Đổi lại: nặng hơn, train lâu hơn, infer chậm hơn một chút, tốn VRAM/RAM hơn.