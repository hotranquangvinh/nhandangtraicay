from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=15,
    imgsz=512,
    batch=12,
    patience=5,
    cache="disk",
    cos_lr=True,
    device="cpu",
    workers=0,
    name="train30_fast"
)
