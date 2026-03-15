from ultralytics import YOLO

model = YOLO("D:/nhandangtraicay/runs/detect/train18/weights/best.pt")

results = model("D:/nhandangtraicay/dataset/test/images")

counts = {}

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        if name not in counts:
            counts[name] = 0

        counts[name] += 1

print("Fruit count:")
for k,v in counts.items():
    print(k,":",v)
