from ultralytics import YOLO
import cv2

model = YOLO("D:/nhandangtraicay/runs/detect/train18/weights/best.pt")

results = model("dataset/test/images")

for r in results:

    # hiển thị ảnh có bounding box
    img = r.plot()
    cv2.imshow("Result", img)
    cv2.waitKey(0)

    # đếm trái cây
    counts = {}

    for c in r.boxes.cls:
        name = model.names[int(c)]

        if name not in counts:
            counts[name] = 0

        counts[name] += 1

    print("Fruit Count:")
    for k,v in counts.items():
        print(k,":",v)

cv2.destroyAllWindows()
