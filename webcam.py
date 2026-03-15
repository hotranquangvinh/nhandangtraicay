from ultralytics import YOLO
import cv2


WINDOW_NAME = "Fruit Detection"
MODEL_PATH = "runs/detect/train19/weights/best.pt"


def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()

    raise RuntimeError("Khong mo duoc webcam. Kiem tra camera hoac quyen truy cap.")


def main():
    print("Dang tai model...", flush=True)
    model = YOLO(MODEL_PATH)

    print("Dang mo webcam...", flush=True)
    cap = open_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    print("Webcam da san sang. Nhan ESC de thoat.", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame tu webcam.", flush=True)
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
