from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

EXPECTED_FRUIT_CLASSES_EN = {
    "durian",
    "mulberries",
    "raspberry",
    "red pomegranate",
    "apple",
    "avocado",
    "banana",
    "blueberry",
    "cantaloupe",
    "carambola",
    "cherry",
    "coconut",
    "grapefruit",
    "grapes",
    "green apple",
    "green grapes",
    "guava",
    "kiwi",
    "lemon",
    "litchi",
    "mango",
    "orange",
    "papaya",
    "passion fruit",
    "pear",
    "pineapple",
    "pitaya",
    "peach",
    "strawberry",
    "watermelon",
}

COCO_NON_FRUIT_SENTINELS = {
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "cat",
    "dog",
}


def resolve_model_path() -> str:
    base_dir = Path(__file__).resolve().parent
    env_model_path = os.environ.get("MODEL_PATH")

    if env_model_path:
        env_path = Path(env_model_path)
        if not env_path.is_absolute():
            env_path = base_dir / env_path
        if env_path.exists():
            return str(env_path)

    candidate_paths = [
        base_dir / "best.pt",
        base_dir / "runs" / "detect" / "train19" / "weights" / "best.pt",
    ]

    for path in candidate_paths:
        if path.exists():
            return str(path)

    # Optional fallback for debugging only.
    if os.environ.get("ALLOW_BASE_MODEL", "0") == "1":
        # Returning a known Ultralytics weight name allows auto-download when missing.
        for path in [base_dir / "yolov8s.pt", base_dir / "yolov8n.pt"]:
            if path.exists():
                return str(path)
        return "yolov8n.pt"

    raise FileNotFoundError(
        "Khong tim thay custom model. Hay dat MODEL_PATH tro den model trai cay "
        "(vi du: runs/detect/train19/weights/best.pt). Neu build Docker, can dam "
        "bao .dockerignore khong loai file best.pt nay ra khoi build context."
    )


def validate_fruit_model(loaded_model) -> None:
    names = getattr(loaded_model, "names", {})

    if isinstance(names, dict):
        raw_names = names.values()
    elif isinstance(names, list):
        raw_names = names
    else:
        raw_names = []

    normalized_names = {str(name).strip().lower() for name in raw_names}
    if not normalized_names:
        raise ValueError("Model khong co danh sach class hop le (model.names rong).")

    allow_base_model = os.environ.get("ALLOW_BASE_MODEL", "0") == "1"
    verify_fruit_model = os.environ.get("VERIFY_FRUIT_MODEL", "1") == "1"

    if not verify_fruit_model or allow_base_model:
        return

    fruit_overlap = normalized_names.intersection(EXPECTED_FRUIT_CLASSES_EN)
    coco_overlap = normalized_names.intersection(COCO_NON_FRUIT_SENTINELS)

    if coco_overlap:
        raise ValueError(
            "Dang nap nham model tong quat (COCO) thay vi model trai cay. "
            f"Class nghi van: {sorted(coco_overlap)}. "
            "Hay kiem tra MODEL_PATH va file best.pt trong Docker image."
        )

    if len(fruit_overlap) < 8:
        raise ValueError(
            "Model dang nap khong giong model trai cay cua du an "
            f"(chi khop {len(fruit_overlap)} class trai cay). "
            "Hay dung dung best.pt da train cua ban."
        )


resolved_model_path = resolve_model_path()
model = YOLO(resolved_model_path)
validate_fruit_model(model)
print(f"[MODEL] Loaded: {resolved_model_path}", flush=True)
print(f"[MODEL] Class count: {len(model.names)}", flush=True)

FRUIT_LABELS_VI = {
    "durian": "sầu riêng",
    "mulberries": "dâu tằm",
    "raspberry": "mâm xôi",
    "red pomegranate": "lựu đỏ",
    "apple": "táo",
    "avocado": "bơ",
    "banana": "chuối",
    "blueberry": "việt quất",
    "cantaloupe": "dưa lưới",
    "carambola": "khế",
    "cherry": "anh đào",
    "coconut": "dừa",
    "grapefruit": "bưởi",
    "grapes": "nho",
    "green apple": "táo xanh",
    "green grapes": "nho xanh",
    "guava": "ổi",
    "kiwi": "kiwi",
    "lemon": "chanh",
    "litchi": "vải",
    "mango": "xoài",
    "orange": "cam",
    "papaya": "đu đủ",
    "passion fruit": "chanh dây",
    "pear": "lê",
    "pineapple": "dứa",
    "pitaya": "thanh long",
    "peach": "đào",
    "strawberry": "dâu tây",
    "watermelon": "dưa hấu",
}

FRUIT_INFO_VI = {
    "sầu riêng": {
        "dinh_duong": "Giàu năng lượng, chất xơ, vitamin C, kali và một số chất chống oxy hóa.",
        "cong_dung": "Hỗ trợ bổ sung năng lượng nhanh, giúp no lâu và góp phần cân bằng điện giải.",
        "goi_y": ["Sinh tố sầu riêng", "Kem sầu riêng", "Xôi sầu riêng"],
    },
    "dâu tằm": {
        "dinh_duong": "Chứa vitamin C, sắt, anthocyanin và chất xơ.",
        "cong_dung": "Hỗ trợ chống oxy hóa, tốt cho máu và giúp thanh mát cơ thể.",
        "goi_y": ["Si rô dâu tằm", "Trà dâu tằm", "Mứt dâu tằm"],
    },
    "mâm xôi": {
        "dinh_duong": "Giàu chất xơ, vitamin C, mangan và polyphenol.",
        "cong_dung": "Hỗ trợ tiêu hóa, chống oxy hóa và phù hợp bữa ăn nhẹ lành mạnh.",
        "goi_y": ["Sinh tố mâm xôi", "Sữa chua mâm xôi", "Mứt mâm xôi"],
    },
    "lựu đỏ": {
        "dinh_duong": "Giàu vitamin C, vitamin K, folate và chất chống oxy hóa.",
        "cong_dung": "Hỗ trợ tim mạch, làm đẹp da và bổ sung nước.",
        "goi_y": ["Nước ép lựu", "Salad lựu", "Soda lựu"],
    },
    "táo": {
        "dinh_duong": "Chứa chất xơ hòa tan, vitamin C và các hợp chất chống oxy hóa.",
        "cong_dung": "Hỗ trợ tiêu hóa, giúp no lâu và phù hợp ăn nhẹ hằng ngày.",
        "goi_y": ["Nước ép táo", "Salad táo", "Táo yến mạch"],
    },
    "bơ": {
        "dinh_duong": "Giàu chất béo tốt, vitamin E, kali và folate.",
        "cong_dung": "Hỗ trợ tim mạch, tốt cho da và tạo cảm giác no lâu.",
        "goi_y": ["Sinh tố bơ", "Bánh mì bơ", "Salad bơ"],
    },
    "chuối": {
        "dinh_duong": "Giàu kali, vitamin B6, carbohydrate và chất xơ.",
        "cong_dung": "Hỗ trợ phục hồi năng lượng, tốt cho cơ bắp và tiêu hóa.",
        "goi_y": ["Sinh tố chuối", "Chuối yến mạch", "Chuối nướng"],
    },
    "việt quất": {
        "dinh_duong": "Giàu anthocyanin, vitamin C, vitamin K và mangan.",
        "cong_dung": "Hỗ trợ chống oxy hóa mạnh, tốt cho mắt và não bộ.",
        "goi_y": ["Sữa chua việt quất", "Sinh tố việt quất", "Muffin việt quất"],
    },
    "dưa lưới": {
        "dinh_duong": "Chứa nhiều nước, vitamin A, vitamin C và kali.",
        "cong_dung": "Giúp giải nhiệt, bổ sung nước và hỗ trợ làn da.",
        "goi_y": ["Sinh tố dưa lưới", "Salad dưa lưới", "Nước ép dưa lưới"],
    },
    "khế": {
        "dinh_duong": "Giàu vitamin C, chất xơ và các hợp chất thực vật.",
        "cong_dung": "Hỗ trợ tiêu hóa, tạo vị thanh mát và chống oxy hóa.",
        "goi_y": ["Nước ép khế", "Khế dầm", "Salad khế"],
    },
    "anh đào": {
        "dinh_duong": "Chứa vitamin C, kali và anthocyanin.",
        "cong_dung": "Hỗ trợ chống oxy hóa và phù hợp món tráng miệng nhẹ.",
        "goi_y": ["Cherry yogurt", "Nước ép cherry", "Bánh cherry"],
    },
    "dừa": {
        "dinh_duong": "Giàu nước, điện giải tự nhiên, mangan và chất béo thực vật.",
        "cong_dung": "Bù nước tốt, hỗ trợ giải nhiệt và phù hợp ngày nóng.",
        "goi_y": ["Nước dừa", "Sinh tố dừa", "Chè dừa"],
    },
    "bưởi": {
        "dinh_duong": "Giàu vitamin C, chất xơ và flavonoid.",
        "cong_dung": "Hỗ trợ miễn dịch, tạo cảm giác nhẹ bụng và thanh mát.",
        "goi_y": ["Gỏi bưởi", "Nước ép bưởi", "Trà bưởi mật ong"],
    },
    "nho": {
        "dinh_duong": "Chứa vitamin K, vitamin C và resveratrol.",
        "cong_dung": "Hỗ trợ chống oxy hóa và phù hợp làm món ăn nhẹ tiện lợi.",
        "goi_y": ["Nho lạnh", "Nước ép nho", "Salad nho"],
    },
    "táo xanh": {
        "dinh_duong": "Giàu chất xơ, vitamin C và vị chua nhẹ dễ ăn.",
        "cong_dung": "Hỗ trợ tiêu hóa, phù hợp người thích vị thanh ít ngọt.",
        "goi_y": ["Táo xanh lắc", "Nước ép táo xanh", "Salad táo xanh"],
    },
    "nho xanh": {
        "dinh_duong": "Giàu nước, vitamin C, vitamin K và chất chống oxy hóa.",
        "cong_dung": "Giúp bổ sung nước, làm món ăn vặt mát và dễ dùng.",
        "goi_y": ["Nho xanh ướp lạnh", "Soda nho xanh", "Salad nho xanh"],
    },
    "ổi": {
        "dinh_duong": "Rất giàu vitamin C, chất xơ và folate.",
        "cong_dung": "Hỗ trợ miễn dịch, tốt cho tiêu hóa và giúp no lâu.",
        "goi_y": ["Ổi lắc", "Nước ép ổi", "Salad ổi"],
    },
    "kiwi": {
        "dinh_duong": "Giàu vitamin C, vitamin E, kali và chất xơ.",
        "cong_dung": "Hỗ trợ miễn dịch, tiêu hóa và tăng cảm giác tươi mát.",
        "goi_y": ["Kiwi smoothie", "Salad kiwi", "Kiwi yogurt"],
    },
    "chanh": {
        "dinh_duong": "Giàu vitamin C, acid citric và hợp chất thơm tự nhiên.",
        "cong_dung": "Hỗ trợ giải khát, tăng vị món ăn và làm đồ uống mát.",
        "goi_y": ["Nước chanh", "Chanh mật ong", "Trà chanh"],
    },
    "vải": {
        "dinh_duong": "Chứa vitamin C, đồng, kali và lượng nước cao.",
        "cong_dung": "Giúp giải nhiệt, bổ sung nước và tạo vị ngọt thơm.",
        "goi_y": ["Trà vải", "Vải ngâm lạnh", "Soda vải"],
    },
    "xoài": {
        "dinh_duong": "Giàu vitamin A, vitamin C, folate và chất xơ.",
        "cong_dung": "Hỗ trợ thị lực, miễn dịch và phù hợp nhiều món tráng miệng.",
        "goi_y": ["Sinh tố xoài", "Xoài lắc", "Sticky rice xoài"],
    },
    "cam": {
        "dinh_duong": "Giàu vitamin C, folate, kali và các flavonoid.",
        "cong_dung": "Hỗ trợ miễn dịch, giải khát và bổ sung chất chống oxy hóa.",
        "goi_y": ["Nước cam", "Cam lát mật ong", "Salad cam"],
    },
    "đu đủ": {
        "dinh_duong": "Chứa vitamin A, vitamin C, folate và enzyme papain.",
        "cong_dung": "Hỗ trợ tiêu hóa, làm mềm thực phẩm và tốt cho da.",
        "goi_y": ["Sinh tố đu đủ", "Đu đủ dầm", "Salad đu đủ"],
    },
    "chanh dây": {
        "dinh_duong": "Giàu vitamin C, vitamin A, chất xơ và hương thơm tự nhiên.",
        "cong_dung": "Giúp giải khát, tăng hương vị và hỗ trợ chống oxy hóa.",
        "goi_y": ["Nước chanh dây", "Soda chanh dây", "Mousse chanh dây"],
    },
    "lê": {
        "dinh_duong": "Giàu chất xơ, vitamin C và nước.",
        "cong_dung": "Hỗ trợ tiêu hóa, tạo cảm giác mát và dễ ăn.",
        "goi_y": ["Lê hấp đường phèn", "Nước ép lê", "Salad lê"],
    },
    "dứa": {
        "dinh_duong": "Chứa vitamin C, mangan và enzyme bromelain.",
        "cong_dung": "Hỗ trợ tiêu hóa, tạo vị chua ngọt và rất hợp món giải nhiệt.",
        "goi_y": ["Nước ép dứa", "Dứa sấy", "Trà dứa"],
    },
    "thanh long": {
        "dinh_duong": "Giàu chất xơ, vitamin C, magie và nhiều nước.",
        "cong_dung": "Hỗ trợ tiêu hóa, thanh mát và phù hợp món ăn nhẹ.",
        "goi_y": ["Sinh tố thanh long", "Thanh long dầm", "Nước ép thanh long"],
    },
    "đào": {
        "dinh_duong": "Chứa vitamin A, vitamin C, kali và các hợp chất chống oxy hóa.",
        "cong_dung": "Hỗ trợ làn da, giải khát và phù hợp đồ uống trái cây.",
        "goi_y": ["Trà đào", "Đào ngâm", "Sinh tố đào"],
    },
    "dâu tây": {
        "dinh_duong": "Giàu vitamin C, mangan, folate và chất chống oxy hóa.",
        "cong_dung": "Hỗ trợ miễn dịch, làm đẹp da và tạo hương vị dễ dùng.",
        "goi_y": ["Dâu tây sữa chua", "Sinh tố dâu", "Mứt dâu"],
    },
    "dưa hấu": {
        "dinh_duong": "Giàu nước, vitamin A, vitamin C và lycopene.",
        "cong_dung": "Giúp giải nhiệt rất tốt, bổ sung nước và thanh mát.",
        "goi_y": ["Nước ép dưa hấu", "Salad dưa hấu", "Dưa hấu lạnh"],
    },
}


def translate_label(label):
    return FRUIT_LABELS_VI.get(label.strip().lower(), label)


def get_translated_names():
    return {
        index: translate_label(name)
        for index, name in model.names.items()
    }


def build_detection_summary(result, translated_names):
    counts = {}
    summary_map = {}

    if result.boxes is None or len(result.boxes) == 0:
        return counts, []

    class_ids = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()

    for class_id, confidence in zip(class_ids, confidences):
        fruit_name = translated_names[int(class_id)]
        confidence_percent = round(float(confidence) * 100, 1)

        counts[fruit_name] = counts.get(fruit_name, 0) + 1
        summary = summary_map.setdefault(
            fruit_name,
            {
                "ten": fruit_name,
                "so_luong": 0,
                "cac_muc_tin_cay": [],
            },
        )
        summary["so_luong"] += 1
        summary["cac_muc_tin_cay"].append(confidence_percent)

    detection_summary = []

    for fruit_name, summary in summary_map.items():
        confidence_values = summary["cac_muc_tin_cay"]
        detection_summary.append(
            {
                "ten": fruit_name,
                "so_luong": summary["so_luong"],
                "do_tin_cay_trung_binh": round(sum(confidence_values) / len(confidence_values), 1),
                "do_tin_cay_cao_nhat": round(max(confidence_values), 1),
                "cac_muc_tin_cay": confidence_values,
            }
        )

    detection_summary.sort(
        key=lambda item: (-item["so_luong"], -item["do_tin_cay_cao_nhat"], item["ten"])
    )

    return counts, detection_summary


def get_fruit_details(detection_summary):
    details = []

    if not detection_summary:
        return details

    for item in detection_summary:
        fruit_name = item["ten"]
        amount = item["so_luong"]
        info = FRUIT_INFO_VI.get(
            fruit_name,
            {
                "dinh_duong": "Chưa có thông tin chi tiết cho loại trái cây này.",
                "cong_dung": "Có thể dùng như một phần của chế độ ăn đa dạng, cân đối.",
                "goi_y": ["Ăn tươi", "Làm nước ép", "Kết hợp salad trái cây"],
            },
        )
        details.append(
            {
                "ten": fruit_name,
                "so_luong": amount,
                "do_tin_cay_trung_binh": item["do_tin_cay_trung_binh"],
                "do_tin_cay_cao_nhat": item["do_tin_cay_cao_nhat"],
                "cac_muc_tin_cay": item["cac_muc_tin_cay"],
                "dinh_duong": info["dinh_duong"],
                "cong_dung": info["cong_dung"],
                "goi_y": info["goi_y"],
            }
        )

    return details


def run_detection_on_image(img):
    results = model(img, verbose=False)
    result = results[0]
    translated_names = get_translated_names()
    result.names = translated_names

    counts, detection_summary = build_detection_summary(result, translated_names)
    fruit_details = get_fruit_details(detection_summary)
    result_img = result.plot()

    return result_img, counts, detection_summary, fruit_details

@app.route("/", methods=["GET","POST"])
def index():

    counts = None
    image_path = None
    detection_summary = []
    fruit_details = []

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":

            img = cv2.imdecode(
                np.frombuffer(file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            result_img, counts, detection_summary, fruit_details = run_detection_on_image(img)

            os.makedirs("static", exist_ok=True)

            image_path = "static/result.jpg"
            cv2.imwrite(image_path, result_img)

    return render_template(
        "index.html",
        counts=counts,
        detection_summary=detection_summary,
        image=image_path,
        fruit_details=fruit_details,
    )


@app.route("/webcam", methods=["GET"])
def webcam_page():
    return render_template("webcam.html")


@app.route("/webcam-detect", methods=["POST"])
def webcam_detect():
    file = request.files.get("frame")

    if file is None:
        return jsonify({"error": "Khong nhan duoc anh tu webcam."}), 400

    frame_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Khong giai ma duoc frame webcam."}), 400

    result_img, counts, detection_summary, fruit_details = run_detection_on_image(frame)
    ok, encoded = cv2.imencode(".jpg", result_img)

    if not ok:
        return jsonify({"error": "Khong ma hoa duoc ket qua webcam."}), 500

    encoded_image = base64.b64encode(encoded.tobytes()).decode("ascii")

    return jsonify(
        {
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "counts": counts,
            "detection_summary": detection_summary,
            "fruit_details": fruit_details,
        }
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
