import base64
import io
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Pro AI Face, Clothes & Shoes Segmenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

# অনেক বেশি শক্তিশালী ২টি ফ্ল্যাগশিপ মডেল:
# 1. SegFormer-B5 Face Parsing (~89 MB) -> চোখ, নাক, কান, ভ্রু, ঠোঁট, চুল, মুখ
# 2. SegFormer-B2 Clothes Parsing (~29 MB) -> জামা, প্যান্ট, জুতা, ব্যাগ, ড্রেস, টুপি
FACE_MODEL_URL = "https://huggingface.co/Xenova/face-parsing/resolve/main/onnx/model_quantized.onnx"
CLOTHES_MODEL_URL = "https://huggingface.co/Xenova/segformer_b2_clothes/resolve/main/onnx/model_quantized.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_lock = threading.Lock()
_face_sess = None
_clothes_sess = None


class SegmentRequest(BaseModel):
    image: str


def download_file(url: str, dest_path: Path):
    tmp_part = dest_path.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_part, "wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp_part, dest_path)


def ensure_model(filename: str, url: str) -> str:
    local_path = LOCAL_MODELS_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return str(local_path)

    TMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_MODELS_DIR / filename
    if not tmp_path.exists() or tmp_path.stat().st_size < 1_000_000:
        download_file(url, tmp_path)
    return str(tmp_path)


def get_sessions():
    global _face_sess, _clothes_sess
    if _face_sess is not None and _clothes_sess is not None:
        return _face_sess, _clothes_sess

    with _model_lock:
        if _face_sess is None or _clothes_sess is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2

            face_path = ensure_model("segformer_b5_face_q8.onnx", FACE_MODEL_URL)
            clothes_path = ensure_model("segformer_b2_clothes_q8.onnx", CLOTHES_MODEL_URL)

            _face_sess = ort.InferenceSession(face_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _clothes_sess = ort.InferenceSession(clothes_path, sess_options=opts, providers=["CPUExecutionProvider"])

    return _face_sess, _clothes_sess


def preprocess(pil_img: Image.Image, size: int = 512) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def logits_to_upsampled_probs(logits: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    ১। আগে logits-কে মূল ছবির সাইজে Bilinear Upsample করে
    ২। তারপর Softmax বের করে — এতে চোখ, নাক, কান, জুতা ও জামার কিনারা একদম স্মুথ হয়।
    """
    c = logits.shape[0]
    up = np.empty((c, target_h, target_w), dtype=np.float32)
    for i in range(c):
        ch_img = Image.fromarray(logits[i], mode="F").resize((target_w, target_h), Image.Resampling.BILINEAR)
        up[i] = np.asarray(ch_img, dtype=np.float32)

    shifted = up - np.max(up, axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=0, keepdims=True)
    return probs


def run_clothes_b2(clothes_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = clothes_sess.get_inputs()[0].name
    inp = preprocess(orig_img, 512)
    logits = clothes_sess.run(None, {inp_name: inp})[0][0]  # (18, 128, 128)
    return logits_to_upsampled_probs(logits, orig_w, orig_h)  # (18, orig_h, orig_w)


def run_face_b5_smart(face_sess: ort.InferenceSession, orig_img: Image.Image, clothes_probs: np.ndarray) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = face_sess.get_inputs()[0].name

    # SegFormer-B2 থেকে মাথার অবস্থান খুঁজে বের করা (Hat=1, Hair=2, Sunglasses=3, Face=11)
    clothes_labels = np.argmax(clothes_probs, axis=0)
    head_mask = np.isin(clothes_labels, [1, 2, 3, 11])
    ys, xs = np.where(head_mask)

    use_crop = False
    x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    if len(xs) >= 30:
        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1

        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        pad_x = int(bw * 0.45)
        pad_y = int(bh * 0.45)

        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        half_side = max(bw + 2 * pad_x, bh + 2 * pad_y) // 2

        x1 = max(0, cx - half_side)
        y1 = max(0, cy - half_side)
        x2 = min(orig_w, cx + half_side)
        y2 = min(orig_h, cy + half_side)

        crop_area = (x2 - x1) * (y2 - y1)
        if crop_area > 64 * 64 and crop_area < 0.72 * orig_w * orig_h:
            use_crop = True

    if use_crop:
        crop_img = orig_img.crop((x1, y1, x2, y2))
        inp = preprocess(crop_img, 512)
        logits = face_sess.run(None, {inp_name: inp})[0][0]  # (19, 128, 128)
        crop_probs = logits_to_upsampled_probs(logits, x2 - x1, y2 - y1)

        full_probs = np.zeros((19, orig_h, orig_w), dtype=np.float32)
        full_probs[0, :, :] = 1.0
        full_probs[:, y1:y2, x1:x2] = crop_probs
        return full_probs
    else:
        inp = preprocess(orig_img, 512)
        logits = face_sess.run(None, {inp_name: inp})[0][0]  # (19, 128, 128)
        return logits_to_upsampled_probs(logits, orig_w, orig_h)


def probs_to_u8(probs: np.ndarray, class_ids: list[int]) -> np.ndarray:
    p_sum = np.sum(probs[class_ids], axis=0)
    return np.clip(p_sum * 255.0, 0, 255).astype(np.uint8)


def mask_u8_to_base64_rgba(mask_u8: np.ndarray) -> str:
    h, w = mask_u8.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = mask_u8

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": ["SegFormer-B5 (Face Parsing)", "SegFormer-B2 (Clothes & Shoes)"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["SegFormer-B5 (Face Parsing)", "SegFormer-B2 (Clothes & Shoes)"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model warmup failed: {e}")


@app.post("/api/segment")
def segment_parts(req: SegmentRequest):
    try:
        b64_data = req.image.split(",")[-1].strip()
        img_bytes = base64.b64decode(b64_data)
        orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {e}")

    face_sess, clothes_sess = get_sessions()

    # 1. SegFormer-B2 দিয়ে জামাকাপড়, প্যান্ট, জুতা, ব্যাগ, শরীর
    clothes_probs = run_clothes_b2(clothes_sess, orig_img)

    # 2. SegFormer-B5 দিয়ে চোখ, নাক, কান, ভ্রু, ঠোঁট, চুল, মুখ
    face_probs = run_face_b5_smart(face_sess, orig_img, clothes_probs)

    clothes_fg = np.clip((1.0 - clothes_probs[0]) * 255.0, 0, 255).astype(np.uint8)
    face_fg = np.clip((1.0 - face_probs[0]) * 255.0, 0, 255).astype(np.uint8)
    subject_u8 = np.maximum(clothes_fg, face_fg)
    bg_u8 = (255 - subject_u8).astype(np.uint8)

    parts = []

    def add_part(label: str, mask_u8: np.ndarray, min_pixels: int = 20):
        if int(np.count_nonzero(mask_u8 >= 120)) >= min_pixels:
            parts.append({
                "label": label,
                "mask": mask_u8_to_base64_rgba(mask_u8),
            })

    add_part("Subject", subject_u8, min_pixels=50)

    # মুখ ও চুল (SegFormer-B5 IDs: 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow,
    # 8:l_ear, 9:r_ear, 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat, 15:ear_r, 17:neck, 18:cloth)
    hair_u8 = np.maximum(probs_to_u8(face_probs, [13]), probs_to_u8(clothes_probs, [2]))
    add_part("Hair", hair_u8, min_pixels=35)

    skin_u8 = probs_to_u8(face_probs, [1])
    if np.count_nonzero(skin_u8 >= 120) < 30:
        skin_u8 = probs_to_u8(clothes_probs, [11])
    add_part("Face Skin", skin_u8, min_pixels=30)

    l_eye_u8 = probs_to_u8(face_probs, [4])
    r_eye_u8 = probs_to_u8(face_probs, [5])
    add_part("Eyes", probs_to_u8(face_probs, [4, 5]), min_pixels=8)
    add_part("Left Eye", l_eye_u8, min_pixels=6)
    add_part("Right Eye", r_eye_u8, min_pixels=6)

    add_part("Eyebrows", probs_to_u8(face_probs, [6, 7]), min_pixels=8)
    add_part("Nose", probs_to_u8(face_probs, [2]), min_pixels=10)

    l_ear_u8 = probs_to_u8(face_probs, [8])
    r_ear_u8 = probs_to_u8(face_probs, [9])
    add_part("Ears", probs_to_u8(face_probs, [8, 9]), min_pixels=10)
    add_part("Left Ear", l_ear_u8, min_pixels=8)
    add_part("Right Ear", r_ear_u8, min_pixels=8)

    add_part("Lips", probs_to_u8(face_probs, [10, 11, 12]), min_pixels=10)

    glasses_u8 = np.maximum(probs_to_u8(face_probs, [3]), probs_to_u8(clothes_probs, [3]))
    add_part("Glasses", glasses_u8, min_pixels=20)

    # জামাকাপড় ও জুতা (SegFormer-B2 IDs: 1:Hat, 4:Upper-clothes, 5:Skirt, 6:Pants, 7:Dress,
    # 8:Belt, 9:Left-shoe, 10:Right-shoe, 12:Left-leg, 13:Right-leg, 14:Left-arm, 15:Right-arm, 16:Bag, 17:Scarf)
    upper_u8 = probs_to_u8(clothes_probs, [4])
    if np.count_nonzero(upper_u8 >= 120) < 60:
        upper_u8 = probs_to_u8(face_probs, [18])
    add_part("Shirt / Top", upper_u8, min_pixels=60)

    add_part("Pants", probs_to_u8(clothes_probs, [6]), min_pixels=60)
    add_part("Dress", probs_to_u8(clothes_probs, [7]), min_pixels=60)
    add_part("Skirt", probs_to_u8(clothes_probs, [5]), min_pixels=50)

    l_shoe_u8 = probs_to_u8(clothes_probs, [9])
    r_shoe_u8 = probs_to_u8(clothes_probs, [10])
    add_part("Shoes", probs_to_u8(clothes_probs, [9, 10]), min_pixels=30)
    add_part("Left Shoe", l_shoe_u8, min_pixels=20)
    add_part("Right Shoe", r_shoe_u8, min_pixels=20)

    hat_u8 = np.maximum(probs_to_u8(clothes_probs, [1]), probs_to_u8(face_probs, [14]))
    add_part("Hat", hat_u8, min_pixels=35)
    add_part("Bag", probs_to_u8(clothes_probs, [16]), min_pixels=40)
    add_part("Scarf", probs_to_u8(clothes_probs, [17]), min_pixels=35)
    add_part("Belt", probs_to_u8(clothes_probs, [8]), min_pixels=20)
    add_part("Neck", probs_to_u8(face_probs, [17]), min_pixels=25)
    add_part("Arms", probs_to_u8(clothes_probs, [14, 15]), min_pixels=45)
    add_part("Legs", probs_to_u8(clothes_probs, [12, 13]), min_pixels=45)

    add_part("Background", bg_u8, min_pixels=50)

    return {"success": True, "parts": parts}
