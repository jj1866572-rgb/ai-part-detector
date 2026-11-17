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

app = FastAPI(title="Pro AI Face & Clothes Segmenter (MiT-B5 + MiT-B2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

# Pro-Grade Transformer Models
FACE_MODEL_URL = "https://huggingface.co/Xenova/face-parsing/resolve/main/onnx/model_quantized.onnx"
CLOTHES_MODEL_URL = "https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx"

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

            face_path = ensure_model("face_mit_b5_q8.onnx", FACE_MODEL_URL)
            clothes_path = ensure_model("clothes_mit_b2_fp32.onnx", CLOTHES_MODEL_URL)

            _face_sess = ort.InferenceSession(face_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _clothes_sess = ort.InferenceSession(clothes_path, sess_options=opts, providers=["CPUExecutionProvider"])

    return _face_sess, _clothes_sess


def preprocess(pil_img: Image.Image, size: int = 512) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def logits_to_pred_and_probs(logits_chw: np.ndarray, out_w: int, out_h: int):
    c = logits_chw.shape[0]
    up = np.empty((c, out_h, out_w), dtype=np.float32)
    for i in range(c):
        up[i] = np.asarray(
            Image.fromarray(logits_chw[i], mode="F").resize((out_w, out_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    pred = np.argmax(up, axis=0).astype(np.uint8)
    max_l = np.max(up, axis=0, keepdims=True)
    ex = np.exp(np.clip(up - max_l, -20.0, 0.0))
    probs = ex / np.sum(ex, axis=0, keepdims=True)
    return pred, probs


def soft_mask(pred: np.ndarray, probs: np.ndarray, class_ids: list[int]) -> np.ndarray:
    prob_sum = np.zeros(pred.shape, dtype=np.float32)
    is_cls = np.zeros(pred.shape, dtype=bool)
    for cid in class_ids:
        prob_sum += probs[cid]
        is_cls |= (pred == cid)
    alpha = np.where(
        is_cls,
        np.clip(prob_sum * 255.0 * 1.15, 195.0, 255.0),
        np.where(prob_sum > 0.25, prob_sum * 255.0, 0.0),
    )
    return np.clip(alpha, 0, 255).astype(np.uint8)


def run_face_smart(face_sess: ort.InferenceSession, orig_img: Image.Image, clothes_pred: np.ndarray):
    orig_w, orig_h = orig_img.size
    inp_name = face_sess.get_inputs()[0].name

    # SegFormer-B2 থেকে মাথা/মুখের এলাকা (Hat=1, Hair=2, Sunglasses=3, Face=11) খুঁজে নেওয়া
    head_mask = np.isin(clothes_pred, [1, 2, 3, 11])
    ys, xs = np.where(head_mask)

    use_crop = False
    x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    if len(xs) >= 30:
        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)

        # মুখের আশেপাশে পর্যাপ্ত ৫৫% জায়গা রেখে স্কয়ার ক্রপ যাতে কান/চুল না কাটে
        pad = int(max(bw, bh) * 0.55)
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        half = (max(bw, bh) + 2 * pad) // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(orig_w, cx + half)
        y2 = min(orig_h, cy + half)

        crop_area = (x2 - x1) * (y2 - y1)
        # ছবিতে মুখ যদি ৫৫% এর চেয়ে ছোট হয়, তবে জুম-ক্রপ করে ডিটেক্ট করবে
        if crop_area > 80 * 80 and crop_area < 0.55 * orig_w * orig_h:
            use_crop = True

    if use_crop:
        cw, ch = x2 - x1, y2 - y1
        crop_img = orig_img.crop((x1, y1, x2, y2))
        logits = face_sess.run(None, {inp_name: preprocess(crop_img, 512)})[0][0]
        c_pred, c_probs = logits_to_pred_and_probs(logits, cw, ch)

        full_pred = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_probs = np.zeros((logits.shape[0], orig_h, orig_w), dtype=np.float32)
        full_probs[0] = 1.0  # background outside crop

        full_pred[y1:y2, x1:x2] = c_pred
        full_probs[:, y1:y2, x1:x2] = c_probs
        return full_pred, full_probs
    else:
        logits = face_sess.run(None, {inp_name: preprocess(orig_img, 512)})[0][0]
        return logits_to_pred_and_probs(logits, orig_w, orig_h)


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
        "models": ["Xenova/face-parsing (MiT-B5)", "mattmdjaga/segformer_b2_clothes (MiT-B2)"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["Xenova/face-parsing (MiT-B5)", "mattmdjaga/segformer_b2_clothes (MiT-B2)"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")


@app.post("/api/segment")
def segment_parts(req: SegmentRequest):
    try:
        b64_data = req.image.split(",")[-1].strip()
        img_bytes = base64.b64decode(b64_data)
        orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {e}")

    orig_w, orig_h = orig_img.size
    face_sess, clothes_sess = get_sessions()

    # ১. SegFormer-B2 Clothes (জামা, জুতা, প্যান্ট, ব্যাগ, টুপি, চুল, শরীর)
    c_inp_name = clothes_sess.get_inputs()[0].name
    c_logits = clothes_sess.run(None, {c_inp_name: preprocess(orig_img, 512)})[0][0]
    c_pred, c_probs = logits_to_pred_and_probs(c_logits, orig_w, orig_h)

    # ২. SegFormer MiT-B5 Face Parsing (চোখ, ভ্রু, নাক, কান, ঠোঁট, ত্বক, চুল)
    # ক্লাস: 0:bg, 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow,
    #        8:l_ear, 9:r_ear, 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat, 15:ear_r, 16:neck_l, 17:neck, 18:cloth
    f_pred, f_probs = run_face_smart(face_sess, orig_img, c_pred)

    parts = []

    def add_part(label: str, mask_u8: np.ndarray, min_pixels: int = 18):
        if int(np.count_nonzero(mask_u8 >= 120)) >= min_pixels:
            parts.append({
                "label": label,
                "mask": mask_u8_to_base64_rgba(mask_u8),
            })

    # জামাকাপড় ও শরীর
    shirt_u8 = soft_mask(c_pred, c_probs, [4])
    pants_u8 = soft_mask(c_pred, c_probs, [6])
    dress_u8 = soft_mask(c_pred, c_probs, [7])
    skirt_u8 = soft_mask(c_pred, c_probs, [5])
    shoes_u8 = soft_mask(c_pred, c_probs, [9, 10])
    bag_u8 = soft_mask(c_pred, c_probs, [16])
    scarf_u8 = soft_mask(c_pred, c_probs, [17])
    belt_u8 = soft_mask(c_pred, c_probs, [8])
    hat_u8 = np.maximum(soft_mask(c_pred, c_probs, [1]), soft_mask(f_pred, f_probs, [14]))
    arms_u8 = soft_mask(c_pred, c_probs, [14, 15])
    legs_u8 = soft_mask(c_pred, c_probs, [12, 13])

    # মুখ, চুল ও সূক্ষ্ম অংশ (MiT-B5 এর হাই-একুরেসি মাস্ক)
    hair_u8 = soft_mask(f_pred, f_probs, [13])
    if np.count_nonzero(hair_u8 >= 120) < 35:
        hair_u8 = soft_mask(c_pred, c_probs, [2])

    skin_u8 = soft_mask(f_pred, f_probs, [1])
    if np.count_nonzero(skin_u8 >= 120) < 30:
        skin_u8 = soft_mask(c_pred, c_probs, [11])

    l_eye_u8 = soft_mask(f_pred, f_probs, [4])
    r_eye_u8 = soft_mask(f_pred, f_probs, [5])
    eyes_u8 = np.maximum(l_eye_u8, r_eye_u8)
    brows_u8 = soft_mask(f_pred, f_probs, [6, 7])
    nose_u8 = soft_mask(f_pred, f_probs, [2])
    l_ear_u8 = soft_mask(f_pred, f_probs, [8])
    r_ear_u8 = soft_mask(f_pred, f_probs, [9])
    ears_u8 = np.maximum(l_ear_u8, r_ear_u8)
    lips_u8 = soft_mask(f_pred, f_probs, [10, 11, 12])
    glasses_u8 = np.maximum(soft_mask(f_pred, f_probs, [3]), soft_mask(c_pred, c_probs, [3]))
    neck_u8 = soft_mask(f_pred, f_probs, [17])

    # Subject & Background (সব ফোরগ্রাউন্ড মিলিয়ে)
    c_fg = np.clip((1.0 - c_probs[0]) * 255.0, 0, 255).astype(np.uint8)
    f_fg = np.clip((1.0 - f_probs[0]) * 255.0, 0, 255).astype(np.uint8)
    subject_u8 = np.maximum(c_fg, f_fg)
    bg_u8 = (255 - subject_u8).astype(np.uint8)

    add_part("Subject", subject_u8, min_pixels=50)
    add_part("Hair", hair_u8, min_pixels=35)
    add_part("Face Skin", skin_u8, min_pixels=30)
    add_part("Eyes", eyes_u8, min_pixels=8)
    add_part("Left Eye", l_eye_u8, min_pixels=5)
    add_part("Right Eye", r_eye_u8, min_pixels=5)
    add_part("Eyebrows", brows_u8, min_pixels=8)
    add_part("Nose", nose_u8, min_pixels=10)
    add_part("Ears", ears_u8, min_pixels=10)
    add_part("Left Ear", l_ear_u8, min_pixels=6)
    add_part("Right Ear", r_ear_u8, min_pixels=6)
    add_part("Lips", lips_u8, min_pixels=10)
    add_part("Glasses", glasses_u8, min_pixels=20)
    add_part("Shirt / Top", shirt_u8, min_pixels=50)
    add_part("Pants", pants_u8, min_pixels=50)
    add_part("Shoes", shoes_u8, min_pixels=30)
    add_part("Dress", dress_u8, min_pixels=50)
    add_part("Skirt", skirt_u8, min_pixels=45)
    add_part("Hat", hat_u8, min_pixels=30)
    add_part("Bag", bag_u8, min_pixels=35)
    add_part("Scarf", scarf_u8, min_pixels=30)
    add_part("Belt", belt_u8, min_pixels=20)
    add_part("Neck", neck_u8, min_pixels=25)
    add_part("Arms", arms_u8, min_pixels=45)
    add_part("Legs", legs_u8, min_pixels=45)
    add_part("Background", bg_u8, min_pixels=50)

    return {"success": True, "parts": parts}
