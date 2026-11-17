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

app = FastAPI(title="Pro Subject & Clothes/Shoes Segmenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

# 1. ORMBG (IS-Net 1024px, Apache-2.0) -> ফটোশপ-মানের নিখুঁত Subject / Background কাটআউট
# 2. SegFormer-B2 Clothes -> জামা, প্যান্ট, জুতা, ড্রেস, স্কার্ট, চুল, ব্যাগ, টুপি
SUBJECT_MODEL_URL = "https://huggingface.co/onnx-community/ormbg-ONNX/resolve/main/onnx/model_quantized.onnx"
CLOTHES_MODEL_URL = "https://huggingface.co/Xenova/segformer_b2_clothes/resolve/main/onnx/model_quantized.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_lock = threading.Lock()
_subject_sess = None
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
    global _subject_sess, _clothes_sess
    if _subject_sess is not None and _clothes_sess is not None:
        return _subject_sess, _clothes_sess

    with _model_lock:
        if _subject_sess is None or _clothes_sess is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2

            subj_path = ensure_model("ormbg_q8.onnx", SUBJECT_MODEL_URL)
            clothes_path = ensure_model("segformer_b2_clothes_q8.onnx", CLOTHES_MODEL_URL)

            _subject_sess = ort.InferenceSession(subj_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _clothes_sess = ort.InferenceSession(clothes_path, sess_options=opts, providers=["CPUExecutionProvider"])

    return _subject_sess, _clothes_sess


def preprocess_ormbg(pil_img: Image.Image, size: int = 1024) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def preprocess_segformer(pil_img: Image.Image, size: int = 512) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def run_subject_ormbg(subject_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = subject_sess.get_inputs()[0].name
    inp = preprocess_ormbg(orig_img, 1024)

    raw = subject_sess.run(None, {inp_name: inp})[0]
    pred = np.squeeze(raw).astype(np.float32)
    if pred.ndim == 3:
        pred = pred[0]

    p_min, p_max = float(pred.min()), float(pred.max())
    if p_min < -0.05 or p_max > 1.05:
        pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -20.0, 20.0)))
    else:
        pred = np.clip(pred, 0.0, 1.0)

    # মূল ছবির সাইজে রিসাইজ
    pred_img = Image.fromarray(pred, mode="F").resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    alpha = np.asarray(pred_img, dtype=np.float32)

    # ব্যাকগ্রাউন্ডের হালকা ধূসর দাগ (haze) পুরোপুরি পরিষ্কার করে ০ এবং সাবজেক্টকে ২৫৫ করা
    alpha = np.clip((alpha - 0.06) / (0.94 - 0.06), 0.0, 1.0)
    return (alpha * 255.0).astype(np.uint8)


def logits_to_upsampled_probs(logits: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    c = logits.shape[0]
    up = np.empty((c, target_h, target_w), dtype=np.float32)
    for i in range(c):
        ch_img = Image.fromarray(logits[i], mode="F").resize((target_w, target_h), Image.Resampling.BILINEAR)
        up[i] = np.asarray(ch_img, dtype=np.float32)

    shifted = up - np.max(up, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=0, keepdims=True)


def run_clothes_b2(clothes_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = clothes_sess.get_inputs()[0].name
    inp = preprocess_segformer(orig_img, 512)
    logits = clothes_sess.run(None, {inp_name: inp})[0][0]  # (18, 128, 128)
    return logits_to_upsampled_probs(logits, orig_w, orig_h)


def probs_to_u8(probs: np.ndarray, class_ids: list[int], subject_gate: np.ndarray) -> np.ndarray:
    p_sum = np.sum(probs[class_ids], axis=0)
    part_u8 = np.clip(p_sum * 255.0, 0, 255).astype(np.uint8)
    # শুধু Subject-এর ভেতরেই অংশগুলো সীমাবদ্ধ রাখা, যাতে কর্নারে কখনো দাগ না আসে
    return np.minimum(part_u8, subject_gate)


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
        "models": ["ORMBG (Subject Cutout)", "SegFormer-B2 (Clothes & Shoes)"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["ORMBG (Subject Cutout)", "SegFormer-B2 (Clothes & Shoes)"],
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

    subject_sess, clothes_sess = get_sessions()

    # 1. ফটোশপ-মানের পরিষ্কার Subject মাস্ক
    subject_u8 = run_subject_ormbg(subject_sess, orig_img)
    bg_u8 = (255 - subject_u8).astype(np.uint8)

    # 2. জামাকাপড়, প্যান্ট, জুতা, ব্যাগ, চুল, মুখ
    clothes_probs = run_clothes_b2(clothes_sess, orig_img)
    subject_gate = np.clip(subject_u8.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

    parts = []

    def add_part(label: str, mask_u8: np.ndarray, min_pixels: int = 30):
        if int(np.count_nonzero(mask_u8 >= 120)) >= min_pixels:
            parts.append({
                "label": label,
                "mask": mask_u8_to_base64_rgba(mask_u8),
            })

    add_part("Subject", subject_u8, min_pixels=50)

    # SegFormer-B2 Clothes IDs:
    # 1:Hat, 2:Hair, 3:Sunglasses, 4:Upper-clothes, 5:Skirt, 6:Pants, 7:Dress,
    # 8:Belt, 9:Left-shoe, 10:Right-shoe, 11:Face, 12:Left-leg, 13:Right-leg,
    # 14:Left-arm, 15:Right-arm, 16:Bag, 17:Scarf
    add_part("Shirt / Top", probs_to_u8(clothes_probs, [4], subject_gate), min_pixels=60)
    add_part("Pants", probs_to_u8(clothes_probs, [6], subject_gate), min_pixels=60)
    add_part("Dress", probs_to_u8(clothes_probs, [7], subject_gate), min_pixels=60)
    add_part("Skirt", probs_to_u8(clothes_probs, [5], subject_gate), min_pixels=50)

    add_part("Shoes", probs_to_u8(clothes_probs, [9, 10], subject_gate), min_pixels=30)
    add_part("Left Shoe", probs_to_u8(clothes_probs, [9], subject_gate), min_pixels=20)
    add_part("Right Shoe", probs_to_u8(clothes_probs, [10], subject_gate), min_pixels=20)

    add_part("Hair", probs_to_u8(clothes_probs, [2], subject_gate), min_pixels=35)
    add_part("Face", probs_to_u8(clothes_probs, [11], subject_gate), min_pixels=35)
    add_part("Hat", probs_to_u8(clothes_probs, [1], subject_gate), min_pixels=35)
    add_part("Glasses", probs_to_u8(clothes_probs, [3], subject_gate), min_pixels=20)
    add_part("Bag", probs_to_u8(clothes_probs, [16], subject_gate), min_pixels=40)
    add_part("Scarf", probs_to_u8(clothes_probs, [17], subject_gate), min_pixels=35)
    add_part("Belt", probs_to_u8(clothes_probs, [8], subject_gate), min_pixels=20)
    add_part("Arms", probs_to_u8(clothes_probs, [14, 15], subject_gate), min_pixels=45)
    add_part("Legs", probs_to_u8(clothes_probs, [12, 13], subject_gate), min_pixels=45)

    add_part("Background", bg_u8, min_pixels=50)

    return {"success": True, "parts": parts}
