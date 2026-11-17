import base64
import io
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel

app = FastAPI(title="Pro Precision Subject & Clothes Segmenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

# 1. ORMBG (IS-Net 1024px) -> নিখুঁত Subject/Background কাটআউট
# 2. Full FP32 SegFormer-B2 Clothes -> quantized-এর চেয়ে অনেক বেশি নিখুঁত জামা, প্যান্ট, জুতা, হাত
SUBJECT_MODEL_URL = "https://huggingface.co/onnx-community/ormbg-ONNX/resolve/main/onnx/model_quantized.onnx"
CLOTHES_MODEL_URL = "https://huggingface.co/Xenova/segformer_b2_clothes/resolve/main/onnx/model.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FLIP_SWAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 13, 12, 15, 14, 16, 17]

_model_lock = threading.Lock()
_subject_sess = None
_clothes_sess = None


class SegmentRequest(BaseModel):
    image: str


def download_file_with_retry(url: str, dest_path: Path, retries: int = 3):
    last_err = None
    for attempt in range(retries):
        tmp_part = dest_path.with_suffix(f".part{attempt}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_part, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp_part, dest_path)
            return
        except Exception as e:
            last_err = e
            if tmp_part.exists():
                try:
                    tmp_part.unlink()
                except Exception:
                    pass
            time.sleep(1.0)
    raise RuntimeError(f"Download failed after {retries} attempts: {last_err}")


def ensure_model(filename: str, url: str) -> str:
    local_path = LOCAL_MODELS_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return str(local_path)

    TMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_MODELS_DIR / filename
    if not tmp_path.exists() or tmp_path.stat().st_size < 1_000_000:
        download_file_with_retry(url, tmp_path)
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
            clothes_path = ensure_model("segformer_b2_clothes_fp32.onnx", CLOTHES_MODEL_URL)

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

    pred_img = Image.fromarray(pred, mode="F").resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    alpha = np.asarray(pred_img, dtype=np.float32)
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


def run_clothes_b2_tta(clothes_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    """
    Horizontal-Flip Test-Time Augmentation (TTA):
    আসল ছবি + ফ্লিপ করা ছবির রেজাল্ট গড় করে নেওয়ায় ভাঁজ করা হাত ও পোশাক অনেক নিখুঁত হয়।
    """
    orig_w, orig_h = orig_img.size
    inp_name = clothes_sess.get_inputs()[0].name

    logits_normal = clothes_sess.run(None, {inp_name: preprocess_segformer(orig_img, 512)})[0][0]
    flipped_img = orig_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    logits_flip = clothes_sess.run(None, {inp_name: preprocess_segformer(flipped_img, 512)})[0][0]
    logits_unflip = logits_flip[FLIP_SWAP, :, ::-1]

    avg_logits = 0.5 * (logits_normal + logits_unflip)
    return logits_to_upsampled_probs(avg_logits, orig_w, orig_h)


def build_clean_part_mask(
    probs: np.ndarray,
    pred_labels: np.ndarray,
    class_ids: list[int],
    subject_gate: np.ndarray,
    close_holes: bool = False,
) -> np.ndarray:
    p_sum = np.sum(probs[class_ids], axis=0)
    is_winner = np.isin(pred_labels, class_ids)

    # কম্পিটিটিভ প্রবাবিলিটি: শুধু যেসব পিক্সেলে আসলেই এই পার্টটি জয়ী বা শক্তিশালী (0.35+) সেগুলোই নেওয়া
    soft = np.clip((p_sum - 0.32) / 0.36, 0.0, 1.0)
    soft = np.where(is_winner, np.maximum(soft, 0.78), soft * 0.55)
    mask_u8 = (soft * 255.0).astype(np.uint8)

    if close_holes:
        pil_m = Image.fromarray(mask_u8, mode="L")
        pil_m = pil_m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
        mask_u8 = np.asarray(pil_m, dtype=np.uint8)

    return np.minimum(mask_u8, subject_gate)


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
        "models": ["ORMBG-1024 (Subject)", "SegFormer-B2-FP32-TTA (Clothes)"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["ORMBG-1024 (Subject)", "SegFormer-B2-FP32-TTA (Clothes)"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model warmup failed: {e}")


@app.post("/api/segment")
def segment_parts(req: SegmentRequest):
    try:
        b64_data = req.image.split(",")[-1].strip()
        img_bytes = base64.b64decode(b64_data)
        orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {e}")

    try:
        subject_sess, clothes_sess = get_sessions()

        # 1. Subject & Background
        subject_u8 = run_subject_ormbg(subject_sess, orig_img)
        bg_u8 = (255 - subject_u8).astype(np.uint8)

        # 2. Full FP32 Clothes Parsing with TTA
        clothes_probs = run_clothes_b2_tta(clothes_sess, orig_img)
        subject_gate = np.clip(subject_u8.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

        fg_mask = subject_u8 >= 100
        subj_pixels = max(1, int(np.count_nonzero(fg_mask)))
        pred_labels = np.argmax(clothes_probs, axis=0)

        counts = {c: int(np.count_nonzero((pred_labels == c) & fg_mask)) for c in range(18)}

        # স্মার্ট ডিসঅ্যাম্বিগুয়েশন: Shirt বনাম Dress এবং Pants বনাম Skirt
        upper_cnt = counts[4]
        dress_cnt = counts[7]
        has_upper = False
        has_dress = False

        if upper_cnt >= dress_cnt * 1.35 or dress_cnt < max(450, int(0.05 * subj_pixels)):
            # এটি টি-শার্ট/টপ, ড্রেস নয় -> ভুয়া ড্রেসের সম্ভাবনা টপে মিশিয়ে দেওয়া
            if upper_cnt >= max(300, int(0.025 * subj_pixels)):
                clothes_probs[4] = np.clip(clothes_probs[4] + clothes_probs[7], 0.0, 1.0)
                clothes_probs[7] = 0.0
                has_upper = True
        elif dress_cnt >= max(450, int(0.05 * subj_pixels)):
            clothes_probs[7] = np.clip(clothes_probs[7] + clothes_probs[4], 0.0, 1.0)
            clothes_probs[4] = 0.0
            has_dress = True

        pants_cnt = counts[6]
        skirt_cnt = counts[5]
        has_pants = pants_cnt >= max(300, int(0.025 * subj_pixels))
        has_skirt = (skirt_cnt >= max(350, int(0.04 * subj_pixels))) and not (pants_cnt >= skirt_cnt * 2.2)

        if has_pants and not has_skirt:
            clothes_probs[6] = np.clip(clothes_probs[6] + clothes_probs[5], 0.0, 1.0)
            clothes_probs[5] = 0.0

        # নতুন করে চূড়ান্ত label map বের করা
        pred_labels = np.argmax(clothes_probs, axis=0)

        parts = []

        def add_part(label: str, class_ids: list[int], min_abs: int, min_ratio: float, close_holes: bool = False):
            win_px = sum(int(np.count_nonzero((pred_labels == cid) & fg_mask)) for cid in class_ids)
            if win_px < max(min_abs, int(min_ratio * subj_pixels)):
                return
            mask_u8 = build_clean_part_mask(clothes_probs, pred_labels, class_ids, subject_gate, close_holes=close_holes)
            if int(np.count_nonzero(mask_u8 >= 120)) >= min_abs:
                parts.append({"label": label, "mask": mask_u8_to_base64_rgba(mask_u8)})

        # Subject
        if int(np.count_nonzero(subject_u8 >= 120)) >= 100:
            parts.append({"label": "Subject", "mask": mask_u8_to_base64_rgba(subject_u8)})

        if has_upper:
            add_part("Shirt / Top", [4], min_abs=300, min_ratio=0.025, close_holes=True)
        if has_dress:
            add_part("Dress", [7], min_abs=450, min_ratio=0.05, close_holes=True)
        if has_pants:
            add_part("Pants", [6], min_abs=300, min_ratio=0.025, close_holes=True)
        if has_skirt:
            add_part("Skirt", [5], min_abs=350, min_ratio=0.04, close_holes=True)

        # জুতা (থাকলে তবেই দেখাবে)
        add_part("Shoes", [9, 10], min_abs=150, min_ratio=0.008, close_holes=True)
        add_part("Left Shoe", [9], min_abs=80, min_ratio=0.004, close_holes=True)
        add_part("Right Shoe", [10], min_abs=80, min_ratio=0.004, close_holes=True)

        # চুল, মুখ, হাত, পা ও অ্যাক্সেসরিজ (সত্যিই ছবিতে থাকলে তবেই দেখাবে)
        add_part("Hair", [2], min_abs=180, min_ratio=0.012)
        add_part("Face", [11], min_abs=180, min_ratio=0.012)
        add_part("Arms", [14, 15], min_abs=180, min_ratio=0.012, close_holes=True)
        add_part("Legs", [12, 13], min_abs=200, min_ratio=0.015, close_holes=True)
        add_part("Hat", [1], min_abs=160, min_ratio=0.01)
        add_part("Glasses", [3], min_abs=90, min_ratio=0.005)
        add_part("Bag", [16], min_abs=200, min_ratio=0.015)
        add_part("Scarf", [17], min_abs=160, min_ratio=0.01)
        add_part("Belt", [8], min_abs=100, min_ratio=0.006)

        if int(np.count_nonzero(bg_u8 >= 120)) >= 100:
            parts.append({"label": "Background", "mask": mask_u8_to_base64_rgba(bg_u8)})

        return {"success": True, "parts": parts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
