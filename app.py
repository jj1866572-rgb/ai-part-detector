import base64
import gc
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

app = FastAPI(title="Pro Subject & FASHN B4 Clothes Segmenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

# 1. ORMBG (IS-Net 1024px) -> নিখুঁত Subject / Background কাটআউট
# 2. FASHN Human Parser (SegFormer-B4) -> নিখুঁত Shirt/Top, Arms+Hands, Pants, Shoes, Hair, Face
SUBJECT_MODEL_URL = "https://huggingface.co/onnx-community/ormbg-ONNX/resolve/main/onnx/model_quantized.onnx"
FASHN_MODEL_URL = "https://huggingface.co/faisal-shohag/fashn-human-parser-onnx/resolve/main/onnx/model_quantized.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_lock = threading.Lock()
_subject_sess = None
_fashn_sess = None


class SegmentRequest(BaseModel):
    image: str


def download_file(url: str, dest_path: Path):
    tmp_part = dest_path.with_suffix(".part")
    for attempt in range(2):
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
        except Exception:
            if tmp_part.exists():
                try:
                    tmp_part.unlink()
                except Exception:
                    pass
            if attempt == 1:
                raise


def ensure_model(filename: str, url: str) -> str:
    local_path = LOCAL_MODELS_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 5_000_000:
        return str(local_path)

    TMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_MODELS_DIR / filename
    if not tmp_path.exists() or tmp_path.stat().st_size < 5_000_000:
        download_file(url, tmp_path)
    return str(tmp_path)


def get_sessions():
    global _subject_sess, _fashn_sess
    if _subject_sess is not None and _fashn_sess is not None:
        return _subject_sess, _fashn_sess

    with _model_lock:
        if _subject_sess is None or _fashn_sess is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2

            subj_path = ensure_model("ormbg_q8.onnx", SUBJECT_MODEL_URL)
            fashn_path = ensure_model("fashn_b4_q8.onnx", FASHN_MODEL_URL)

            try:
                _subject_sess = ort.InferenceSession(subj_path, sess_options=opts, providers=["CPUExecutionProvider"])
                _fashn_sess = ort.InferenceSession(fashn_path, sess_options=opts, providers=["CPUExecutionProvider"])
            except Exception:
                # করাপ্টেড ক্যাশ থাকলে পরিষ্কার করে পুনরায় ডাউনলোড
                for p in (subj_path, fashn_path):
                    if p.startswith("/tmp/") and os.path.exists(p):
                        os.remove(p)
                raise

    return _subject_sess, _fashn_sess


def preprocess_ormbg(pil_img: Image.Image, size: int = 1024) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def preprocess_fashn(pil_img: Image.Image) -> np.ndarray:
    # FASHN Human Parser (SegFormer-B4) ইনপুট সাইজ: 384 (W) x 576 (H)
    resized = pil_img.resize((384, 576), Image.Resampling.BILINEAR)
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


def run_fashn_b4(fashn_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = fashn_sess.get_inputs()[0].name
    inp = preprocess_fashn(orig_img)
    logits = fashn_sess.run(None, {inp_name: inp})[0][0]  # (18, 144, 96)
    return logits_to_upsampled_probs(logits, orig_w, orig_h)


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
        "models": ["ORMBG (Subject)", "FASHN SegFormer-B4 (Clothes & Parts)"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["ORMBG (Subject)", "FASHN SegFormer-B4 (Clothes & Parts)"],
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

    try:
        subject_sess, fashn_sess = get_sessions()

        # 1. Subject & Background
        subject_u8 = run_subject_ormbg(subject_sess, orig_img)
        bg_u8 = (255 - subject_u8).astype(np.uint8)

        # 2. FASHN SegFormer-B4 (18 classes)
        # 0:bg, 1:face, 2:hair, 3:top, 4:dress, 5:skirt, 6:pants, 7:belt, 8:bag,
        # 9:hat, 10:scarf, 11:glasses, 12:arms, 13:hands, 14:legs, 15:feet, 16:torso, 17:jewelry
        f_probs = run_fashn_b4(fashn_sess, orig_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    subject_valid = subject_u8 >= 90
    subj_area = max(1, int(np.count_nonzero(subject_u8 >= 120)))
    winning_cls = np.argmax(f_probs, axis=0)
    subject_gate = np.clip(subject_u8.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

    top_cnt = int(np.count_nonzero((winning_cls == 3) & subject_valid))
    dress_cnt = int(np.count_nonzero((winning_cls == 4) & subject_valid))

    parts = []

    def add_smart_part(label: str, class_ids: list[int], min_ratio: float, min_abs: int, min_conf: float = 0.45):
        win_mask = np.isin(winning_cls, class_ids) & subject_valid
        win_cnt = int(np.count_nonzero(win_mask))
        req_pixels = max(min_abs, int(subj_area * min_ratio))
        if win_cnt < req_pixels:
            return

        p_sum = np.sum(f_probs[class_ids], axis=0)
        if float(np.mean(p_sum[win_mask])) < min_conf:
            return

        part_u8 = np.clip(p_sum * 255.0, 0, 255).astype(np.uint8)
        part_u8 = np.minimum(part_u8, subject_gate)
        parts.append({
            "label": label,
            "mask": mask_u8_to_base64_rgba(part_u8),
        })

    if subj_area >= 50:
        parts.append({"label": "Subject", "mask": mask_u8_to_base64_rgba(subject_u8)})

    add_smart_part("Shirt / Top", [3], min_ratio=0.025, min_abs=350)

    # টি-শার্টের ওপর ছোটখাটো Dress নয়েজ এড়াতে স্মার্ট ফিল্টার
    if not (top_cnt > dress_cnt * 2.2 and dress_cnt < subj_area * 0.08):
        add_smart_part("Dress", [4], min_ratio=0.04, min_abs=500)

    add_smart_part("Pants", [6], min_ratio=0.025, min_abs=350)
    add_smart_part("Skirt", [5], min_ratio=0.025, min_abs=350)
    add_smart_part("Shoes", [15], min_ratio=0.008, min_abs=140)
    add_smart_part("Hair", [2], min_ratio=0.012, min_abs=180)
    add_smart_part("Face", [1], min_ratio=0.012, min_abs=180)
    add_smart_part("Arms", [12, 13], min_ratio=0.012, min_abs=200)
    add_smart_part("Hands", [13], min_ratio=0.006, min_abs=120)
    add_smart_part("Legs", [14], min_ratio=0.015, min_abs=220)
    add_smart_part("Neck / Body", [16], min_ratio=0.008, min_abs=140)
    add_smart_part("Hat", [9], min_ratio=0.010, min_abs=160)
    add_smart_part("Glasses", [11], min_ratio=0.004, min_abs=80)
    add_smart_part("Bag", [8], min_ratio=0.010, min_abs=160)
    add_smart_part("Scarf", [10], min_ratio=0.008, min_abs=140)
    add_smart_part("Belt", [7], min_ratio=0.004, min_abs=80)
    add_smart_part("Jewelry", [17], min_ratio=0.003, min_abs=60)

    if int(np.count_nonzero(bg_u8 >= 120)) >= 50:
        parts.append({"label": "Background", "mask": mask_u8_to_base64_rgba(bg_u8)})

    gc.collect()
    return {"success": True, "parts": parts}
