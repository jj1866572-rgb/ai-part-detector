import base64
import gc
import io
import os
import threading
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
from pydantic import BaseModel

app = FastAPI(title="Fast Pro Subject, Mask & FASHN B4 Segmenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

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
                for p in (subj_path, fashn_path):
                    if p.startswith("/tmp/") and os.path.exists(p):
                        os.remove(p)
                raise

    return _subject_sess, _fashn_sess


def preprocess_ormbg(pil_img: Image.Image, size: int) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def preprocess_fashn(pil_img: Image.Image) -> np.ndarray:
    resized = pil_img.resize((384, 576), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def run_subject_ormbg(subject_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_meta = subject_sess.get_inputs()[0]
    inp_shape = inp_meta.shape
    subj_size = 640 if (len(inp_shape) >= 4 and not isinstance(inp_shape[2], int)) else int(inp_shape[2])

    inp = preprocess_ormbg(orig_img, subj_size)
    raw = subject_sess.run(None, {inp_meta.name: inp})[0]
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
    alpha = np.clip((alpha - 0.14) / (0.86 - 0.14), 0.0, 1.0)
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
    logits = fashn_sess.run(None, {inp_name: inp})[0][0]
    return logits_to_upsampled_probs(logits, orig_w, orig_h)


def remove_small_islands(mask_u8: np.ndarray, min_ratio: float = 0.05, min_abs: int = 18) -> np.ndarray:
    """
    Hair, Face, Face Mask ইত্যাদি থেকে ছোট ছোট বিচ্ছিন্ন পয়েন্ট/বিন্দু সম্পূর্ণ মুছে ফেলে।
    """
    h, w = mask_u8.shape
    grid = 128
    small = np.asarray(Image.fromarray(mask_u8, mode="L").resize((grid, grid), Image.Resampling.BILINEAR), dtype=np.uint8)
    fg = (small >= 110).ravel().tolist()
    n = grid * grid
    comp_id = [0] * n
    comp_sizes = [0]
    cid = 0

    for start in range(n):
        if not fg[start] or comp_id[start] != 0:
            continue
        cid += 1
        q = deque([start])
        comp_id[start] = cid
        cnt = 0
        while q:
            idx = q.popleft()
            cnt += 1
            x = idx % grid
            if x > 0 and fg[idx - 1] and comp_id[idx - 1] == 0:
                comp_id[idx - 1] = cid
                q.append(idx - 1)
            if x < grid - 1 and fg[idx + 1] and comp_id[idx + 1] == 0:
                comp_id[idx + 1] = cid
                q.append(idx + 1)
            if idx >= grid and fg[idx - grid] and comp_id[idx - grid] == 0:
                comp_id[idx - grid] = cid
                q.append(idx - grid)
            if idx + grid < n and fg[idx + grid] and comp_id[idx + grid] == 0:
                comp_id[idx + grid] = cid
                q.append(idx + grid)
        comp_sizes.append(cnt)

    if cid == 0:
        return np.zeros((h, w), dtype=np.uint8)

    max_sz = max(comp_sizes)
    cutoff = max(min_abs, int(max_sz * min_ratio))
    keep = [1 if comp_sizes[comp_id[i]] >= cutoff else 0 for i in range(n)]
    keep_img = Image.fromarray((np.array(keep, dtype=np.uint8).reshape(grid, grid)) * 255, mode="L")
    keep_full = np.asarray(keep_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
    return np.where(keep_full >= 90, mask_u8, 0).astype(np.uint8)


def clean_part_mask(mask_u8: np.ndarray, island_ratio: float = 0.05) -> np.ndarray:
    img = Image.fromarray(mask_u8, mode="L")
    img = img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip((arr - 0.25) / (0.75 - 0.25), 0.0, 1.0)
    cleaned = (arr * 255.0).astype(np.uint8)
    return remove_small_islands(cleaned, min_ratio=island_ratio, min_abs=14)


def detect_face_mask(subject_u8: np.ndarray, winning_cls: np.ndarray) -> np.ndarray | None:
    """
    মুখে মাস্ক পরা থাকলে Face (eyes/forehead) এবং Neck/Top-এর মাঝখানে ORMBG Subject-এর
    ভেতরের অংশ থেকে নিখুঁত Face Mask আলাদা করে।
    """
    h, w = subject_u8.shape
    face_ys, face_xs = np.where((winning_cls == 1) & (subject_u8 >= 90))
    if len(face_xs) < 40:
        return None

    fx1, fx2 = int(face_xs.min()), int(face_xs.max()) + 1
    fy1, fy2 = int(face_ys.min()), int(face_ys.max()) + 1
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    face_area = len(face_xs)

    roi_x1 = max(0, fx1 - int(fw * 0.12))
    roi_x2 = min(w, fx2 + int(fw * 0.12))
    roi_y1 = max(0, fy1 + int(fh * 0.18))
    roi_y2 = min(h, fy2 + int(fh * 1.15))

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = True

    # মুখের মাস্ক: Subject-এর ভেতরে আছে, কিন্তু Face(1), Top(3), Arms(12,13), Torso/Neck(16) নয়
    non_mask_cls = np.isin(winning_cls, [1, 3, 12, 13, 16])
    cand = roi_mask & (subject_u8 >= 120) & (~non_mask_cls)
    cand_u8 = (cand.astype(np.uint8)) * 255

    # শুধু মূল বড় মাস্কটি রাখা (১টি বৃহত্তম আইল্যান্ড)
    cleaned = remove_small_islands(cand_u8, min_ratio=0.45, min_abs=20)
    cleaned = clean_part_mask(cleaned, island_ratio=0.45)

    my, mx = np.where(cleaned >= 120)
    if len(mx) < max(250, int(face_area * 0.25)):
        return None
    if (int(mx.max()) - int(mx.min())) < fw * 0.35:
        return None
    return cleaned


def mask_u8_to_base64_rgba(mask_u8: np.ndarray) -> str:
    h, w = mask_u8.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = mask_u8

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": ["ORMBG (Fast Crisp Subject)", "FASHN SegFormer-B4 (Clothes, Face & Face Mask)"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["ORMBG (Fast Crisp Subject)", "FASHN SegFormer-B4 (Clothes, Face & Face Mask)"],
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
        subject_u8 = run_subject_ormbg(subject_sess, orig_img)
        bg_u8 = (255 - subject_u8).astype(np.uint8)
        f_probs = run_fashn_b4(fashn_sess, orig_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    subject_valid = subject_u8 >= 90
    subj_area = max(1, int(np.count_nonzero(subject_u8 >= 120)))
    winning_cls = np.argmax(f_probs, axis=0)
    subject_gate = np.clip(subject_u8.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

    top_cnt = int(np.count_nonzero((winning_cls == 3) & subject_valid))
    dress_cnt = int(np.count_nonzero((winning_cls == 4) & subject_valid))

    face_mask_u8 = detect_face_mask(subject_u8, winning_cls)

    parts = []

    def add_smart_part(label: str, class_ids: list[int], min_ratio: float, min_abs: int, min_conf: float = 0.45, island_ratio: float = 0.05):
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
        if face_mask_u8 is not None and label in ("Hair", "Face"):
            part_u8 = np.minimum(part_u8, (255 - face_mask_u8).astype(np.uint8))
        part_u8 = clean_part_mask(part_u8, island_ratio=island_ratio)
        if int(np.count_nonzero(part_u8 >= 120)) < min_abs // 2:
            return
        parts.append({
            "label": label,
            "mask": mask_u8_to_base64_rgba(part_u8),
        })

    if subj_area >= 50:
        parts.append({"label": "Subject", "mask": mask_u8_to_base64_rgba(subject_u8)})

    add_smart_part("Shirt / Top", [3], min_ratio=0.025, min_abs=350)

    if not (top_cnt > dress_cnt * 2.2 and dress_cnt < subj_area * 0.08):
        add_smart_part("Dress", [4], min_ratio=0.04, min_abs=500)

    add_smart_part("Pants", [6], min_ratio=0.025, min_abs=350)
    add_smart_part("Skirt", [5], min_ratio=0.025, min_abs=350)
    add_smart_part("Shoes", [15], min_ratio=0.008, min_abs=140, island_ratio=0.12)
    add_smart_part("Hair", [2], min_ratio=0.012, min_abs=180, island_ratio=0.08)
    add_smart_part("Face", [1], min_ratio=0.012, min_abs=180, island_ratio=0.10)

    if face_mask_u8 is not None:
        parts.append({"label": "Face Mask", "mask": mask_u8_to_base64_rgba(face_mask_u8)})

    add_smart_part("Arms", [12, 13], min_ratio=0.012, min_abs=200, island_ratio=0.08)
    add_smart_part("Hands", [13], min_ratio=0.006, min_abs=120, island_ratio=0.12)
    add_smart_part("Legs", [14], min_ratio=0.015, min_abs=220, island_ratio=0.10)
    add_smart_part("Neck / Body", [16], min_ratio=0.008, min_abs=140, island_ratio=0.12)
    add_smart_part("Hat", [9], min_ratio=0.010, min_abs=160, island_ratio=0.15)
    add_smart_part("Glasses", [11], min_ratio=0.004, min_abs=80, island_ratio=0.15)
    add_smart_part("Bag", [8], min_ratio=0.010, min_abs=160, island_ratio=0.12)
    add_smart_part("Scarf", [10], min_ratio=0.008, min_abs=140, island_ratio=0.12)
    add_smart_part("Belt", [7], min_ratio=0.004, min_abs=80, island_ratio=0.15)
    add_smart_part("Jewelry", [17], min_ratio=0.003, min_abs=60, island_ratio=0.15)

    if int(np.count_nonzero(bg_u8 >= 120)) >= 50:
        parts.append({"label": "Background", "mask": mask_u8_to_base64_rgba(bg_u8)})

    gc.collect()
    return {"success": True, "parts": parts}
