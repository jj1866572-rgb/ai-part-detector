import base64
import gc
import hashlib
import io
import os
import threading
import urllib.request
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
from pydantic import BaseModel

app = FastAPI(title="Smart Pro Subject, Clothes, Objects & Mask Segmenter")

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

# সার্ভার চাপ কমাতে ইন-মেমোরি ইমেজ ক্যাশ (একই ছবি হলে ০ সেকেন্ডে উত্তর)
_cache_lock = threading.Lock()
_result_cache: OrderedDict[str, list] = OrderedDict()
_CACHE_MAX = 20


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


def preprocess_fashn_letterbox(pil_img: Image.Image, tw: int = 384, th: int = 576):
    ow, oh = pil_img.size
    scale = min(tw / float(ow), th / float(oh))
    nw = max(16, int(round(ow * scale)))
    nh = max(16, int(round(oh * scale)))
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2

    canvas = Image.new("RGB", (tw, th), (128, 128, 128))
    resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    tensor = np.ascontiguousarray(arr[None, ...], dtype=np.float32)
    return tensor, (pad_x, pad_y, nw, nh, tw, th)


def run_subject_ormbg(subject_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_meta = subject_sess.get_inputs()[0]
    inp_shape = inp_meta.shape
    subj_size = 512 if (len(inp_shape) >= 4 and not isinstance(inp_shape[2], int)) else int(inp_shape[2])

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


def logits_to_upsampled_probs(logits: np.ndarray, target_w: int, target_h: int, box_info=None) -> np.ndarray:
    c, lh, lw = logits.shape
    if box_info is not None:
        pad_x, pad_y, nw, nh, tw, th = box_info
        bx1 = pad_x * (lw / float(tw))
        by1 = pad_y * (lh / float(th))
        bx2 = (pad_x + nw) * (lw / float(tw))
        by2 = (pad_y + nh) * (lh / float(th))
        crop_box = (bx1, by1, bx2, by2)
    else:
        crop_box = None

    up = np.empty((c, target_h, target_w), dtype=np.float32)
    for i in range(c):
        ch_img = Image.fromarray(logits[i], mode="F").resize(
            (target_w, target_h), Image.Resampling.BILINEAR, box=crop_box
        )
        up[i] = np.asarray(ch_img, dtype=np.float32)

    shifted = up - np.max(up, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=0, keepdims=True)


def run_fashn_b4(fashn_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = fashn_sess.get_inputs()[0].name
    inp, box_info = preprocess_fashn_letterbox(orig_img, 384, 576)
    logits = fashn_sess.run(None, {inp_name: inp})[0][0]
    return logits_to_upsampled_probs(logits, orig_w, orig_h, box_info=box_info)


def find_connected_components_128(mask_u8: np.ndarray):
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

    return grid, comp_id, comp_sizes


def remove_small_islands(mask_u8: np.ndarray, min_ratio: float = 0.05, min_abs: int = 14) -> np.ndarray:
    h, w = mask_u8.shape
    grid, comp_id, comp_sizes = find_connected_components_128(mask_u8)
    if len(comp_sizes) <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    max_sz = max(comp_sizes)
    cutoff = max(min_abs, int(max_sz * min_ratio))
    keep = [1 if comp_sizes[c] >= cutoff else 0 for c in comp_id]
    keep_img = Image.fromarray((np.array(keep, dtype=np.uint8).reshape(grid, grid)) * 255, mode="L")
    keep_full = np.asarray(keep_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
    return np.where(keep_full >= 90, mask_u8, 0).astype(np.uint8)


def extract_top_component_masks(mask_u8: np.ndarray, min_pixels_full: int, max_objects: int = 2):
    h, w = mask_u8.shape
    grid, comp_id, comp_sizes = find_connected_components_128(mask_u8)
    if len(comp_sizes) <= 1:
        return []

    scale_area = (h * w) / float(grid * grid)
    ranked = sorted(
        [(cid, sz) for cid, sz in enumerate(comp_sizes) if cid > 0 and sz * scale_area >= min_pixels_full],
        key=lambda x: x[1],
        reverse=True,
    )
    results = []
    for cid, _ in ranked[:max_objects]:
        keep = [255 if c == cid else 0 for c in comp_id]
        keep_img = Image.fromarray(np.array(keep, dtype=np.uint8).reshape(grid, grid), mode="L")
        keep_full = np.asarray(keep_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
        obj_mask = np.where(keep_full >= 90, mask_u8, 0).astype(np.uint8)
        if int(np.count_nonzero(obj_mask >= 120)) >= min_pixels_full:
            results.append(obj_mask)
    return results


def clean_part_mask(mask_u8: np.ndarray, island_ratio: float = 0.05, erode: bool = True, close_size: int = 3) -> np.ndarray:
    img = Image.fromarray(mask_u8, mode="L")
    img = img.filter(ImageFilter.MaxFilter(close_size)).filter(ImageFilter.MinFilter(close_size))
    if erode:
        img = img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip((arr - 0.25) / (0.75 - 0.25), 0.0, 1.0)
    cleaned = (arr * 255.0).astype(np.uint8)
    return remove_small_islands(cleaned, min_ratio=island_ratio, min_abs=12)


def detect_face_mask(ormbg_u8: np.ndarray, winning_cls: np.ndarray, f_probs: np.ndarray) -> np.ndarray | None:
    """
    দাড়ি, হাসি বা ল্যাপটপে কখনো ভুয়া Face Mask দেবে না—
    শুধুমাত্র মুখের নাক-মুখ ঢেকে রাখা ও Top-এর ওপরে থাকা প্রকৃত মাস্কই চিনবে।
    """
    h, w = ormbg_u8.shape
    face_ys, face_xs = np.where((winning_cls == 1) & (ormbg_u8 >= 90))
    if len(face_xs) < 40:
        return None

    fx1, fx2 = int(face_xs.min()), int(face_xs.max()) + 1
    fy1, fy2 = int(face_ys.min()), int(face_ys.max()) + 1
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    face_area = len(face_xs)
    face_cx = (fx1 + fx2) * 0.5

    # যদি চেহারা আগেই পুরো লম্বা থাকে (দাড়ি-মুখ খোলা), তবে ওপর থেকে চিন-বরাবর মাঝের গ্যাপে মাস্ক আছে কিনা দেখা
    roi_x1 = max(0, fx1 - int(fw * 0.06))
    roi_x2 = min(w, fx2 + int(fw * 0.06))
    roi_y1 = max(0, fy1 + int(fh * 0.28))
    roi_y2 = min(h, fy2 + max(int(fh * 1.20), int(fw * 0.85)))

    # জামা/হুডি শুরু হলে তার নিচে আর মাস্ক খুঁজবে না (যাতে ল্যাপটপ/কলার মাস্ক না ভাবে)
    top_ys = np.where((winning_cls == 3) & (ormbg_u8 >= 90))[0]
    if len(top_ys) > 30:
        top_min_y = int(np.percentile(top_ys, 5))
        roi_y2 = min(roi_y2, top_min_y + int(fw * 0.12))

    if roi_y2 <= roi_y1 + 8:
        return None

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = True

    is_other_part = (
        (f_probs[1] >= 0.42)
        | (f_probs[2] >= 0.45)
        | (f_probs[3] >= 0.45)
        | (f_probs[12] >= 0.45)
        | (f_probs[13] >= 0.45)
        | (f_probs[16] >= 0.45)
    )
    cand = roi_mask & (ormbg_u8 >= 125) & (~is_other_part)
    cand_u8 = (cand.astype(np.uint8)) * 255
    cleaned = clean_part_mask(cand_u8, island_ratio=0.50, erode=False, close_size=5)

    my, mx = np.where(cleaned >= 120)
    if len(mx) < max(300, int(face_area * 0.30)):
        return None

    mx1, mx2 = int(mx.min()), int(mx.max()) + 1
    my1, my2 = int(my.min()), int(my.max()) + 1
    mw_box = max(1, mx2 - mx1)
    mh_box = max(1, my2 - my1)

    # মাস্ক অবশ্যই চেহারার মাঝখানে ও কমপ্যাক্ট হতে হবে
    if mw_box < fw * 0.42 or mw_box > fw * 1.15:
        return None
    if abs((mx1 + mx2) * 0.5 - face_cx) > fw * 0.22:
        return None
    if my1 > fy2 + int(fh * 0.12):
        return None
    if (len(mx) / float(mw_box * mh_box)) < 0.42:
        return None

    # আসল মাস্কের ভেতরে Face skin probability খুবই কম থাকবে (< 0.22), দাড়ি হলে বেশি থাকত
    if float(np.mean(f_probs[1][cleaned >= 120])) > 0.22:
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
        "models": ["ORMBG", "FASHN SegFormer-B4"],
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["ORMBG", "FASHN SegFormer-B4"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model warmup failed: {e}")


@app.post("/api/segment")
def segment_parts(req: SegmentRequest):
    try:
        b64_data = req.image.split(",")[-1].strip()
        img_bytes = base64.b64decode(b64_data)
        img_hash = hashlib.sha256(img_bytes).hexdigest()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {e}")

    with _cache_lock:
        if img_hash in _result_cache:
            _result_cache.move_to_end(img_hash)
            return {"success": True, "parts": _result_cache[img_hash], "cached": True}

    try:
        orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        subject_sess, fashn_sess = get_sessions()
        ormbg_u8 = run_subject_ormbg(subject_sess, orig_img)
        f_probs = run_fashn_b4(fashn_sess, orig_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    orig_w, orig_h = orig_img.size
    img_area = orig_w * orig_h
    ormbg_area = int(np.count_nonzero(ormbg_u8 >= 120))

    # FASHN থেকে শুধু মানুষের অংশ বের করা (যাতে সোফা বা আসবাব Subject-এ না ঢোকে)
    fashn_person_prob = 1.0 - f_probs[0]
    person_soft_u8 = np.clip(((fashn_person_prob - 0.18) / 0.55) * 255.0, 0, 255).astype(np.uint8)
    person_u8 = np.minimum(ormbg_u8, person_soft_u8)

    winning_cls = np.argmax(f_probs, axis=0)
    face_mask_u8 = detect_face_mask(ormbg_u8, winning_cls, f_probs)
    if face_mask_u8 is not None:
        person_u8 = np.maximum(person_u8, face_mask_u8)

    person_area = int(np.count_nonzero(person_u8 >= 120))

    # মানুষ থাকলে Subject = শুধু মানুষ (সোফা বাদ!); মানুষ না থাকলে Subject = পুরো ORMBG বস্তু
    has_person = person_area >= max(400, int(img_area * 0.015))
    main_subject_u8 = remove_small_islands(person_u8, min_ratio=0.05, min_abs=20) if has_person else ormbg_u8

    # মানুষের বাইরে থাকা বড় বস্তু (যেমন সোফা, চেয়ার, গাড়ি, ল্যাপটপ) আলাদা করা
    env_objects = []
    if has_person and ormbg_area > person_area:
        env_raw = np.clip(ormbg_u8.astype(np.int16) - main_subject_u8.astype(np.int16), 0, 255).astype(np.uint8)
        if face_mask_u8 is not None:
            env_raw = np.minimum(env_raw, (255 - face_mask_u8).astype(np.uint8))
        # মানুষের চারপাশের পাতলা ২px হ্যালো মুছে ফেলা
        env_clean = Image.fromarray(env_raw, mode="L").filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
        env_u8 = np.asarray(env_clean, dtype=np.uint8)
        min_obj_px = max(600, int(img_area * 0.02))
        env_objects = extract_top_component_masks(env_u8, min_pixels_full=min_obj_px, max_objects=2)

    subj_gate = np.clip(main_subject_u8.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    subject_valid = main_subject_u8 >= 90
    subj_area = max(1, int(np.count_nonzero(main_subject_u8 >= 120)))

    top_cnt = int(np.count_nonzero((winning_cls == 3) & subject_valid))
    dress_cnt = int(np.count_nonzero((winning_cls == 4) & subject_valid))

    parts = []

    def add_smart_part(
        label: str,
        class_ids: list[int],
        min_ratio: float,
        min_abs: int,
        min_conf: float = 0.45,
        island_ratio: float = 0.05,
        erode: bool = True,
    ):
        win_mask = np.isin(winning_cls, class_ids) & subject_valid
        win_cnt = int(np.count_nonzero(win_mask))
        req_pixels = max(min_abs, int(subj_area * min_ratio))
        if win_cnt < req_pixels:
            return

        p_sum = np.sum(f_probs[class_ids], axis=0)
        if float(np.mean(p_sum[win_mask])) < min_conf:
            return

        part_u8 = np.clip(p_sum * 255.0, 0, 255).astype(np.uint8)
        part_u8 = np.minimum(part_u8, subj_gate)
        if face_mask_u8 is not None and label in ("Hair", "Face"):
            part_u8 = np.minimum(part_u8, (255 - face_mask_u8).astype(np.uint8))
        part_u8 = clean_part_mask(part_u8, island_ratio=island_ratio, erode=erode)
        if int(np.count_nonzero(part_u8 >= 120)) < min_abs // 2:
            return
        parts.append({
            "label": label,
            "mask": mask_u8_to_base64_rgba(part_u8),
        })

    if subj_area >= 50:
        parts.append({"label": "Subject", "mask": mask_u8_to_base64_rgba(main_subject_u8)})

    if len(env_objects) >= 1:
        obj_area_1 = int(np.count_nonzero(env_objects[0] >= 120))
        obj1_label = "Sofa / Object" if obj_area_1 >= int(img_area * 0.08) else "Object / Prop"
        parts.append({"label": obj1_label, "mask": mask_u8_to_base64_rgba(env_objects[0])})
    if len(env_objects) >= 2:
        parts.append({"label": "Object 2", "mask": mask_u8_to_base64_rgba(env_objects[1])})

    if has_person:
        add_smart_part("Shirt / Top", [3], min_ratio=0.025, min_abs=350)
        if not (top_cnt > dress_cnt * 2.2 and dress_cnt < subj_area * 0.08):
            add_smart_part("Dress", [4], min_ratio=0.04, min_abs=500)
        add_smart_part("Pants", [6], min_ratio=0.025, min_abs=350)
        add_smart_part("Skirt", [5], min_ratio=0.025, min_abs=350)
        add_smart_part("Shoes", [15], min_ratio=0.008, min_abs=140, island_ratio=0.12)
        add_smart_part("Hair", [2], min_ratio=0.012, min_abs=180, island_ratio=0.08)
        add_smart_part("Face", [1], min_ratio=0.012, min_abs=180, island_ratio=0.025, erode=False)

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

    # Background = মূল ORMBG পুরো ফোরগ্রাউন্ডের বিপরীত (যাতে সোফা/মানুষ সবই ব্যাকগ্রাউন্ডের বাইরে থাকে)
    bg_u8 = (255 - ormbg_u8).astype(np.uint8)
    if int(np.count_nonzero(bg_u8 >= 120)) >= 50:
        parts.append({"label": "Background", "mask": mask_u8_to_base64_rgba(bg_u8)})

    with _cache_lock:
        _result_cache[img_hash] = parts
        while len(_result_cache) > _CACHE_MAX:
            _result_cache.popitem(last=False)

    gc.collect()
    return {"success": True, "parts": parts}
