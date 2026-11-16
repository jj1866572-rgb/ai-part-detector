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
from PIL import Image, ImageFilter
from pydantic import BaseModel

app = FastAPI(title="AI Subject & Face/Body Parts Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"
TMP_MODELS_DIR = Path("/tmp/models")

FACE_MODEL_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
CLOTHES_MODEL_URL = "https://huggingface.co/Xenova/segformer_b0_clothes/resolve/main/onnx/model.onnx"

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
    if local_path.exists() and local_path.stat().st_size > 100_000:
        return str(local_path)

    TMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_MODELS_DIR / filename
    if not tmp_path.exists() or tmp_path.stat().st_size < 100_000:
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

            face_path = ensure_model("face_resnet18.onnx", FACE_MODEL_URL)
            clothes_path = ensure_model("clothes_b0.onnx", CLOTHES_MODEL_URL)

            _face_sess = ort.InferenceSession(face_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _clothes_sess = ort.InferenceSession(clothes_path, sess_options=opts, providers=["CPUExecutionProvider"])

    return _face_sess, _clothes_sess


def preprocess(pil_img: Image.Image, size: int = 512) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def run_clothes_model(clothes_sess: ort.InferenceSession, img: Image.Image) -> np.ndarray:
    inp_name = clothes_sess.get_inputs()[0].name
    inp = preprocess(img, 512)
    logits = clothes_sess.run(None, {inp_name: inp})[0]
    return np.argmax(logits[0], axis=0).astype(np.uint8)


def run_face_model_smart(face_sess: ort.InferenceSession, orig_img: Image.Image, clothes_map: np.ndarray) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = face_sess.get_inputs()[0].name

    # মাথা ও মুখের অংশ (Hat=1, Hair=2, Sunglasses=3, Face=11) খুঁজে বের করে স্মার্ট ক্রপ
    head_mask = np.isin(clothes_map, [1, 2, 3, 11])
    ys, xs = np.where(head_mask)

    use_crop = False
    x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    if len(xs) >= 12:
        ch_h, ch_w = clothes_map.shape
        bx1 = int(xs.min() * orig_w / ch_w)
        bx2 = int((xs.max() + 1) * orig_w / ch_w)
        by1 = int(ys.min() * orig_h / ch_h)
        by2 = int((ys.max() + 1) * orig_h / ch_h)

        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        pad_x = int(bw * 0.35)
        pad_y = int(bh * 0.35)

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
        logits = face_sess.run(None, {inp_name: inp})[0]
        pred_512 = np.argmax(logits[0], axis=0).astype(np.uint8)
        pred_crop = np.asarray(
            Image.fromarray(pred_512, mode="L").resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        full_face_map = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_face_map[y1:y2, x1:x2] = pred_crop
        return full_face_map
    else:
        inp = preprocess(orig_img, 512)
        logits = face_sess.run(None, {inp_name: inp})[0]
        pred_512 = np.argmax(logits[0], axis=0).astype(np.uint8)
        return np.asarray(
            Image.fromarray(pred_512, mode="L").resize((orig_w, orig_h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )


def upsample_bool_mask(small_bool: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    u8 = (small_bool.astype(np.uint8)) * 255
    img = Image.fromarray(u8, mode="L").resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def mask_u8_to_base64_rgba(mask_u8: np.ndarray, feather: bool = True) -> str:
    h, w = mask_u8.shape
    if feather:
        alpha_img = Image.fromarray(mask_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=0.6))
        alpha_arr = np.asarray(alpha_img, dtype=np.uint8)
    else:
        alpha_arr = mask_u8

    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = alpha_arr

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "AI Segmenter API is live! Open /api/warmup to pre-load models.",
    }


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {
            "ready": True,
            "models": ["BiSeNet-ResNet18 (Face)", "SegFormer-B0 (Clothes/Body)"],
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

    orig_w, orig_h = orig_img.size
    face_sess, clothes_sess = get_sessions()

    clothes_small = run_clothes_model(clothes_sess, orig_img)
    face_full = run_face_model_smart(face_sess, orig_img, clothes_small)

    clothes_fg_u8 = upsample_bool_mask(clothes_small != 0, orig_w, orig_h)
    face_fg_u8 = ((face_full != 0).astype(np.uint8)) * 255

    subject_u8 = np.maximum(clothes_fg_u8, face_fg_u8)
    bg_u8 = (255 - subject_u8).astype(np.uint8)

    parts = []

    def add_part(label: str, mask_u8: np.ndarray, min_pixels: int = 20, feather: bool = True):
        if int(np.count_nonzero(mask_u8 >= 120)) >= min_pixels:
            parts.append({
                "label": label,
                "mask": mask_u8_to_base64_rgba(mask_u8, feather=feather),
            })

    add_part("Subject", subject_u8, min_pixels=50)

    hair_clothes_u8 = upsample_bool_mask(clothes_small == 2, orig_w, orig_h)
    hair_face_u8 = ((face_full == 17).astype(np.uint8)) * 255
    add_part("Hair", np.maximum(hair_clothes_u8, hair_face_u8), min_pixels=35)

    skin_u8 = ((face_full == 1).astype(np.uint8)) * 255
    if np.count_nonzero(skin_u8 >= 120) < 25:
        skin_u8 = upsample_bool_mask(clothes_small == 11, orig_w, orig_h)
    add_part("Face Skin", skin_u8, min_pixels=30)

    l_eye_u8 = ((face_full == 4).astype(np.uint8)) * 255
    r_eye_u8 = ((face_full == 5).astype(np.uint8)) * 255
    add_part("Eyes", np.maximum(l_eye_u8, r_eye_u8), min_pixels=8, feather=False)
    add_part("Left Eye", l_eye_u8, min_pixels=6, feather=False)
    add_part("Right Eye", r_eye_u8, min_pixels=6, feather=False)

    brows_u8 = (np.isin(face_full, [2, 3]).astype(np.uint8)) * 255
    add_part("Eyebrows", brows_u8, min_pixels=8, feather=False)

    nose_u8 = ((face_full == 10).astype(np.uint8)) * 255
    add_part("Nose", nose_u8, min_pixels=10, feather=False)

    l_ear_u8 = ((face_full == 7).astype(np.uint8)) * 255
    r_ear_u8 = ((face_full == 8).astype(np.uint8)) * 255
    add_part("Ears", np.maximum(l_ear_u8, r_ear_u8), min_pixels=10, feather=False)
    add_part("Left Ear", l_ear_u8, min_pixels=8, feather=False)
    add_part("Right Ear", r_ear_u8, min_pixels=8, feather=False)

    lips_u8 = (np.isin(face_full, [11, 12, 13]).astype(np.uint8)) * 255
    add_part("Lips", lips_u8, min_pixels=10, feather=False)

    glasses_u8 = np.maximum(
        ((face_full == 6).astype(np.uint8)) * 255,
        upsample_bool_mask(clothes_small == 3, orig_w, orig_h),
    )
    add_part("Glasses", glasses_u8, min_pixels=20)

    upper_u8 = upsample_bool_mask(clothes_small == 4, orig_w, orig_h)
    if np.count_nonzero(upper_u8 >= 120) < 60:
        upper_u8 = ((face_full == 16).astype(np.uint8)) * 255
    add_part("Shirt / Top", upper_u8, min_pixels=60)

    add_part("Pants", upsample_bool_mask(clothes_small == 6, orig_w, orig_h), min_pixels=60)
    add_part("Dress", upsample_bool_mask(clothes_small == 7, orig_w, orig_h), min_pixels=60)
    add_part("Skirt", upsample_bool_mask(clothes_small == 5, orig_w, orig_h), min_pixels=50)

    shoes_u8 = upsample_bool_mask(np.isin(clothes_small, [9, 10]), orig_w, orig_h)
    add_part("Shoes", shoes_u8, min_pixels=40)

    hat_u8 = np.maximum(
        upsample_bool_mask(clothes_small == 1, orig_w, orig_h),
        ((face_full == 18).astype(np.uint8)) * 255,
    )
    add_part("Hat", hat_u8, min_pixels=35)
    add_part("Bag", upsample_bool_mask(clothes_small == 16, orig_w, orig_h), min_pixels=45)
    add_part("Scarf", upsample_bool_mask(clothes_small == 17, orig_w, orig_h), min_pixels=40)
    add_part("Belt", upsample_bool_mask(clothes_small == 8, orig_w, orig_h), min_pixels=25)
    add_part("Neck", ((face_full == 14).astype(np.uint8)) * 255, min_pixels=25)
    add_part("Arms", upsample_bool_mask(np.isin(clothes_small, [14, 15]), orig_w, orig_h), min_pixels=50)
    add_part("Legs", upsample_bool_mask(np.isin(clothes_small, [12, 13]), orig_w, orig_h), min_pixels=50)

    add_part("Background", bg_u8, min_pixels=50)

    return {"success": True, "parts": parts}
