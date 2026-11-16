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

app = FastAPI(title="Pro Photoshop-Grade Subject & Parts Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_MODELS_DIR = Path("/tmp/models")

# Pro-grade ONNX models with fallback mirrors
RMBG_URLS = [
    "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
    "https://huggingface.co/SolonD/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
    "https://huggingface.co/onnx-community/ISNet-ONNX/resolve/main/onnx/model_quantized.onnx",
]
FACE_PARSE_URLS = [
    "https://huggingface.co/Xenova/face-parsing/resolve/main/onnx/model_quantized.onnx",
    "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx",
]
CLOTHES_URLS = [
    "https://huggingface.co/Xenova/segformer_b2_clothes/resolve/main/onnx/model_quantized.onnx",
    "https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx",
    "https://huggingface.co/Xenova/segformer_b0_clothes/resolve/main/onnx/model.onnx",
]
ULTRAFACE_URLS = [
    "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx",
    "https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/main/models/version-RFB-320.onnx",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_lock = threading.Lock()
_rmbg_sess = None
_face_sess = None
_clothes_sess = None
_det_sess = None


class SegmentRequest(BaseModel):
    image: str


def download_with_fallback(urls: list[str], dest_path: Path):
    last_err = None
    for url in urls:
        tmp_part = dest_path.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_part, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp_part.stat().st_size > 100_000:
                os.replace(tmp_part, dest_path)
                return
        except Exception as e:
            last_err = e
            if tmp_part.exists():
                tmp_part.unlink(missing_ok=True)
    raise RuntimeError(f"Failed downloading {dest_path.name}: {last_err}")


def ensure_model(filename: str, urls: list[str]) -> str:
    TMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = TMP_MODELS_DIR / filename
    if not dest.exists() or dest.stat().st_size < 100_000:
        download_with_fallback(urls, dest)
    return str(dest)


def get_sessions():
    global _rmbg_sess, _face_sess, _clothes_sess, _det_sess
    if _rmbg_sess is not None and _face_sess is not None and _clothes_sess is not None and _det_sess is not None:
        return _rmbg_sess, _face_sess, _clothes_sess, _det_sess

    with _model_lock:
        if _rmbg_sess is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2

            rmbg_p = ensure_model("rmbg_14_q.onnx", RMBG_URLS)
            face_p = ensure_model("segformer_face_q.onnx", FACE_PARSE_URLS)
            clothes_p = ensure_model("segformer_b2_clothes_q.onnx", CLOTHES_URLS)
            det_p = ensure_model("ultraface_320.onnx", ULTRAFACE_URLS)

            _rmbg_sess = ort.InferenceSession(rmbg_p, sess_options=opts, providers=["CPUExecutionProvider"])
            _face_sess = ort.InferenceSession(face_p, sess_options=opts, providers=["CPUExecutionProvider"])
            _clothes_sess = ort.InferenceSession(clothes_p, sess_options=opts, providers=["CPUExecutionProvider"])
            _det_sess = ort.InferenceSession(det_p, sess_options=opts, providers=["CPUExecutionProvider"])

    return _rmbg_sess, _face_sess, _clothes_sess, _det_sess


def preprocess_imagenet(pil_img: Image.Image, size: int = 512) -> np.ndarray:
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def run_rmbg(rmbg_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    resized = orig_img.resize((1024, 1024), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0 - 0.5
    inp = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...], dtype=np.float32)

    inp_name = rmbg_sess.get_inputs()[0].name
    out = rmbg_sess.run(None, {inp_name: inp})[0]
    matte = np.squeeze(out).astype(np.float32)
    mn, mx = float(matte.min()), float(matte.max())
    if mx - mn > 1e-6:
        matte = (matte - mn) / (mx - mn)
    else:
        matte = np.zeros_like(matte)

    matte_img = Image.fromarray(np.clip(matte * 255.0, 0, 255).astype(np.uint8), mode="L")
    matte_full = matte_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    return np.asarray(matte_full, dtype=np.uint8)


def detect_main_face_box(det_sess: ort.InferenceSession, orig_img: Image.Image):
    orig_w, orig_h = orig_img.size
    resized = orig_img.resize((320, 240), Image.Resampling.BILINEAR)
    arr = (np.asarray(resized, dtype=np.float32) - 127.0) / 128.0
    inp = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...], dtype=np.float32)

    inp_name = det_sess.get_inputs()[0].name
    outs = det_sess.run(None, {inp_name: inp})

    scores = None
    boxes = None
    for o in outs:
        if o.ndim == 3 and o.shape[-1] == 2:
            scores = o[0]
        elif o.ndim == 3 and o.shape[-1] == 4:
            boxes = o[0]
    if scores is None or boxes is None:
        return None

    face_probs = scores[:, 1]
    best_idx = int(np.argmax(face_probs))
    if float(face_probs[best_idx]) < 0.60:
        return None

    # Multiple faces check: if >1 distinct strong face, parse full image
    strong = np.where(face_probs > 0.75)[0]
    if len(strong) > 1:
        b_strong = boxes[strong]
        spread_x = float(b_strong[:, 0].max() - b_strong[:, 0].min())
        if spread_x > 0.28:
            return None

    x1n, y1n, x2n, y2n = boxes[best_idx]
    bx1 = max(0, int(x1n * orig_w))
    by1 = max(0, int(y1n * orig_h))
    bx2 = min(orig_w, int(x2n * orig_w))
    by2 = min(orig_h, int(y2n * orig_h))
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    # Generous padding for hair (top), ears (sides), neck (bottom)
    px1 = max(0, bx1 - int(bw * 0.55))
    px2 = min(orig_w, bx2 + int(bw * 0.55))
    py1 = max(0, by1 - int(bh * 0.78))
    py2 = min(orig_h, by2 + int(bh * 0.50))

    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2
    half_side = max(px2 - px1, py2 - py1) // 2

    sx1 = max(0, cx - half_side)
    sy1 = max(0, cy - half_side)
    sx2 = min(orig_w, cx + half_side)
    sy2 = min(orig_h, cy + half_side)

    if (sx2 - sx1) * (sy2 - sy1) >= 0.72 * orig_w * orig_h:
        return None
    if (sx2 - sx1) < 64 or (sy2 - sy1) < 64:
        return None
    return (sx1, sy1, sx2, sy2)


def logits_to_upsampled_probs(logits_chw: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    c = logits_chw.shape[0]
    mx = np.max(logits_chw, axis=0, keepdims=True)
    ex = np.exp(logits_chw - mx)
    probs = ex / (np.sum(ex, axis=0, keepdims=True) + 1e-8)

    up = np.empty((c, out_h, out_w), dtype=np.float32)
    for idx in range(c):
        p_img = Image.fromarray(probs[idx].astype(np.float32), mode="F").resize(
            (out_w, out_h), Image.Resampling.BILINEAR
        )
        up[idx] = np.asarray(p_img, dtype=np.float32)
    return up


def run_face_parsing(face_sess: ort.InferenceSession, det_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = face_sess.get_inputs()[0].name
    box = detect_main_face_box(det_sess, orig_img)

    if box is not None:
        x1, y1, x2, y2 = box
        crop = orig_img.crop((x1, y1, x2, y2))
        logits = face_sess.run(None, {inp_name: preprocess_imagenet(crop, 512)})[0][0]
        crop_probs = logits_to_upsampled_probs(logits, x2 - x1, y2 - y1)
        full_probs = np.zeros((19, orig_h, orig_w), dtype=np.float32)
        full_probs[0, :, :] = 1.0
        full_probs[:, y1:y2, x1:x2] = crop_probs
        return full_probs
    else:
        logits = face_sess.run(None, {inp_name: preprocess_imagenet(orig_img, 512)})[0][0]
        return logits_to_upsampled_probs(logits, orig_w, orig_h)


def run_clothes_parsing(clothes_sess: ort.InferenceSession, orig_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = orig_img.size
    inp_name = clothes_sess.get_inputs()[0].name
    logits = clothes_sess.run(None, {inp_name: preprocess_imagenet(orig_img, 512)})[0][0]
    return logits_to_upsampled_probs(logits, orig_w, orig_h)


def prob_to_u8(prob_arr: np.ndarray, subject_u8: np.ndarray | None = None) -> np.ndarray:
    u8 = np.clip(prob_arr * 255.0, 0, 255).astype(np.uint8)
    if subject_u8 is not None:
        # Keep clean Photoshop edge from RMBG-1.4 so nothing bleeds into background
        soft_clip = np.clip(subject_u8.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        u8 = np.minimum(u8, soft_clip)
    return u8


def mask_u8_to_base64_rgba(mask_u8: np.ndarray) -> str:
    h, w = mask_u8.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = mask_u8
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "RMBG-1.4 + SegFormer-Face + SegFormer-B2-Clothes + UltraFace"}


@app.get("/api/warmup")
def warmup():
    try:
        get_sessions()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")


@app.post("/api/segment")
def segment_parts(req: SegmentRequest):
    try:
        b64_data = req.image.split(",")[-1].strip()
        orig_img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    rmbg_sess, face_sess, clothes_sess, det_sess = get_sessions()

    # 1. Photoshop-grade Subject Alpha Matte
    subject_u8 = run_rmbg(rmbg_sess, orig_img)
    bg_u8 = (255 - subject_u8).astype(np.uint8)

    # 2. Transformer Face Parsing (Xenova/face-parsing id2label)
    # 0:bg, 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow,
    # 8:l_ear, 9:r_ear, 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat, 15:ear_r, 16:neck_l, 17:neck, 18:cloth
    fp = run_face_parsing(face_sess, det_sess, orig_img)
    fp_cls = np.argmax(fp, axis=0)

    # 3. Transformer Clothes Parsing (segformer_b2_clothes)
    # 0:bg, 1:Hat, 2:Hair, 3:Sunglasses, 4:Upper-clothes, 5:Skirt, 6:Pants, 7:Dress,
    # 8:Belt, 9:L-shoe, 10:R-shoe, 11:Face, 12:L-leg, 13:R-leg, 14:L-arm, 15:R-arm, 16:Bag, 17:Scarf
    cp = run_clothes_parsing(clothes_sess, orig_img)
    cp_cls = np.argmax(cp, axis=0)

    parts = []

    def add_part(label: str, mask_u8: np.ndarray, min_pixels: int = 20):
        if int(np.count_nonzero(mask_u8 >= 120)) >= min_pixels:
            parts.append({"label": label, "mask": mask_u8_to_base64_rgba(mask_u8)})

    add_part("Subject", subject_u8, min_pixels=50)

    # Hair: Use SegFormer-B5 face-parsing Hair (13) primarily; fallback to B2 clothes Hair (2) only if face hair missing
    hair_fp_u8 = prob_to_u8(fp[13] * (fp_cls == 13).astype(np.float32), subject_u8)
    if np.count_nonzero(hair_fp_u8 >= 120) >= 35:
        add_part("Hair", hair_fp_u8, min_pixels=35)
    else:
        add_part("Hair", prob_to_u8(cp[2], subject_u8), min_pixels=35)

    # Face Skin (1)
    skin_u8 = prob_to_u8(fp[1], subject_u8)
    if np.count_nonzero(skin_u8 >= 120) < 30:
        skin_u8 = prob_to_u8(cp[11], subject_u8)
    add_part("Face Skin", skin_u8, min_pixels=30)

    # Eyes (4: l_eye, 5: r_eye)
    l_eye_u8 = prob_to_u8(fp[4], subject_u8)
    r_eye_u8 = prob_to_u8(fp[5], subject_u8)
    add_part("Eyes", prob_to_u8(np.clip(fp[4] + fp[5], 0, 1), subject_u8), min_pixels=8)
    add_part("Left Eye", l_eye_u8, min_pixels=6)
    add_part("Right Eye", r_eye_u8, min_pixels=6)

    # Eyebrows (6: l_brow, 7: r_brow)
    add_part("Eyebrows", prob_to_u8(np.clip(fp[6] + fp[7], 0, 1), subject_u8), min_pixels=8)

    # Nose (2 in Xenova/face-parsing)
    add_part("Nose", prob_to_u8(fp[2], subject_u8), min_pixels=10)

    # Ears (8: l_ear, 9: r_ear)
    add_part("Ears", prob_to_u8(np.clip(fp[8] + fp[9], 0, 1), subject_u8), min_pixels=10)
    add_part("Left Ear", prob_to_u8(fp[8], subject_u8), min_pixels=8)
    add_part("Right Ear", prob_to_u8(fp[9], subject_u8), min_pixels=8)

    # Lips (10: mouth, 11: u_lip, 12: l_lip)
    add_part("Lips", prob_to_u8(np.clip(fp[10] + fp[11] + fp[12], 0, 1), subject_u8), min_pixels=10)

    # Glasses (3: eye_g / 3: Sunglasses)
    add_part("Glasses", prob_to_u8(np.maximum(fp[3], cp[3]), subject_u8), min_pixels=20)

    # Shirt / Top (4: Upper-clothes, fallback 18: cloth)
    upper_u8 = prob_to_u8(cp[4] * (cp_cls == 4).astype(np.float32), subject_u8)
    if np.count_nonzero(upper_u8 >= 120) < 60:
        upper_u8 = prob_to_u8(fp[18], subject_u8)
    add_part("Shirt / Top", upper_u8, min_pixels=60)

    add_part("Pants", prob_to_u8(cp[6], subject_u8), min_pixels=60)
    add_part("Dress", prob_to_u8(cp[7], subject_u8), min_pixels=60)
    add_part("Skirt", prob_to_u8(cp[5], subject_u8), min_pixels=50)
    add_part("Shoes", prob_to_u8(np.clip(cp[9] + cp[10], 0, 1), subject_u8), min_pixels=40)
    add_part("Hat", prob_to_u8(np.maximum(fp[14], cp[1]), subject_u8), min_pixels=35)
    add_part("Bag", prob_to_u8(cp[16], subject_u8), min_pixels=45)
    add_part("Scarf", prob_to_u8(cp[17], subject_u8), min_pixels=40)
    add_part("Belt", prob_to_u8(cp[8], subject_u8), min_pixels=25)
    add_part("Neck", prob_to_u8(fp[17], subject_u8), min_pixels=25)
    add_part("Arms", prob_to_u8(np.clip(cp[14] + cp[15], 0, 1), subject_u8), min_pixels=50)
    add_part("Legs", prob_to_u8(np.clip(cp[12] + cp[13], 0, 1), subject_u8), min_pixels=50)

    add_part("Background", bg_u8, min_pixels=50)

    return {"success": True, "parts": parts}
