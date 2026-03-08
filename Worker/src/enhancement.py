import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import warnings

warnings.filterwarnings("ignore")

print("🚀 Initializing Real-ESRGAN Enhancement Engine...")

device     = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"⚡ Running on: {device.upper()}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

CACHE_DIR    = os.getenv("TORCH_HOME", "/app/.cache/torch")
ESRGAN_DIR   = os.path.join(CACHE_DIR, "realesrgan")
NAFNET_DIR   = os.path.join(CACHE_DIR, "nafnet")
os.makedirs(ESRGAN_DIR, exist_ok=True)
os.makedirs(NAFNET_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# 1. Real-ESRGAN — super-resolution (all modes)
# ──────────────────────────────────────────────────────────────────
ESRGAN_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ESRGAN_FILE = os.path.join(ESRGAN_DIR, "RealESRGAN_x4plus.pth")

enhancer = None
try:
    if not os.path.exists(ESRGAN_FILE):
        print("⬇️  Downloading Real-ESRGAN weights (~67 MB)...")
        load_file_from_url(url=ESRGAN_URL, model_dir=ESRGAN_DIR,
                           progress=True, file_name="RealESRGAN_x4plus.pth")

    _esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3,
                             num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    enhancer = RealESRGANer(
        scale=4, model_path=ESRGAN_FILE, model=_esrgan_model,
        tile=512, tile_pad=32, pre_pad=0,
        half=(device == "cuda"), device=device,
    )
    print("✅ Real-ESRGAN x4plus loaded.")
except Exception as e:
    print(f"❌ Real-ESRGAN Load Failed: {e}")

# ──────────────────────────────────────────────────────────────────
# 2. NAFNet — motion/defocus deblurring (document mode)
# Architecture bundled in src/nafnet_arch.py — no external repo needed.
# Weights: HuggingFace mirror of the official GoPro checkpoint.
# ──────────────────────────────────────────────────────────────────
NAFNET_URL  = ("https://huggingface.co/nyanko7/nafnet-models/resolve/main/"
               "NAFNet-GoPro-width64.pth")
NAFNET_FILE = os.path.join(NAFNET_DIR, "NAFNet-GoPro-width64.pth")

nafnet = None
try:
    if not os.path.exists(NAFNET_FILE):
        print("⬇️  Downloading NAFNet weights (~272 MB)...")
        load_file_from_url(url=NAFNET_URL, model_dir=NAFNET_DIR,
                           progress=True, file_name="NAFNet-GoPro-width64.pth")

    from src.nafnet_arch import NAFNet
    nafnet = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    checkpoint = torch.load(NAFNET_FILE, map_location=device)
    state = (checkpoint.get("params_ema")
             or checkpoint.get("params")
             or checkpoint)
    nafnet.load_state_dict(state, strict=True)
    nafnet.to(device).eval()
    print("✅ NAFNet deblur engine loaded.")

except Exception as e:
    print(f"⚠️  NAFNet unavailable ({e}) — falling back to Wiener deconvolution.")
    nafnet = None

# ──────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ──────────────────────────────────────────────────────────────────

def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _is_document(img_rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return (np.sum(gray > 200) / gray.size) > 0.55


# ──────────────────────────────────────────────────────────────────
# DOCUMENT DEBLUR  (NAFNet or Wiener fallback)
# ──────────────────────────────────────────────────────────────────

def _nafnet_deblur(img_rgb: np.ndarray) -> np.ndarray:
    """Runs NAFNet inference. Input/output: uint8 RGB numpy array."""
    inp = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    inp = inp.to(device)

    # NAFNet requires dimensions divisible by 16 — pad if necessary
    _, _, h, w = inp.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode():
        out = nafnet(inp)

    # Crop padding back off
    out = out[:, :, :h, :w]
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (out * 255).astype(np.uint8)


def _wiener_deblur(img_rgb: np.ndarray, blur: float) -> np.ndarray:
    """
    Wiener filter fallback when NAFNet is unavailable.
    Uses skimage's unsupervised_wiener which estimates the noise
    level automatically — more stable than Richardson-Lucy.
    """
    from skimage.restoration import unsupervised_wiener
    from skimage.draw import disk

    radius = max(1, min(5, int(3.5 - (blur - 20) / 60)))
    psf = np.zeros((radius * 2 + 1, radius * 2 + 1))
    rr, cc = disk((radius, radius), radius + 0.5)
    psf[rr, cc] = 1
    psf = psf / psf.sum()

    out = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        ch = img_rgb[:, :, c].astype(np.float64) / 255.0
        restored, _ = unsupervised_wiener(ch, psf)
        out[:, :, c] = np.clip(restored, 0, 1) * 255
    return out.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────
# PHOTO PRE-PROCESSING  (before ESRGAN)
#
# This version was confirmed to produce the best photo results.
# CLAHE normalises local contrast; the unsharp mask recovers
# apparent sharpness that ESRGAN then preserves at 4× resolution.
# ──────────────────────────────────────────────────────────────────

def _preprocess_photo(img_rgb: np.ndarray, blur: float) -> np.ndarray:
    img = img_rgb.copy()

    # CLAHE on luminance
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # Unsharp mask scaled to blur severity
    if blur < 300:
        sigma  = 2.0 if blur < 80 else 1.2
        amount = 1.8 if blur < 80 else 1.2
        blurred_gauss = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1 + amount, blurred_gauss, -amount, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# ──────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ──────────────────────────────────────────────────────────────────

def _postprocess_document(img_rgb: np.ndarray) -> np.ndarray:
    """
    After NAFNet + ESRGAN: adaptive threshold blend for crisp text,
    bilateral filter to suppress any residual ringing on paper.
    """
    # Bilateral: smooth paper without blurring ink edges
    smooth = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=20, sigmaSpace=20)

    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold — handles uneven page illumination
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=9
    )
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # 50/50 blend: preserves color ink, gains crispness of threshold
    blended = cv2.addWeighted(smooth, 0.45, thresh_rgb, 0.55, 0)

    pil = Image.fromarray(blended)
    pil = ImageEnhance.Sharpness(pil).enhance(2.0)
    pil = ImageEnhance.Contrast(pil).enhance(1.3)
    return np.array(pil)


def _postprocess_photo(pil_img: Image.Image, blur: float) -> Image.Image:
    blur_t = max(0.0, min(1.0, (300 - blur) / 300))
    usm_pct = int(100 + blur_t * 120)

    img = pil_img.filter(ImageFilter.UnsharpMask(
        radius=1.5, percent=usm_pct, threshold=2
    ))
    img = ImageEnhance.Sharpness(img).enhance(1.6 + blur_t * 0.8)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    return img


# ──────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────

def enhance_image(img_array: np.ndarray) -> np.ndarray | str:
    if enhancer is None:
        return "Error: Real-ESRGAN engine failed to load. Check worker logs."

    try:
        h, w   = img_array.shape[:2]
        gray   = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur   = _blur_score(gray)
        is_doc = _is_document(img_array)

        print(f"🔍 Input: {w}×{h}px | blur={blur:.0f} | "
              f"mode={'document' if is_doc else 'photo'}")

        if is_doc:
            # ── Stage 1: neural deblur (NAFNet or Wiener fallback) ──
            if nafnet is not None:
                print("   NAFNet deblurring...")
                deblurred = _nafnet_deblur(img_array)
            else:
                print("   Wiener deblurring (NAFNet unavailable)...")
                deblurred = _wiener_deblur(img_array, blur)

            # ── Stage 2: ESRGAN 4× upscale ────────────────────────
            bgr_in        = deblurred[:, :, ::-1].copy()
            output_bgr, _ = enhancer.enhance(bgr_in, outscale=4.0)
            output_rgb    = output_bgr[:, :, ::-1].copy()

            # ── Stage 3: adaptive threshold + sharpening ───────────
            result = _postprocess_document(output_rgb)

        else:
            # ── Stage 1: pre-process (CLAHE + unsharp mask) ────────
            preprocessed = _preprocess_photo(img_array, blur)

            # ── Stage 2: ESRGAN 4× upscale ────────────────────────
            bgr_in        = preprocessed[:, :, ::-1].copy()
            output_bgr, _ = enhancer.enhance(bgr_in, outscale=4.0)
            output_rgb    = output_bgr[:, :, ::-1].copy()

            # ── Stage 3: post-sharpen ──────────────────────────────
            pil    = Image.fromarray(output_rgb.astype(np.uint8))
            result = np.array(_postprocess_photo(pil, blur))

        result = result.astype(np.uint8)
        oh, ow = result.shape[:2]
        print(f"✨ Done: {w}×{h} → {ow}×{oh}px")
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return "Error: GPU OOM — reduce tile size from 512 to 256."
        return f"Runtime Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"