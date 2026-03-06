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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Running on: {device.upper()}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

WEIGHTS_DIR  = os.path.join(os.getenv("TORCH_HOME", "/app/.cache/torch"), "realesrgan")
WEIGHTS_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

enhancer = None
try:
    if not os.path.exists(WEIGHTS_FILE):
        print("⬇️  Downloading Real-ESRGAN weights (~67 MB)...")
        load_file_from_url(url=WEIGHTS_URL, model_dir=WEIGHTS_DIR,
                           progress=True, file_name="RealESRGAN_x4plus.pth")

    model = RRDBNet(num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    enhancer = RealESRGANer(
        scale=4,
        model_path=WEIGHTS_FILE,
        model=model,
        tile=512,
        tile_pad=32,
        pre_pad=0,
        half=(device == "cuda"),
        device=device,
    )
    print("✅ Real-ESRGAN x4plus Engine loaded and ready.")

except Exception as e:
    print(f"❌ Real-ESRGAN Load Failed: {e}")
    enhancer = None


# ──────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────

def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_document(img_rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return (np.sum(gray > 200) / gray.size) > 0.55


# ──────────────────────────────────────────────────────────────────
# DOCUMENT PIPELINE
# The fundamental problem with blurry documents is optical camera
# blur — ESRGAN cannot remove this because it was not trained for it.
# Richardson-Lucy deconvolution reverses the blur mathematically before
# ESRGAN runs, so ESRGAN receives a deblurred image and focuses on
# reconstructing fine stroke detail at 4× resolution.
# ──────────────────────────────────────────────────────────────────

def _make_psf(radius: float, size: int = 15) -> np.ndarray:
    """
    Builds a Gaussian point-spread function approximating the camera blur.
    radius is estimated from the blur score: lower score = more blur = wider PSF.
    """
    k = np.zeros((size, size), dtype=np.float64)
    centre = size // 2
    for i in range(size):
        for j in range(size):
            d = ((i - centre) ** 2 + (j - centre) ** 2) ** 0.5
            k[i, j] = np.exp(-d ** 2 / (2 * radius ** 2))
    k /= k.sum()
    return k


def _richardson_lucy(channel: np.ndarray, psf: np.ndarray,
                     iterations: int = 25) -> np.ndarray:
    """
    Richardson-Lucy deconvolution on a single float channel [0, 1].
    Uses OpenCV filter2D for the convolution steps — no scipy required.
    """
    psf_mirror = np.flip(psf)
    u = channel.copy()
    eps = 1e-8

    for _ in range(iterations):
        conv = cv2.filter2D(u, -1, psf.astype(np.float32))
        relative_blur = channel / (conv + eps)
        correction = cv2.filter2D(relative_blur, -1, psf_mirror.astype(np.float32))
        u = u * correction
        u = np.clip(u, 0, 1)

    return u


def _deblur_document(img_rgb: np.ndarray, blur: float) -> np.ndarray:
    """
    Applies Richardson-Lucy deconvolution per channel.
    PSF radius is derived from the blur score: a lower score means
    more blur and therefore a wider point-spread function to invert.
    """
    # Map blur score to PSF radius: blur=20 → radius=3.5, blur=100 → radius=1.8
    radius = max(1.0, min(4.0, 3.5 - (blur - 20) / 50))
    iterations = 30 if blur < 60 else 20
    print(f"   RL deconv: PSF radius={radius:.1f}, iterations={iterations}")

    psf = _make_psf(radius)
    out = np.zeros_like(img_rgb, dtype=np.float32)

    for c in range(3):
        channel = img_rgb[:, :, c].astype(np.float64) / 255.0
        deblurred = _richardson_lucy(channel, psf, iterations)
        out[:, :, c] = (deblurred * 255).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def _postprocess_document(img_rgb: np.ndarray) -> np.ndarray:
    """
    Post-ESRGAN document finishing.

    At 4× resolution the deconvolved strokes are sharp but may have
    slight ringing (RL deconv artifact). A bilateral filter removes
    ringing from the paper area while preserving ink edges, then
    adaptive thresholding produces clean black-on-white text.
    """
    # ── Bilateral: suppress ringing on paper background ────────────
    # Edge-aware: stops at ink-paper boundary automatically.
    smoothed = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=20, sigmaSpace=20)

    # ── Convert to grayscale for thresholding analysis ─────────────
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)

    # ── Adaptive threshold: produces crisp black text on white paper.
    # Block size 31 handles varying illumination across the page.
    # C=9 subtracts a constant to ensure faint strokes aren't lost.
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=9
    )

    # ── Blend: 50% threshold + 50% bilateral to preserve any colour
    # ink while still getting the crispness benefit of thresholding.
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(smoothed, 0.45, thresh_rgb, 0.55, 0)

    # ── Final sharpness lift ───────────────────────────────────────
    pil = Image.fromarray(blended)
    pil = ImageEnhance.Sharpness(pil).enhance(2.0)
    pil = ImageEnhance.Contrast(pil).enhance(1.3)

    return np.array(pil)


# ──────────────────────────────────────────────────────────────────
# PHOTO PIPELINE
# Critical rule: do NOT pre-process before ESRGAN.
# Pre-sharpening creates ringing artifacts that ESRGAN amplifies.
# ESRGAN needs clean, natural input — all enhancement is post-inference.
# ──────────────────────────────────────────────────────────────────

def _postprocess_photo(pil_img: Image.Image, blur: float) -> Image.Image:
    """
    Photo post-processing only — no pre-processing.

    Sharpening strength scales with measured blur so a crisp input
    gets a gentle finishing pass while a very blurry one gets the
    full treatment.
    """
    blur_t = max(0.0, min(1.0, (400 - blur) / 400))

    # Unsharp mask: radius 1.5 targets pixel-level detail at 4× res
    usm_pct = int(120 + blur_t * 130)   # 120–250
    img = pil_img.filter(ImageFilter.UnsharpMask(
        radius=1.5, percent=usm_pct, threshold=2
    ))

    # Sharpness: 1.8 base + up to 1.0 extra for very blurry inputs
    img = ImageEnhance.Sharpness(img).enhance(1.8 + blur_t * 1.0)

    # Mild contrast lift — improves perceived detail without clipping
    img = ImageEnhance.Contrast(img).enhance(1.1)

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
            # ── Document: deconvolve first, then ESRGAN, then threshold
            print("   Running Richardson-Lucy deconvolution...")
            deblurred = _deblur_document(img_array, blur)

            bgr_in        = deblurred[:, :, ::-1].copy()
            output_bgr, _ = enhancer.enhance(bgr_in, outscale=4.0)
            output_rgb    = output_bgr[:, :, ::-1].copy()

            result = _postprocess_document(output_rgb)

        else:
            # ── Photo: ESRGAN on raw input, sharpen after
            bgr_in        = img_array[:, :, ::-1].copy()
            output_bgr, _ = enhancer.enhance(bgr_in, outscale=4.0)
            output_rgb    = output_bgr[:, :, ::-1].copy()

            output_pil = Image.fromarray(output_rgb.astype(np.uint8))
            result = np.array(_postprocess_photo(output_pil, blur))

        result = result.astype(np.uint8)
        oh, ow = result.shape[:2]
        print(f"✨ Done: {w}×{h} → {ow}×{oh}px")
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return "Error: GPU OOM — reduce tile size from 512 to 256 in enhancement.py."
        return f"Runtime Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"