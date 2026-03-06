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
# ANALYSIS HELPERS
# ──────────────────────────────────────────────────────────────────

def _blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — lower = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_document(img_rgb: np.ndarray) -> bool:
    """
    True when >55% of pixels are very bright (paper/whiteboard).
    Documents need a completely different processing pipeline to photos.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return (np.sum(gray > 200) / gray.size) > 0.55


# ──────────────────────────────────────────────────────────────────
# DOCUMENT PIPELINE
# ──────────────────────────────────────────────────────────────────

def _preprocess_document(img_rgb: np.ndarray, blur: float) -> np.ndarray:
    """
    Document pre-processing pipeline.

    The key insight: documents have two distinct regions — ink (dark,
    high-frequency edges) and paper (bright, should be clean white).
    Treating them uniformly amplifies paper grain into visible noise.

    Strategy:
      1. Denoise first — remove grain from paper background BEFORE any
         sharpening so the sharpening step can't amplify it.
      2. CLAHE only on the ink region — boost contrast of dark text
         strokes without touching the already-bright background.
      3. Mild unsharp mask — a restrained pass that lifts edge contrast
         without introducing haloing on letter strokes.
    """
    img = img_rgb.copy()

    # ── Step 1: Denoise — MUST come before any sharpening ──────────
    # h=6 is conservative: removes grain while preserving stroke edges.
    # fastNlMeansDenoisingColored works in LAB space internally, which
    # prevents colour bleeding at ink-paper boundaries.
    img = cv2.fastNlMeansDenoisingColored(img, None,
                                          h=6, hColor=6,
                                          templateWindowSize=7,
                                          searchWindowSize=21)

    # ── Step 2: Targeted CLAHE on ink region only ──────────────────
    # Build a mask of dark pixels (ink) to restrict CLAHE application.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, ink_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Low clip limit (2.0) — only enhances existing contrast, doesn't
    # create new contrast that would make grain visible.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Blend: apply enhanced L only where ink is present
    ink_norm  = ink_mask.astype(np.float32) / 255.0
    l_blended = (l_enhanced * ink_norm + l * (1 - ink_norm)).astype(np.uint8)

    img = cv2.cvtColor(cv2.merge([l_blended, a, b]), cv2.COLOR_LAB2RGB)

    # ── Step 3: Restrained unsharp mask for blurry documents ───────
    # Only apply if actually blurry. Threshold raised vs photo mode
    # because document grain masquerades as high-frequency signal.
    if blur < 150:
        sigma  = 1.5 if blur < 60 else 0.8
        amount = 1.0 if blur < 60 else 0.6   # conservative — grain amplification risk
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _postprocess_document(pil_img: Image.Image) -> Image.Image:
    """
    Document post-processing.

    After ESRGAN the upscaled document has good detail but may still
    look slightly soft on thin strokes. A final targeted sharpening
    pass lifts the ink crispness without touching the paper background.

    We deliberately avoid the Laplacian kernel here — it was the cause
    of the heavy noise in the previous version by amplifying paper
    texture at 4× resolution.
    """
    img_cv = np.array(pil_img)

    # ── Bilateral filter: smooths paper, preserves ink edges ───────
    # d=9 neighbourhood, sigmaColor=25 (tight — only smooths similar tones),
    # sigmaSpace=25. This step cleans up any residual ESRGAN grain on paper.
    img_cv = cv2.bilateralFilter(img_cv, d=9, sigmaColor=25, sigmaSpace=25)
    img_pil = Image.fromarray(img_cv)

    # ── Unsharp mask: lift text stroke edges ───────────────────────
    # radius=1.0 targets character-level edges at 4× resolution.
    # percent=140 is noticeable but won't introduce haloing.
    img_pil = img_pil.filter(ImageFilter.UnsharpMask(
        radius=1.0, percent=140, threshold=2
    ))

    # ── Contrast: make ink darker relative to paper ────────────────
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.4)

    # ── Sharpness: final crispness boost ──────────────────────────
    img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.9)

    return img_pil


# ──────────────────────────────────────────────────────────────────
# PHOTO PIPELINE
# ──────────────────────────────────────────────────────────────────

def _preprocess_photo(img_rgb: np.ndarray, blur: float) -> np.ndarray:
    """
    Photo pre-processing.

    Gentler than document mode — natural photos have intentional
    bokeh, skin tones, and gradients that aggressive sharpening destroys.
    """
    img = img_rgb.copy()

    # Mild CLAHE to lift shadow detail and normalise exposure
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # Unsharp mask scaled to blur severity
    if blur < 300:
        sigma  = 1.8 if blur < 80 else 1.0
        amount = 1.4 if blur < 80 else 0.9
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _postprocess_photo(pil_img: Image.Image, blur: float) -> Image.Image:
    """
    Photo post-processing.

    Scales sharpening strength with measured blur so a sharp-but-low-res
    photo gets a lighter touch than a genuinely blurry one.
    """
    # blur_t: 0.0 = sharp input, 1.0 = very blurry input
    blur_t = max(0.0, min(1.0, (300 - blur) / 300))

    # Unsharp mask: radius targets pixel-level detail at 4× resolution
    usm_radius  = 1.2
    usm_percent = int(100 + blur_t * 120)   # 100–220 depending on blur
    usm_thresh  = 2

    img = pil_img.filter(ImageFilter.UnsharpMask(
        radius=usm_radius, percent=usm_percent, threshold=usm_thresh
    ))

    # Sharpness: 1.6 base + up to 0.8 extra for blurry inputs → max 2.4
    img = ImageEnhance.Sharpness(img).enhance(1.6 + blur_t * 0.8)

    # Mild contrast boost — lifts perceived detail without clipping
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

        # ── Stage 1: pre-process ─────────────────────────────────
        if is_doc:
            preprocessed = _preprocess_document(img_array, blur)
        else:
            preprocessed = _preprocess_photo(img_array, blur)

        # ── Stage 2: ESRGAN 4× super-resolution ──────────────────
        bgr_in         = preprocessed[:, :, ::-1].copy()
        output_bgr, _  = enhancer.enhance(bgr_in, outscale=4.0)
        output_rgb     = output_bgr[:, :, ::-1].copy()

        # ── Stage 3: post-process ────────────────────────────────
        output_pil = Image.fromarray(output_rgb.astype(np.uint8))
        if is_doc:
            output_final = _postprocess_document(output_pil)
        else:
            output_final = _postprocess_photo(output_pil, blur)

        result = np.array(output_final).astype(np.uint8)
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