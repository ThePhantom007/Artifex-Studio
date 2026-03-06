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


# ─── Blur detector ───────────────────────────────────────────────
def _blur_score(gray: np.ndarray) -> float:
    """
    Returns the Laplacian variance of a grayscale image.
    Lower = blurrier. Empirically, scores below 120 benefit from
    aggressive pre-deblurring; above 120 are already reasonably sharp.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── Document detector ───────────────────────────────────────────
def _is_document(img_rgb: np.ndarray) -> bool:
    """
    Heuristic: if >55% of pixels are very bright (paper/whiteboard)
    treat the image as a document and apply a stronger pipeline.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bright_ratio = np.sum(gray > 200) / gray.size
    return bright_ratio > 0.55


# ─── Pre-processing ───────────────────────────────────────────────
def _preprocess(img_rgb: np.ndarray, is_doc: bool, blur: float) -> np.ndarray:
    """
    Sharpens the input BEFORE feeding it to ESRGAN.

    ESRGAN was trained on specific synthetic degradations. Giving it a
    pre-sharpened image means it treats less of the input as noise and
    reconstructs sharper high-frequency detail.

    Pipeline:
      1. CLAHE  — normalise local contrast so dark text on bright paper
                  isn't washed out during the neural pass
      2. Wiener-style unsharp mask  — aggressive edge recovery for blur > 2px
      3. Laplacian kernel sharpening — thin-stroke enhancement (critical for
                  handwriting and small fonts)
    """
    img = img_rgb.copy()

    # ── 1. CLAHE on luminance channel ──────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clip = 4.0 if is_doc else 2.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # ── 2. Aggressive unsharp mask for blurry inputs ───────────────
    if blur < 200:                          # anything below 200 is noticeably blurry
        sigma  = 2.0 if blur < 80 else 1.2
        amount = 1.8 if blur < 80 else 1.2  # how much edge to add back
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    # ── 3. Laplacian kernel sharpening (document mode only) ────────
    if is_doc:
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0],
        ], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# ─── Post-processing ──────────────────────────────────────────────
def _postprocess(pil_img: Image.Image, is_doc: bool, blur: float) -> Image.Image:
    """
    Final sharpening and contrast pass AFTER ESRGAN inference.

    Values are intentionally higher than typical enhancement apps:
    ESRGAN on a pre-sharpened blurry input needs a strong finishing
    pass to make recovered detail perceptually obvious.
    """
    img = pil_img

    if is_doc:
        # Documents: maximise text crispness and contrast
        # Unsharp mask with tight radius — targets character stroke edges
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(2.2)
        img = ImageEnhance.Contrast(img).enhance(1.35)
    else:
        # Photos: balanced sharpening that avoids haloing
        blur_strength = max(0, min(1, (200 - blur) / 200))   # 0→sharp, 1→very blurry
        usm_pct = int(80 + blur_strength * 100)               # 80–180 depending on blur
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=usm_pct, threshold=3))
        img = ImageEnhance.Sharpness(img).enhance(1.5 + blur_strength * 0.5)

    return img


# ─── Main entry point ─────────────────────────────────────────────
def enhance_image(img_array: np.ndarray) -> np.ndarray | str:
    if enhancer is None:
        return "Error: Real-ESRGAN engine failed to load. Check worker logs."

    try:
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        blur    = _blur_score(gray)
        is_doc  = _is_document(img_array)

        print(f"🔍 Input: {w}×{h}px  |  blur={blur:.0f}  |  mode={'document' if is_doc else 'photo'}")

        # ── Stage 1: pre-process ─────────────────────────────────
        preprocessed = _preprocess(img_array, is_doc, blur)

        # ── Stage 2: ESRGAN super-resolution ─────────────────────
        bgr_in      = preprocessed[:, :, ::-1].copy()
        output_bgr, _ = enhancer.enhance(bgr_in, outscale=4.0)
        output_rgb  = output_bgr[:, :, ::-1].copy()

        # ── Stage 3: post-process ────────────────────────────────
        output_pil   = Image.fromarray(output_rgb.astype(np.uint8))
        output_final = _postprocess(output_pil, is_doc, blur)
        result       = np.array(output_final).astype(np.uint8)

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