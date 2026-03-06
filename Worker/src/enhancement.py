import os
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

WEIGHTS_DIR = os.path.join(
    os.getenv("TORCH_HOME", "/app/.cache/torch"), "realesrgan"
)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

WEIGHTS_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

enhancer = None
try:
    if not os.path.exists(WEIGHTS_FILE):
        print("⬇️  Downloading Real-ESRGAN weights (~67 MB)...")
        load_file_from_url(
            url=WEIGHTS_URL,
            model_dir=WEIGHTS_DIR,
            progress=True,
            file_name="RealESRGAN_x4plus.pth"
        )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )

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


def _sharpen_result(pil_img: Image.Image) -> Image.Image:
    """
    Applies a two-stage sharpening pass after ESRGAN inference.

    Stage 1 — Unsharp mask: enhances fine edges and micro-texture
    without introducing haloing. Radius 1.5 targets the pixel-level
    detail that ESRGAN reconstructs.

    Stage 2 — Sharpness enhancer: boosts the overall perceived crispness
    by a moderate factor (1.45). Values above ~1.6 introduce artefacts.
    """
    # Unsharp mask (radius, percent strength, threshold)
    sharpened = pil_img.filter(ImageFilter.UnsharpMask(
        radius=1.5,
        percent=60,
        threshold=3,
    ))
    # Global sharpness boost
    sharpened = ImageEnhance.Sharpness(sharpened).enhance(1.45)
    return sharpened


def enhance_image(img_array: np.ndarray) -> np.ndarray | str:
    if enhancer is None:
        return "Error: Real-ESRGAN engine failed to load. Check worker logs."
    try:
        h, w = img_array.shape[:2]
        print(f"🔍 Enhancing {w}×{h}px image with Real-ESRGAN + sharpening...")

        img_bgr = img_array[:, :, ::-1].copy()
        output_bgr, _ = enhancer.enhance(img_bgr, outscale=4.0)
        output_rgb = output_bgr[:, :, ::-1].copy()

        # Post-process: sharpening pass makes the 4× detail visible
        output_pil    = Image.fromarray(output_rgb.astype(np.uint8))
        output_sharp  = _sharpen_result(output_pil)
        output_final  = np.array(output_sharp)

        oh, ow = output_final.shape[:2]
        print(f"✨ Enhancement complete: {w}×{h} → {ow}×{oh}px")
        return output_final.astype(np.uint8)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return "Error: GPU OOM — reduce tile size from 512 to 256 in enhancement.py."
        return f"Runtime Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"