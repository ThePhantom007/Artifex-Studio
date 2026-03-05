import os
import torch
import numpy as np
from PIL import Image
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

# ── Weights path ──────────────────────────────────────────────────
# When model_path is a URL, RealESRGANer internally calls load_file_from_url
# with model_dir hardcoded to {realesrgan_package}/weights/ — a root-owned
# system directory that is not writable at runtime.
# Fix: pre-download to our own writable cache dir, pass the local path.
WEIGHTS_DIR = os.path.join(
    os.getenv("TORCH_HOME", "/app/.cache/torch"), "realesrgan"
)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

WEIGHTS_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

enhancer = None
try:
    # Download only if not already cached
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
        model_path=WEIGHTS_FILE,   # local path — never touches the package dir
        model=model,
        tile=512,
        tile_pad=32,
        pre_pad=0,
        half=True if device == "cuda" else False,
        device=device,
    )
    print("✅ Real-ESRGAN x4plus Engine loaded and ready.")

except Exception as e:
    print(f"❌ Real-ESRGAN Load Failed: {e}")
    enhancer = None


def enhance_image(img_array: np.ndarray) -> np.ndarray | str:
    if enhancer is None:
        return "Error: Real-ESRGAN engine failed to load. Check worker logs."
    try:
        print("🔍 Real-ESRGAN is restoring image quality...")
        img_bgr = img_array[:, :, ::-1].copy()
        output_bgr, _ = enhancer.enhance(img_bgr, outscale=4.0)
        output_rgb = output_bgr[:, :, ::-1].copy()
        print("✨ Enhancement complete.")
        return output_rgb.astype(np.uint8)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return "Error: GPU OOM. Reduce tile size from 512 to 256 in enhancement.py and rebuild."
        return f"Runtime Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Inference Error: {str(e)}"