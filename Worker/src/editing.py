import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

print("🚀 Initializing Magic Editor Engine...")

device = "cuda" if torch.cuda.is_available() else "cpu"
hf_device_id = 0 if torch.cuda.is_available() else -1
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# ── 1. Background Removal (RMBG-2.0) ────────────────────────────
# RMBG-2.0 uses a custom BiRefNet-based architecture that cannot be
# loaded via the generic pipeline() function. Load it directly using
# AutoModelForImageSegmentation as BriaAI documents.
rmbg_model = None
try:
    from transformers import AutoModelForImageSegmentation
    rmbg_model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    rmbg_model.to(device)
    rmbg_model.eval()
    print("✅ RMBG-2.0 Background Removal Engine loaded.")
except Exception as e:
    print(f"❌ RMBG-2.0 Load Failed: {e}")
    rmbg_model = None

# Transform pipeline expected by RMBG-2.0
_RMBG_SIZE = (1024, 1024)
_rmbg_transform = transforms.Compose([
    transforms.Resize(_RMBG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 2. Generative Erase (LaMa) ───────────────────────────────────
lama_model = None
try:
    from simple_lama_inpainting import SimpleLama
    lama_model = SimpleLama()
    print("✅ LaMa Generative Erase Engine loaded.")
except Exception as e:
    print(f"❌ LaMa Load Failed: {e}")
    lama_model = None


def _run_rmbg(pil_img: Image.Image) -> Image.Image:
    """
    Runs RMBG-2.0 inference and returns a PIL RGBA image with
    the background removed (transparent).
    """
    original_size = pil_img.size          # (W, H)

    # Preprocess
    inp = _rmbg_transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.inference_mode():
        preds = rmbg_model(inp)[-1].sigmoid().cpu()

    # preds shape: (1, 1, H, W) — squeeze to (H, W)
    mask = preds[0].squeeze()

    # Convert mask tensor → PIL grayscale, resize back to original
    mask_pil = transforms.ToPILImage()(mask).convert("L")
    mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)

    # Composite: paste mask as alpha onto the original RGB image
    result = pil_img.convert("RGBA")
    result.putalpha(mask_pil)
    return result


def edit_image(
    img_array: np.ndarray,
    action: str = "remove_bg",
    mask_array: np.ndarray = None,
) -> np.ndarray | str:
    """
    remove_bg — RMBG-2.0 neural matting → RGBA transparent PNG.
    erase     — LaMa inpainting → seamless background reconstruction.
    """
    try:
        if action == "remove_bg":
            if rmbg_model is None:
                return "Error: Background removal model failed to load. Check worker logs."

            print("✂️ Extracting subject with RMBG-2.0...")
            pil_img = Image.fromarray(img_array).convert("RGB")
            result_pil = _run_rmbg(pil_img)
            print("✅ Subject extracted.")
            return np.array(result_pil)

        elif action == "erase":
            if lama_model is None:
                return "Error: LaMa erase engine failed to load. Check worker logs."
            if mask_array is None:
                return "Error: A mask image is required for Generative Erase."

            print("🖌️ LaMa Generative Erase in progress...")
            init_image = Image.fromarray(img_array).convert("RGB")
            original_size = init_image.size

            # NEAREST resize prevents anti-aliasing grey fringe on mask edges
            mask_image = Image.fromarray(mask_array).convert("L")
            mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)

            # Cap at 2048px to prevent OOM
            max_dim = 2048
            if max(original_size) > max_dim:
                ratio = max_dim / max(original_size)
                new_size = (int(original_size[0] * ratio),
                            int(original_size[1] * ratio))
                init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)
                mask_image = mask_image.resize(new_size, Image.Resampling.NEAREST)

            result_image = lama_model(init_image, mask_image)

            if result_image.size != original_size:
                result_image = result_image.resize(
                    original_size, Image.Resampling.LANCZOS
                )

            print("✅ Object erased.")
            return np.array(result_image.convert("RGB"))

        else:
            return f"Error: Unknown action '{action}'. Valid: 'remove_bg', 'erase'."

    except Exception as e:
        return f"Unexpected Error: {str(e)}"