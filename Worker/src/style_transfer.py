import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from transformers import CLIPVisionModelWithProjection
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = None

try:
    print("👁️ Loading CLIP Vision Encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print("🧠 Loading Stable Diffusion XL...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        base_model,
        image_encoder=image_encoder,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # FIX: variant="fp16" only exists for CUDA builds — passing it on CPU crashes the load.
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )

    print("🔌 Attaching IP-Adapter weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.bin"
    )

    # enable_model_cpu_offload handles all device placement internally.
    # Do NOT also call .to(device) — combining both causes a device conflict.
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    print("✅ Style Transfer Pipeline ready.")

except Exception as e:
    print(f"❌ CRITICAL PIPELINE FAILURE: {e}")
    pipe = None


def _snap_to_multiple(value: int, multiple: int = 64) -> int:
    """
    SDXL's VAE encoder and decoder operate on 64px grid patches.
    Dimensions that aren't exact multiples of 64 cause misaligned feature maps
    in the decoder, producing subtle grid-pattern artefacts in the output.
    """
    return max(multiple, (value // multiple) * multiple)


def _resize_for_sdxl(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resizes preserving aspect ratio, then snaps both dimensions to multiples of 64."""
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        w = int(w * ratio)
        h = int(h * ratio)
    w = _snap_to_multiple(w)
    h = _snap_to_multiple(h)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def apply_style_transfer(
    content_arr: np.ndarray,
    style_arr: np.ndarray,
    prompt: str = ""
) -> np.ndarray | str:
    """
    Redraws the content image in the visual style of the style image using IP-Adapter + SDXL.

    The IP-Adapter injects style image features directly into SDXL's cross-attention
    layers — giving finer, more faithful style transfer than neural gram-matrix methods.

    Args:
        content_arr: The scene/composition to preserve (RGB numpy array).
        style_arr:   The artwork or photo whose visual style to apply (RGB numpy array).
        prompt:      Optional text to steer style direction, e.g. "oil painting, warm tones".
                     Leave empty to let the style image speak for itself.
    """
    if pipe is None:
        return "Error: Style Transfer pipeline failed to load. Check worker logs."

    try:
        # --- 1. Prepare and resize inputs ---
        content_image = _resize_for_sdxl(Image.fromarray(content_arr).convert("RGB"))
        style_image = _resize_for_sdxl(Image.fromarray(style_arr).convert("RGB"))

        # --- 2. VRAM flush ---
        # SDXL leaves residual tensors in VRAM between runs. On 8GB cards this
        # pushes the next inference over the limit mid-denoise. Flush first.
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- 3. IP-Adapter scale ---
        # 1.0 (original) = model ignores text prompt entirely, style image dominates.
        # 0.7 = strong style influence while preserving scene structure and respecting prompt.
        # 0.5 = subtle style, strong structure preservation.
        pipe.set_ip_adapter_scale(0.7)

        # --- 4. Prompts ---
        final_prompt = prompt.strip() if prompt.strip() else "masterpiece, best quality, highly detailed"
        # Negative prompt suppresses common SDXL failure modes: blurring, deformation, grid artefacts.
        negative_prompt = "blurry, low quality, distorted, ugly, artefacts, deformed, watermark"

        # --- 5. Inference ---
        print("🎨 Injecting style into cross-attention layers...")
        with torch.inference_mode():
            result = pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                image=content_image,
                ip_adapter_image=style_image,
                # 0.60: 60% AI regeneration preserves recognisable scene composition
                # while still applying style strongly. Raise toward 0.80 for more
                # artistic transformation; lower toward 0.45 for subtle texture overlay.
                strength=0.60,
                guidance_scale=7.5,
                # 30 steps: +5 over original. Negligible time cost at SDXL scale,
                # noticeably cleaner fine detail and colour coherence.
                num_inference_steps=30
            ).images[0]

        print("✨ Style Transfer complete.")
        return np.array(result)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return (
                "Error: GPU ran out of memory during inference. "
                "Restart the worker container to fully clear VRAM, then retry."
            )
        return f"Runtime Error: {str(e)}"

    except Exception as e:
        return f"Unexpected Error: {str(e)}"