"""
ArtifexStudio — Modal deployment

Replaces the Celery + Redis + always-on-GPU-worker architecture with:
  • 3 GPU functions (enhance, edit, style-transfer) that scale to zero
  • 1 CPU function (stitch) — no GPU needed, matches the original code
  • 1 lightweight FastAPI app, served by Modal as an ASGI app, that
    replaces Backend/main.py's dispatch logic (Celery -> Modal spawn/poll)

Deploy with:
    modal deploy modal_app/app.py

First deploy will build two images (a few minutes). After that, redeploys
that only touch backend/app code are fast; GPU image rebuilds only happen
when you change worker/src/*.py or the pinned dependencies below.
"""

import io
import time
import uuid
import mimetypes
from pathlib import Path
from typing import List, Optional

import modal

# ──────────────────────────────────────────────────────────────────
# APP, VOLUMES, SECRET
# ──────────────────────────────────────────────────────────────────
app = modal.App("artifex-studio")

# Persists downloaded model weights (ESRGAN, RMBG-2.0, LaMa, SDXL, IP-Adapter)
# across cold starts. Without this, every cold start re-downloads SDXL (~6.5GB).
model_cache = modal.Volume.from_name("artifex-model-cache", create_if_missing=True)

# Persists generated output images so /image and /download can serve them
# back after the GPU/CPU function that created them has already exited.
data_volume = modal.Volume.from_name("artifex-data", create_if_missing=True)

# Create this once via:
#   modal secret create artifex-hf-secret HUGGING_FACE_HUB_TOKEN=hf_your_token_here
hf_secret = modal.Secret.from_name("artifex-hf-secret")

CACHE_PATH = "/cache"
DATA_PATH = "/data"
VOLUME_CONFIG = {CACHE_PATH: model_cache, DATA_PATH: data_volume}

THIS_DIR = Path(__file__).parent
WORKER_SRC = THIS_DIR.parent / "Worker" / "src"

ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/webp",
    "image/tiff", "image/bmp", "image/gif",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB, same as the original backend


# ──────────────────────────────────────────────────────────────────
# IMAGES
# ──────────────────────────────────────────────────────────────────
# basicsr (and, in some versions, facexlib) import
# `torchvision.transforms.functional_tensor`, which newer torchvision
# releases removed. patch_torchvision_compat.py locates the installed
# package by its actual import path (not a filesystem search)
_PATCH_SCRIPT = THIS_DIR / "patch_torchvision_compat.py"

# GPU image — mirrors Worker/Dockerfile + Worker/requirements.txt.
# Uses stock pip torch wheels (no nvidia/cuda base image needed on Modal —
# the pip CUDA wheels bundle their own runtime libs).
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "git", "curl",
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "opencv-python-headless", "numpy<2.0", "pillow", "scikit-image",
        "diffusers", "transformers", "accelerate", "safetensors", "scipy", "ftfy",
        "basicsr", "realesrgan", "simple-lama-inpainting", "timm", "kornia",
    )
    .add_local_file(str(_PATCH_SCRIPT), remote_path="/root/patch_torchvision_compat.py", copy=True)
    .run_commands("python3 /root/patch_torchvision_compat.py")
    .env({
        "HF_HOME": f"{CACHE_PATH}/huggingface",
        "TORCH_HOME": f"{CACHE_PATH}/torch",
        "TRANSFORMERS_CACHE": f"{CACHE_PATH}/huggingface",
        "PYTHONPATH": "/root",
    })
    .add_local_dir(str(WORKER_SRC), remote_path="/root/src")
)

# CPU image for Deep Stitch — no torch, no GPU. Matches the fact that
# stitching.py is pure OpenCV/SIFT and never touches CUDA.
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("opencv-python-headless", "numpy<2.0", "pillow", "scikit-image")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(WORKER_SRC), remote_path="/root/src")
)

backend_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi", "python-multipart", "aiofiles", "pillow"
)


# ──────────────────────────────────────────────────────────────────
# HELPERS (shared by the GPU/CPU functions)
# ──────────────────────────────────────────────────────────────────
def _save_output(np_array, prefix: str, mode: str = "RGB") -> str:
    """Writes a numpy image array to the data volume, commits, returns filename."""
    from PIL import Image as PILImage

    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    dest = Path(DATA_PATH) / filename
    PILImage.fromarray(np_array, mode=mode if mode == "RGBA" else None).save(
        dest, format="PNG", optimize=False
    )
    data_volume.commit()
    return filename


# ──────────────────────────────────────────────────────────────────
# GPU FUNCTION 1 — Crystal Clarity (Real-ESRGAN)
# ──────────────────────────────────────────────────────────────────
@app.function(
    image=gpu_image,
    gpu="T4",
    volumes=VOLUME_CONFIG,
    timeout=300,
    scaledown_window=300,  # stays warm 5 min after last call — smooths out demo bursts
)
def run_enhance(image_bytes: bytes) -> dict:
    import numpy as np
    from PIL import Image as PILImage
    from src.enhancement import enhance_image

    img = np.array(PILImage.open(io.BytesIO(image_bytes)).convert("RGB"))
    result = enhance_image(img)

    if isinstance(result, str):
        return {"status": "failed", "error": result}

    filename = _save_output(result, "enhanced")
    return {"status": "success", "output_path": filename}


# ──────────────────────────────────────────────────────────────────
# GPU FUNCTION 2 — Magic Eraser (RMBG-2.0 + LaMa)
# ──────────────────────────────────────────────────────────────────
@app.function(
    image=gpu_image,
    gpu="T4",
    volumes=VOLUME_CONFIG,
    secrets=[hf_secret],
    timeout=300,
    scaledown_window=300,
)
def run_edit(image_bytes: bytes, action: str, mask_bytes: Optional[bytes] = None) -> dict:
    import numpy as np
    from PIL import Image as PILImage
    from src.editing import edit_image

    img_array = np.array(PILImage.open(io.BytesIO(image_bytes)).convert("RGB"))
    mask_array = (
        np.array(PILImage.open(io.BytesIO(mask_bytes)).convert("RGB"))
        if mask_bytes else None
    )

    result = edit_image(img_array, action, mask_array)

    if isinstance(result, str):
        return {"status": "failed", "error": result}

    if action == "remove_bg":
        filename = _save_output(result, "nobg", mode="RGBA")
    else:
        filename = _save_output(result, "edited")

    return {"status": "success", "output_path": filename}


# ──────────────────────────────────────────────────────────────────
# GPU FUNCTION 3 — Artistic Vision (SDXL + IP-Adapter)
# ──────────────────────────────────────────────────────────────────
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes=VOLUME_CONFIG,
    secrets=[hf_secret],
    timeout=600,
    scaledown_window=300,
)
def run_style_transfer(content_bytes: bytes, style_bytes: bytes, prompt: str = "") -> dict:
    import numpy as np
    from PIL import Image as PILImage
    from src.style_transfer import apply_style_transfer

    content_arr = np.array(PILImage.open(io.BytesIO(content_bytes)).convert("RGB"))
    style_arr = np.array(PILImage.open(io.BytesIO(style_bytes)).convert("RGB"))

    result = apply_style_transfer(content_arr, style_arr, prompt)

    if isinstance(result, str):
        return {"status": "failed", "error": result}

    filename = _save_output(result, "styled")
    return {"status": "success", "output_path": filename}


# ──────────────────────────────────────────────────────────────────
# CPU FUNCTION — Deep Stitch (OpenCV, no GPU — matches the original code)
# ──────────────────────────────────────────────────────────────────
@app.function(
    image=cpu_image,
    volumes=VOLUME_CONFIG,
    timeout=180,
)
def run_stitch(images_bytes: List[bytes]) -> dict:
    import cv2
    import numpy as np
    from src.stitching import stitch_images

    images = []
    for b in images_bytes:
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "failed", "error": "One of the uploaded files could not be decoded as an image."}
        images.append(img)

    result = stitch_images(images)

    if isinstance(result, str):
        return {"status": "failed", "error": result}

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    filename = _save_output(result_rgb, "panorama")
    return {"status": "success", "output_path": filename}


# ──────────────────────────────────────────────────────────────────
# BACKEND — FastAPI app served by Modal
# ──────────────────────────────────────────────────────────────────
@app.function(
    image=backend_image,
    volumes={DATA_PATH: data_volume},
    scaledown_window=60,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, JSONResponse

    web_app = FastAPI(title="ArtifexStudio API Gateway (Modal)", version="3.0.0")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _safe_filename(name: str) -> str:
        return Path(name).name

    def _detect_mime(filepath: Path) -> str:
        mime, _ = mimetypes.guess_type(str(filepath))
        return mime or "application/octet-stream"

    async def _read_and_validate(file: UploadFile) -> bytes:
        content_type = (file.content_type or "").replace("image/jpg", "image/jpeg")
        if content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.content_type}'. "
                       f"Accepted: {', '.join(sorted(ALLOWED_MIME))}.",
            )
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(raw) / 1024 / 1024:.1f} MB). "
                       f"Maximum is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
            )
        return raw

    @web_app.get("/", tags=["Infrastructure"])
    def root():
        return {"status": "online", "service": "ArtifexStudio Backend (Modal)"}

    @web_app.get("/health", tags=["Infrastructure"])
    def health():
        # Modal manages worker scaling itself — there's no persistent worker
        # count to report the way Celery had one, so this just confirms
        # the backend function itself is reachable.
        return {"status": "healthy", "platform": "modal"}

    @web_app.get("/image/{filename}", tags=["Files"])
    def get_image(filename: str):
        data_volume.reload()
        safe = _safe_filename(filename)
        file_path = Path(DATA_PATH) / safe
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found.")
        return Response(content=file_path.read_bytes(), media_type=_detect_mime(file_path))

    @web_app.get("/download/{filename}", tags=["Files"])
    def download_image(filename: str):
        data_volume.reload()
        safe = _safe_filename(filename)
        file_path = Path(DATA_PATH) / safe
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found.")
        return Response(
            content=file_path.read_bytes(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="ArtifexStudio_{safe}"',
                "Cache-Control": "no-store",
            },
        )

    @web_app.get("/status/{task_id}", tags=["Tasks"])
    def get_status(task_id: str):
        try:
            fc = modal.FunctionCall.from_id(task_id)
        except Exception as exc:
            return {"status": "FAILURE", "error": f"Unknown task id: {exc}"}

        try:
            result = fc.get(timeout=0)
        except TimeoutError:
            return {"status": "PENDING"}
        except Exception as exc:
            return {"status": "FAILURE", "error": str(exc)[:300]}

        if isinstance(result, dict):
            if result.get("status") == "success":
                return {"status": "SUCCESS", "result": result.get("output_path")}
            if result.get("status") == "failed":
                return {"status": "FAILURE", "error": result.get("error", "Worker reported failure.")}

        return {"status": "FAILURE", "error": f"Unexpected result format: {str(result)[:300]}"}

    @web_app.delete("/cancel/{task_id}", tags=["Tasks"])
    def cancel_task(task_id: str):
        try:
            fc = modal.FunctionCall.from_id(task_id)
            fc.cancel()
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
        return {"status": "revoked", "task_id": task_id}

    @web_app.delete("/cleanup", tags=["Admin"])
    def cleanup_old_files(max_age_hours: float = 6.0):
        data_volume.reload()
        cutoff = time.time() - (max_age_hours * 3600)
        deleted_count = 0
        freed_bytes = 0
        for f in Path(DATA_PATH).iterdir():
            if not f.is_file():
                continue
            stat = f.stat()
            if stat.st_mtime < cutoff:
                freed_bytes += stat.st_size
                f.unlink()
                deleted_count += 1
        data_volume.commit()
        return {
            "deleted_files": deleted_count,
            "freed_mb": round(freed_bytes / 1024 / 1024, 2),
            "cutoff_hours": max_age_hours,
        }

    @web_app.post("/enhance", tags=["AI Tasks"])
    async def enhance_endpoint(file: UploadFile = File(...)):
        raw = await _read_and_validate(file)
        fc = run_enhance.spawn(raw)
        return {"task_id": fc.object_id, "status": "QUEUED"}

    @web_app.post("/stitch", tags=["AI Tasks"])
    async def stitch_endpoint(files: List[UploadFile] = File(...)):
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images are required.")
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 images per stitch.")
        images_bytes = [await _read_and_validate(f) for f in files]
        fc = run_stitch.spawn(images_bytes)
        return {"task_id": fc.object_id, "status": "QUEUED"}

    @web_app.post("/style-transfer", tags=["AI Tasks"])
    async def style_transfer_endpoint(
        content_image: UploadFile = File(...),
        style_image: UploadFile = File(...),
        prompt: str = Form(""),
    ):
        content_raw = await _read_and_validate(content_image)
        style_raw = await _read_and_validate(style_image)
        clean_prompt = prompt.strip()[:300] if prompt else ""
        fc = run_style_transfer.spawn(content_raw, style_raw, clean_prompt)
        return {"task_id": fc.object_id, "status": "QUEUED"}

    @web_app.post("/edit", tags=["AI Tasks"])
    async def edit_endpoint(
        image: UploadFile = File(...),
        mask: Optional[UploadFile] = File(None),
        action: str = Form(...),
    ):
        allowed_actions = {"remove_bg", "erase"}
        if action not in allowed_actions:
            raise HTTPException(status_code=400, detail=f"Unknown action '{action}'. Valid: {sorted(allowed_actions)}.")
        if action == "erase" and mask is None:
            raise HTTPException(status_code=400, detail="action='erase' requires a mask.")

        img_raw = await _read_and_validate(image)
        mask_raw = await _read_and_validate(mask) if mask else None
        fc = run_edit.spawn(img_raw, action, mask_raw)
        return {"task_id": fc.object_id, "status": "QUEUED"}

    @web_app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})

    return web_app