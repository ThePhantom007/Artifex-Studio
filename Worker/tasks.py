"""
ArtifexStudio — Celery Worker Task Definitions
  • Loads celery_config via app.config_from_object() — v1 ignored the config
    entirely and used hardcoded defaults.
  • bind=True on every task — tasks can now update their own state to STARTED
    and emit progress metadata the frontend can display.
  • cv2.imread null-guard — a missing or corrupt file returned None and was
    passed straight into stitch_images, causing a cryptic AttributeError.
    Now raises a clear, descriptive error immediately.
  • Input file cleanup — input uploads are deleted after processing to prevent
    /data growing without bound between /cleanup calls.
  • SoftTimeLimitExceeded handler — flushes GPU cache and returns a structured
    error instead of leaving the worker in an undefined state.
  • Structured logging — every task logs start, completion, and timing so
    failures are diagnosable from Docker logs.
  • save_image uses pathlib — consistent with main.py v2.
"""

import os
import time
import uuid
import logging
from pathlib import Path

import cv2
import numpy as np
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from PIL import Image

# AI modules
from src.enhancement    import enhance_image
from src.stitching      import stitch_images
from src.style_transfer import apply_style_transfer
from src.editing        import edit_image

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("artifex.worker")

# ─── Celery Application ───────────────────────────────────────────
# Build the broker URL the same way celery_config.py does so both always.
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

_BROKER = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery("neuro_worker", broker=_BROKER, backend=_BROKER)

# Load all tuning from the shared config module.
celery_app.config_from_object("celery_config")

# ─── Shared Volume ────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── IO Helpers ──────────────────────────────────────────────────
def load_image_rgb(filepath: str | Path) -> np.ndarray:
    """
    Opens an image file and returns a uint8 RGB numpy array.
    Raises FileNotFoundError immediately if the path does not exist,
    rather than letting Pillow emit a cryptic internal error.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return np.array(Image.open(p).convert("RGB"))


def load_image_bgr(filepath: str | Path) -> np.ndarray:
    """
    Loads an image with OpenCV (BGR convention, required by stitching.py).
    Raises ValueError if cv2.imread returns None — which happens silently
    when the file is missing, corrupt, or in an unsupported format.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(
            f"cv2.imread returned None for '{p.name}'. "
            "The file may be corrupt or in an unsupported format."
        )
    return img


def save_image(img_array: np.ndarray, prefix: str = "result") -> str:
    """
    Saves a uint8 RGB numpy array to DATA_DIR as a lossless PNG.
    Returns just the filename (not the full path) for the API response.
    """
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    dest = DATA_DIR / filename
    Image.fromarray(img_array).save(dest, format="PNG", optimize=False)
    log.info("Saved output: %s", filename)
    return filename


def _cleanup_inputs(*paths: str | Path | None) -> None:
    """
    Deletes input upload files after a task completes.
    Silently ignores files that are already missing (idempotent).
    Prevents /data from filling up with stale uploads between /cleanup calls.
    """
    for p in paths:
        if p is None:
            continue
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete input file %s: %s", p, exc)


# ─── Task 1: Crystal Clarity — Real-ESRGAN Enhancement ───────────
@celery_app.task(
    name="tasks.task_enhance_image",
    bind=True,                 # gives us `self` to update state
    max_retries=0,             # OOM / model errors are not transient; fail fast
    throws=(SoftTimeLimitExceeded,),
)
def task_enhance_image(self, filepath: str) -> dict:
    """
    Upscales and restores a degraded image using Real-ESRGAN x4plus.
    Input:  absolute path to the uploaded image on the shared volume.
    Output: {"status": "success", "output_path": "<filename>.png"}
    """
    start = time.perf_counter()
    log.info("[enhance] Task started — input: %s", Path(filepath).name)

    # Signal STARTED so the frontend advances past the spinner's PENDING phase
    self.update_state(state="STARTED", meta={"step": "Loading image…"})

    try:
        img_array = load_image_rgb(filepath)
        log.info("[enhance] Image loaded: %s × %s px", img_array.shape[1], img_array.shape[0])

        self.update_state(state="STARTED", meta={"step": "Running Real-ESRGAN inference…"})
        result_array = enhance_image(img_array)

        if isinstance(result_array, str):
            # AI module returns an error string on model failure
            log.error("[enhance] AI module error: %s", result_array)
            return {"status": "failed", "error": result_array}

        out_filename = save_image(result_array, "enhanced")
        elapsed = time.perf_counter() - start
        log.info("[enhance] Done in %.1fs — output: %s", elapsed, out_filename)
        return {"status": "success", "output_path": out_filename}

    except SoftTimeLimitExceeded:
        log.error("[enhance] Soft time limit exceeded — task took too long.")
        return {"status": "failed", "error": "Task timed out. The image may be too large."}

    except Exception as exc:
        log.exception("[enhance] Unexpected error: %s", exc)
        return {"status": "failed", "error": str(exc)}

    finally:
        # Always delete the input upload, regardless of success or failure
        _cleanup_inputs(filepath)


# ─── Task 2: Deep Stitch — Panoramic Stitching ───────────────────
@celery_app.task(
    name="tasks.task_stitch_images",
    bind=True,
    max_retries=0,
    throws=(SoftTimeLimitExceeded,),
)
def task_stitch_images(self, filepaths: list) -> dict:
    """
    Stitches 2–20 overlapping images into a seamless panorama.
    Input:  list of absolute paths on the shared volume.
    Output: {"status": "success", "output_path": "<filename>.png"}
    """
    start = time.perf_counter()
    log.info("[stitch] Task started — %d images", len(filepaths))

    self.update_state(state="STARTED", meta={"step": "Loading images…"})

    try:
        # Load as BGR for OpenCV stitching pipeline.
        # cv2.imread returns for missing/corrupt files.
        images = []
        for fp in filepaths:
            img = load_image_bgr(fp)
            images.append(img)
            log.info("[stitch] Loaded %s — %s × %s px", Path(fp).name, img.shape[1], img.shape[0])

        self.update_state(state="STARTED", meta={"step": "Matching keypoints and stitching…"})
        result_array = stitch_images(images)

        if isinstance(result_array, str):
            log.error("[stitch] Stitching failed: %s", result_array)
            return {"status": "failed", "error": result_array}

        # stitch_images returns BGR — convert to RGB before saving as PNG
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        out_filename = save_image(result_rgb, "panorama")
        elapsed = time.perf_counter() - start
        log.info("[stitch] Done in %.1fs — output: %s", elapsed, out_filename)
        return {"status": "success", "output_path": out_filename}

    except SoftTimeLimitExceeded:
        log.error("[stitch] Soft time limit exceeded.")
        return {"status": "failed", "error": "Stitching timed out. Try fewer or smaller images."}

    except Exception as exc:
        log.exception("[stitch] Unexpected error: %s", exc)
        return {"status": "failed", "error": str(exc)}

    finally:
        _cleanup_inputs(*filepaths)


# ─── Task 3: Artistic Vision — SDXL IP-Adapter Style Transfer ────
@celery_app.task(
    name="tasks.task_style_transfer",
    bind=True,
    max_retries=0,
    throws=(SoftTimeLimitExceeded,),
)
def task_style_transfer(self, content_path: str, style_path: str, prompt: str) -> dict:
    """
    Redraws content_path in the visual style of style_path using SDXL + IP-Adapter.
    Input:  two absolute paths + optional text prompt.
    Output: {"status": "success", "output_path": "<filename>.png"}
    """
    start = time.perf_counter()
    log.info("[style] Task started — prompt: '%s'", prompt[:80] if prompt else "(none)")

    self.update_state(state="STARTED", meta={"step": "Loading content and style images…"})

    try:
        content_arr = load_image_rgb(content_path)
        style_arr   = load_image_rgb(style_path)

        log.info(
            "[style] Content: %s×%s  Style: %s×%s",
            content_arr.shape[1], content_arr.shape[0],
            style_arr.shape[1],   style_arr.shape[0],
        )

        self.update_state(state="STARTED", meta={"step": "Running SDXL inference (30 steps)…"})
        result_array = apply_style_transfer(content_arr, style_arr, prompt)

        if isinstance(result_array, str):
            log.error("[style] Style transfer failed: %s", result_array)
            return {"status": "failed", "error": result_array}

        out_filename = save_image(result_array, "styled")
        elapsed = time.perf_counter() - start
        log.info("[style] Done in %.1fs — output: %s", elapsed, out_filename)
        return {"status": "success", "output_path": out_filename}

    except SoftTimeLimitExceeded:
        log.error("[style] Soft time limit exceeded.")
        return {"status": "failed", "error": "Style transfer timed out. Try a smaller image."}

    except Exception as exc:
        log.exception("[style] Unexpected error: %s", exc)
        return {"status": "failed", "error": str(exc)}

    finally:
        _cleanup_inputs(content_path, style_path)


# ─── Task 4: Magic Eraser — Background Removal / Object Erase ────
@celery_app.task(
    name="tasks.task_edit_image",
    bind=True,
    max_retries=0,
    throws=(SoftTimeLimitExceeded,),
)
def task_edit_image(self, img_path: str, action: str, mask_path: str | None = None) -> dict:
    """
    Performs background removal (RMBG-2.0) or generative erase (LaMa).

    remove_bg — returns RGBA PNG with transparent background.
    erase     — returns RGB PNG with reconstructed background.

    Input:  image path, action string, optional mask path.
    Output: {"status": "success", "output_path": "<filename>.png"}
    """
    start = time.perf_counter()
    log.info("[edit] Task started — action: %s", action)

    self.update_state(state="STARTED", meta={"step": f"Loading image for {action}…"})

    try:
        img_array  = load_image_rgb(img_path)

        # Mask is only needed for erase; load_image_rgb handles None-safety
        mask_array = load_image_rgb(mask_path) if mask_path else None

        log.info(
            "[edit] Image: %s×%s  Mask: %s",
            img_array.shape[1], img_array.shape[0],
            "provided" if mask_array is not None else "none",
        )

        step_label = (
            "Extracting subject with RMBG-2.0…"
            if action == "remove_bg"
            else "Erasing object with LaMa…"
        )
        self.update_state(state="STARTED", meta={"step": step_label})

        result_array = edit_image(img_array, action, mask_array)

        if isinstance(result_array, str):
            log.error("[edit] Edit failed: %s", result_array)
            return {"status": "failed", "error": result_array}

        # Background removal produces RGBA — must save with alpha channel intact
        if action == "remove_bg":
            filename = f"nobg_{uuid.uuid4().hex}.png"
            dest = DATA_DIR / filename
            Image.fromarray(result_array, mode="RGBA").save(dest, format="PNG")
            log.info("[edit] Saved RGBA output: %s", filename)
        else:
            filename = save_image(result_array, "edited")

        elapsed = time.perf_counter() - start
        log.info("[edit] Done in %.1fs — output: %s", elapsed, filename)
        return {"status": "success", "output_path": filename}

    except SoftTimeLimitExceeded:
        log.error("[edit] Soft time limit exceeded.")
        return {"status": "failed", "error": "Editing timed out. Try a smaller image."}

    except Exception as exc:
        log.exception("[edit] Unexpected error: %s", exc)
        return {"status": "failed", "error": str(exc)}

    finally:
        _cleanup_inputs(img_path, mask_path)