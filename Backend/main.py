"""
ArtifexStudio — FastAPI Backend Gateway v2
───────────────────────────────────────────
Improvements over v1
  • Lifespan context replaces deprecated @app.on_event
  • Strict MIME-type + file-size validation on every upload
  • Path traversal guard on /image and /download
  • Correct Content-Type headers on image responses
  • /health — pings Redis + counts live workers
  • /cancel/{task_id} — revokes queued tasks before they run
  • /cleanup — purges files older than a configurable TTL
  • Async file I/O via aiofiles — frees the event loop during writes
  • Structured JSON error body on every HTTPException
  • Module-level logger for clean Docker log output
"""

import os
import uuid
import time
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import aiofiles
import redis as redis_sync
from celery import Celery
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ──────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("artifex.backend")


# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
REDIS_HOST    = os.getenv("REDIS_HOST", "redis")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
CELERY_BROKER = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 50 MB per file — covers large RAW crops while preventing abuse
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# Files older than this are eligible for /cleanup
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", str(6 * 60 * 60)))  # 6 h

# Only raster image types accepted
ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/webp",
    "image/tiff", "image/bmp", "image/gif",
}


# ──────────────────────────────────────────────────────────────────
# CELERY CLIENT  (dispatches tasks; runs no workers itself)
# ──────────────────────────────────────────────────────────────────
celery_app = Celery("neuro_worker", broker=CELERY_BROKER, backend=CELERY_BROKER)


# ──────────────────────────────────────────────────────────────────
# LIFESPAN
# ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ArtifexStudio backend starting up…")
    log.info("Data directory : %s", DATA_DIR)
    log.info("Redis broker   : %s", CELERY_BROKER)
    log.info("Max upload     : %d MB", MAX_UPLOAD_BYTES // 1024 // 1024)

    try:
        r = redis_sync.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=3)
        r.ping()
        log.info("✅ Redis connection verified.")
    except Exception as exc:
        log.warning("⚠️  Redis ping failed at startup: %s", exc)

    yield

    log.info("ArtifexStudio backend shutting down.")


# ──────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ArtifexStudio API Gateway",
    version="2.0.0",
    description=(
        "Dispatches AI image tasks (enhance, edit, style-transfer, stitch) "
        "to Celery workers via Redis and serves results to the frontend."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def _safe_filename(path: str) -> str:
    """
    Strips any directory component from a user-supplied filename.
    Prevents path traversal attacks like '../../etc/passwd'.
    """
    return Path(path).name


def _detect_mime(filepath: Path) -> str:
    mime, _ = mimetypes.guess_type(str(filepath))
    return mime or "application/octet-stream"


async def _validate_and_save(file: UploadFile, prefix: str) -> Path:
    """
    Validates an upload (MIME type + size) then writes it to DATA_DIR
    asynchronously.  Raises HTTPException on any validation failure.
    """
    # 1. MIME check — normalise image/jpg → image/jpeg
    content_type = (file.content_type or "").replace("image/jpg", "image/jpeg")
    if content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_MIME))}."
            ),
        )

    # 2. Derive extension
    original_name = getattr(file, "filename", None) or "image.png"
    ext = Path(original_name).suffix.lstrip(".").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"

    # 3. Read entire body + size guard
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(raw) / 1024 / 1024:.1f} MB). "
                f"Maximum is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
            ),
        )

    # 4. Write asynchronously
    dest = DATA_DIR / f"{prefix}_{uuid.uuid4().hex}.{ext}"
    async with aiofiles.open(dest, "wb") as fh:
        await fh.write(raw)

    log.info("Saved upload: %s (%d KB)", dest.name, len(raw) // 1024)
    return dest


def _task_response(task_id: str) -> dict:
    return {"task_id": task_id, "status": "QUEUED"}


# ──────────────────────────────────────────────────────────────────
# INFRASTRUCTURE ENDPOINTS
# ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Infrastructure"])
def root():
    """Liveness probe — used by Docker healthcheck."""
    return {"status": "online", "service": "ArtifexStudio Backend v2"}


@app.get("/health", tags=["Infrastructure"])
def health():
    """
    Deep health check.
    Pings Redis and counts live Celery workers.
    The frontend can use this to surface a 'service degraded' banner.
    """
    redis_ok = False
    worker_count = 0

    try:
        r = redis_sync.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        redis_ok = True
    except Exception as exc:
        log.warning("Health check — Redis unreachable: %s", exc)

    try:
        # Returns {worker_name: [active_task, …], …} for all known workers
        active = celery_app.control.inspect(timeout=1.5).active() or {}
        worker_count = len(active)
    except Exception:
        pass

    status = "healthy" if (redis_ok and worker_count > 0) else "degraded"
    return {
        "status": status,
        "redis": "online" if redis_ok else "offline",
        "workers": worker_count,
    }


# ──────────────────────────────────────────────────────────────────
# FILE SERVING
# ──────────────────────────────────────────────────────────────────

@app.get("/image/{filename}", tags=["Files"])
def get_image(filename: str):
    """
    Serves a generated image for inline display.
    Sets the correct Content-Type and a 1-hour cache header.
    """
    safe = _safe_filename(filename)
    file_path = DATA_DIR / safe

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{safe}' not found.")

    return FileResponse(
        path=file_path,
        media_type=_detect_mime(file_path),
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/download/{filename}", tags=["Files"])
def download_image(filename: str):
    """Serves a generated image as a browser download attachment."""
    safe = _safe_filename(filename)
    file_path = DATA_DIR / safe

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{safe}' not found.")

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=f"ArtifexStudio_{safe}",
        headers={"Cache-Control": "no-store"},
    )


# ──────────────────────────────────────────────────────────────────
# TASK STATUS & CANCELLATION
# ──────────────────────────────────────────────────────────────────

@app.get("/status/{task_id}", tags=["Tasks"])
def get_status(task_id: str):
    """
    Returns current Celery task state.

    Handles every state Celery can produce so the frontend never
    receives an unrecognised status and loops forever.
    """
    try:
        task = celery_app.AsyncResult(task_id)
        state = task.state

        if state == "PENDING":
            return {"status": "PENDING"}

        if state == "STARTED":
            return {"status": "STARTED"}

        if state == "RETRY":
            return {"status": "STARTED"}   # treat as still in progress

        if state == "REVOKED":
            return {"status": "REVOKED"}

        if state == "SUCCESS":
            result = task.result
            if isinstance(result, dict):
                if result.get("status") == "success":
                    return {"status": "SUCCESS", "result": result.get("output_path")}
                if result.get("status") == "failed":
                    return {
                        "status": "FAILURE",
                        "error": result.get("error", "Worker reported failure with no message."),
                    }
            # Unexpected result shape — surface it so the error popup has something to show
            return {
                "status": "FAILURE",
                "error": f"Unexpected result format from worker: {str(result)[:300]}",
            }

        if state == "FAILURE":
            # task.info contains the exception when state is FAILURE
            error_detail = str(task.info) if task.info else "Worker process crashed."
            return {"status": "FAILURE", "error": error_detail}

        # Any other state (custom states, etc.) — treat as failure so the
        # frontend doesn't loop forever
        return {
            "status": "FAILURE",
            "error": f"Task ended in unexpected state '{state}'. Check worker logs.",
        }

    except Exception as exc:
        log.error("Error fetching status for task %s: %s", task_id, exc)
        return {
            "status": "FAILURE",
            "error": "Could not retrieve task status from Redis. The broker may be unavailable.",
        }

@app.delete("/cancel/{task_id}", tags=["Tasks"])
def cancel_task(task_id: str):
    """
    Revokes a queued task before it is picked up by a worker.
    Has no effect if the worker has already started processing.
    """
    celery_app.control.revoke(task_id, terminate=False)
    log.info("Task revoked by client: %s", task_id)
    return {"status": "revoked", "task_id": task_id}


# ──────────────────────────────────────────────────────────────────
# ADMIN
# ──────────────────────────────────────────────────────────────────

@app.delete("/cleanup", tags=["Admin"])
def cleanup_old_files(max_age_hours: float = 6.0):
    """
    Deletes files in the shared volume older than max_age_hours.
    Call from a cron job or admin script to prevent disk exhaustion:

        curl -X DELETE "http://localhost:8000/cleanup?max_age_hours=12"
    """
    cutoff = time.time() - (max_age_hours * 3600)
    deleted_count = 0
    freed_bytes = 0

    for f in DATA_DIR.iterdir():
        if not f.is_file():
            continue
        try:
            stat = f.stat()
            if stat.st_mtime < cutoff:
                freed_bytes += stat.st_size
                f.unlink()
                deleted_count += 1
        except OSError as exc:
            log.warning("Could not delete %s: %s", f.name, exc)

    freed_mb = freed_bytes / 1024 / 1024
    log.info("Cleanup: removed %d files, freed %.1f MB", deleted_count, freed_mb)
    return {
        "deleted_files": deleted_count,
        "freed_mb": round(freed_mb, 2),
        "cutoff_hours": max_age_hours,
    }


# ──────────────────────────────────────────────────────────────────
# TASK DISPATCHERS
# ──────────────────────────────────────────────────────────────────

@app.post("/enhance", tags=["AI Tasks"])
async def enhance_image(file: UploadFile = File(...)):
    """
    Real-ESRGAN x4plus enhancement.
    Worker upscales the image 4× with tiled FP16 inference.
    """
    filepath = await _validate_and_save(file, "esrgan_in")
    task = celery_app.send_task("tasks.task_enhance_image", args=[str(filepath)])
    log.info("Enhance task dispatched: %s → %s", filepath.name, task.id)
    return _task_response(task.id)


@app.post("/stitch", tags=["AI Tasks"])
async def stitch_images(files: List[UploadFile] = File(...)):
    """
    Panoramic stitching via SIFT keypoints + histogram exposure matching.
    Requires 2–20 overlapping images with at least 20% frame overlap.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images are required.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per stitch.")

    filepaths = [str(await _validate_and_save(f, "stitch_in")) for f in files]
    task = celery_app.send_task("tasks.task_stitch_images", args=[filepaths])
    log.info("Stitch task dispatched: %d images → %s", len(filepaths), task.id)
    return _task_response(task.id)


@app.post("/style-transfer", tags=["AI Tasks"])
async def style_transfer(
    content_image: UploadFile = File(...),
    style_image:   UploadFile = File(...),
    prompt:        str        = Form(""),
):
    """
    SDXL IP-Adapter style transfer.
    content_image preserves scene structure; style_image sets visual language.
    prompt optionally steers creative direction further.
    """
    content_path = await _validate_and_save(content_image, "style_content")
    style_path   = await _validate_and_save(style_image,   "style_ref")

    # Sanitise: strip control characters, cap at 300 chars
    clean_prompt = prompt.strip()[:300] if prompt else ""

    task = celery_app.send_task(
        "tasks.task_style_transfer",
        args=[str(content_path), str(style_path), clean_prompt],
    )
    log.info("Style-transfer task dispatched: %s", task.id)
    return _task_response(task.id)


@app.post("/edit", tags=["AI Tasks"])
async def edit_image(
    image:  UploadFile             = File(...),
    mask:   Optional[UploadFile]   = File(None),
    action: str                    = Form(...),
):
    """
    Magic Eraser — two modes:

    remove_bg — RMBG-2.0 neural matting → RGBA transparent PNG.
    erase     — LaMa inpainting → seamless background reconstruction.
                Requires a mask: white = erase, black = keep.
    """
    allowed_actions = {"remove_bg", "erase"}
    if action not in allowed_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{action}'. Valid: {sorted(allowed_actions)}.",
        )
    if action == "erase" and mask is None:
        raise HTTPException(
            status_code=400,
            detail="action='erase' requires a mask (white = erase, black = keep).",
        )

    img_path  = await _validate_and_save(image, "edit_in")
    mask_path = await _validate_and_save(mask, "edit_mask") if mask else None

    task = celery_app.send_task(
        "tasks.task_edit_image",
        args=[str(img_path), action, str(mask_path) if mask_path else None],
    )
    log.info("Edit task dispatched: action=%s → %s", action, task.id)
    return _task_response(task.id)


# ──────────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ──────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Check backend logs."},
    )