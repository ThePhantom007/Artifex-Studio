FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

RUN python -m pip install --index-url https://pypi.org/simple/ --upgrade pip

# ── PyTorch with CUDA 12.8 ───────────────────────────────────────
RUN pip install \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

# ── AI dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements.txt

# ── Patch basicsr / facexlib torchvision compatibility ───────────
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py && \
    find /usr/local/lib/python3.11/dist-packages/facexlib \
         -name "*.py" \
         -exec sed -i \
         's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
         {} + 2>/dev/null || true

# ── Pre-create cache directories inside the container layer ──────
# These sit in /app/.cache which is the container's own writable
# filesystem — no volume mount, no ownership conflicts.
RUN mkdir -p \
    /app/.cache/huggingface/hub \
    /app/.cache/torch/realesrgan \
    /app/.cache/torch/hub/checkpoints

# ── Application code ─────────────────────────────────────────────
COPY . .

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD celery -A tasks inspect ping --timeout=10 || exit 1

CMD ["celery", "-A", "tasks", "worker", \
     "--pool=solo", \
     "--concurrency=1", \
     "--queues=enhance,stitch,style,edit,default", \
     "--loglevel=info"]