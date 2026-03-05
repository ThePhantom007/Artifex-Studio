# ArtifexStudio 🎨

A self-hosted, GPU-accelerated AI image processing suite. Four professional-grade
tools — image enhancement, background removal, style transfer, and panoramic
stitching — served through a polished web interface and powered by your own
RTX GPU.

ArtifexStudio Interface
<img width="1918" height="995" alt="Screenshot 2026-03-05 224451" src="https://github.com/user-attachments/assets/5052ff30-a5ba-484a-8d00-3e7159ee36c7" />

---

## Features

### 🔬 Crystal Clarity — Image Enhancement
Rescue blurry, noisy, or heavily compressed photos. Real-ESRGAN reconstructs
lost texture and detail at 4× native resolution using 23 residual blocks of
deep learning inference.

![Crystal Clarity Demo](<img width="1918" height="995" alt="image" src="https://github.com/user-attachments/assets/f97c80af-bba6-4fbd-8713-02d68f9a4283" />
)

---

### ✂️ Magic Eraser — Background Removal & Object Erase
Two AI systems in one. RMBG-2.0 performs neural matting with pixel-perfect
edges down to individual hair strands. LaMa reconstructs the background
seamlessly as if the object never existed.

![Magic Eraser Demo](<img width="1918" height="993" alt="image" src="https://github.com/user-attachments/assets/7eb52f41-4a3a-4eda-b62d-0b95d4c25d69" />
)

---

### 🎨 Artistic Vision — Style Transfer
IP-Adapter injects the visual language of any reference artwork directly into
SDXL's cross-attention layers. Upload a photo and a painting — the AI repaints
your scene in that style while preserving your original composition.

![Artistic Vision Demo](<img width="1918" height="995" alt="image" src="https://github.com/user-attachments/assets/62de8d75-9f75-4439-9b15-b827265e8437" />
)

---

### 🌅 Deep Stitch — Panoramic Stitching
SIFT keypoint detection, homography warping, histogram exposure matching, and
multiband seam blending — fused into a single drag-and-drop tool that produces
seamless widescreen panoramas with automatic black-border cropping.

![Deep Stitch Demo](<img width="1918" height="992" alt="image" src="https://github.com/user-attachments/assets/b0145e14-ab6e-4832-8be8-f89b16e995ea" />
)

---

| Tool | Model | What it does |
|---|---|---|
| **Crystal Clarity** | Real-ESRGAN x4plus | 4× upscaling and restoration |
| **Magic Eraser** | RMBG-2.0 + LaMa | Background removal and object erase |
| **Artistic Vision** | SDXL + IP-Adapter | Reference-guided style transfer |
| **Deep Stitch** | OpenCV + histogram matching | Seamless panorama stitching |
---

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                    Docker Network                    │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │  Nginx   │    │ FastAPI  │    │ Celery Worker │  │
│  │ :80      │───▶│ :8000    │───▶│ (GPU)         │  │
│  │ frontend │    │ backend  │    │               │  │
│  └──────────┘    └────┬─────┘    └───────┬───────┘  │
│                       │                  │          │
│                  ┌────▼──────────────────▼────┐     │
│                  │         Redis :6379         │     │
│                  │   (broker + result store)   │     │
│                  └─────────────────────────────┘     │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │         /data  (named Docker volume)          │   │
│  │   upload staging · generated outputs          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

- **Frontend** — Nginx serving a single-page HTML/CSS/JS app
- **Backend** — FastAPI gateway that validates uploads, dispatches Celery tasks, and serves results
- **Worker** — Celery worker with full CUDA access running all AI inference
- **Redis** — Message broker (task queue) and result backend

---

## Prerequisites

### Hardware
- NVIDIA GPU with at least 6 GB VRAM (8 GB recommended for SDXL style transfer)
- Tested on RTX 5060 Laptop GPU (Blackwell, 8 GB)

### Software
- Windows 11 with WSL2 enabled, or Linux
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
- NVIDIA GPU driver ≥ 570 ([download](https://www.nvidia.com/Download/index.aspx))
- NVIDIA Container Toolkit ([setup guide](#nvidia-container-toolkit-setup))

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/artifex-studio.git
cd artifex-studio
```

### 2. Create your `.env` file
```bash
cp .env.example .env
```

Open `.env` and fill in your HuggingFace token (required for RMBG-2.0):
```env
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
(Read access is sufficient). You must also accept the RMBG-2.0 model license at
[huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0).

### 3. Build and start
```bash
docker compose up -d --build
```

The first build takes 20–40 minutes — it pulls the CUDA base image (~3 GB) and
installs PyTorch with CUDA 12.8 support (~2.5 GB). Subsequent builds use the
layer cache and complete in under a minute.

### 4. Open the app

Navigate to [http://localhost](http://localhost) in your browser.

---

## First-Run Model Downloads

The first time each feature is used, the worker downloads model weights. This
happens once — weights are cached in the container's `/app/.cache` directory.

| Model | Size | Feature |
|---|---|---|
| Real-ESRGAN x4plus | 67 MB | Crystal Clarity |
| RMBG-2.0 | 176 MB | Magic Eraser — background removal |
| LaMa | 207 MB | Magic Eraser — generative erase |
| SDXL base | ~6.5 GB | Artistic Vision |
| IP-Adapter | ~1 GB | Artistic Vision |

Monitor download progress with:
```bash
docker compose logs -f worker
```

---

## Project Structure
```
artifex-studio/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                  # FastAPI gateway
│
├── worker/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── celery_config.py         # Celery broker + performance tuning
│   ├── tasks.py                 # Task definitions
│   └── src/
│       ├── enhancement.py       # Real-ESRGAN x4plus
│       ├── editing.py           # RMBG-2.0 + LaMa
│       ├── style_transfer.py    # SDXL + IP-Adapter
│       └── stitching.py         # OpenCV panorama
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## NVIDIA Container Toolkit Setup

Required once on Windows/WSL2 before the first build.
```bash
# Inside your WSL2 Ubuntu terminal

# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
```

Then **restart Docker Desktop** from the system tray (right-click → Restart).

Verify GPU passthrough is working:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

---

## Useful Commands
```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Rebuild a single service (e.g. after editing worker code)
docker compose up -d --build worker

# View live worker logs
docker compose logs -f worker

# Check all container health
docker compose ps

# Verify GPU is accessible inside the worker
docker exec neuro_worker python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')
"

# Check what tasks are currently running
docker exec neuro_worker celery -A tasks inspect active

# Manually trigger disk cleanup (deletes output files older than 6 hours)
curl -X DELETE "http://localhost:8000/cleanup?max_age_hours=6"

# Full clean rebuild (wipes all caches and volumes)
docker compose down
docker volume rm artifex_studio_data artifex_redis_data
docker builder prune -af
docker compose up -d --build --no-cache
```

---

## Troubleshooting

### Task runs forever / infinite loading spinner
```bash
# Check if tasks are queued but not being picked up
docker exec neuro_redis redis-cli llen celery

# Check what the worker is currently doing
docker exec neuro_worker celery -A tasks inspect active

# Check worker logs for errors
docker compose logs --tail=80 worker
```

If `llen celery` returns a number > 0, tasks are queuing but the worker isn't
consuming them. Ensure the worker command in `docker-compose.yml` does not
specify `--queues` and that `celery_config.py` has no `task_routes` defined.

### GPU not detected (`CUDA available: False`)

1. Confirm `nvidia-smi` works in PowerShell (Windows driver check)
2. Confirm `nvidia-smi` works inside WSL2
3. Confirm the NVIDIA Container Toolkit is installed (see setup section above)
4. Restart Docker Desktop after toolkit installation
5. Check the worker Dockerfile uses `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`
   as its base image, not `python:3.11-slim`

### Permission denied on `/data`

The `/data` named Docker volume mounts as root-owned. Both the backend and
worker Dockerfiles must **not** use a non-root `USER` directive. If you see
`[Errno 13] Permission denied: '/data/...'`, check that neither Dockerfile
contains a `USER` instruction.

### `No module named torchvision.transforms.functional_tensor`

This is a `basicsr` / `facexlib` compatibility bug with torchvision ≥ 0.16.
The worker Dockerfile contains a `sed` patch that fixes it automatically. If
you see this error, ensure the patch step is present in your `worker/Dockerfile`
and run `docker compose up -d --build worker`.

### RMBG-2.0 gated repo error (401)

1. Accept the license at [huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
2. Create a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add `HUGGING_FACE_HUB_TOKEN=hf_...` to your `.env` file
4. Restart the worker: `docker compose up -d worker`

### Docker build cache corruption
```bash
docker compose down
docker builder prune -af
docker compose up -d --build --no-cache
```

---

## Environment Variables

All variables are set in `.env`. Copy `.env.example` as a starting point.

| Variable | Default | Description |
|---|---|---|
| `HUGGING_FACE_HUB_TOKEN` | — | **Required.** HuggingFace read token for RMBG-2.0 |
| `REDIS_HOST` | `redis` | Redis hostname (Docker service name) |
| `REDIS_PORT` | `6379` | Redis port |
| `DATA_DIR` | `/data` | Shared volume path for uploads and outputs |
| `MAX_UPLOAD_BYTES` | `52428800` | Max upload size (default 50 MB) |
| `FILE_TTL_SECONDS` | `21600` | Output file lifetime before `/cleanup` removes them (6 h) |

---

## Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — async Python web framework
- **[Celery](https://docs.celeryq.dev/)** — distributed task queue
- **[Redis](https://redis.io/)** — message broker and result backend
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** — image restoration
- **[RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)** — background removal
- **[LaMa](https://github.com/advimman/lama)** — image inpainting
- **[Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)** — generative image model
- **[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)** — style transfer via cross-attention
- **[OpenCV](https://opencv.org/)** — panoramic stitching
- **[Nginx](https://nginx.org/)** — static frontend server
- **[PyTorch](https://pytorch.org/)** 2.6+ with CUDA 12.8
