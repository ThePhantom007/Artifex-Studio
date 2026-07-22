# Deploying ArtifexStudio — Vercel + Modal

This replaces the local `docker compose` stack (Redis + Celery + one
always-on GPU worker) with two hosted, pay-per-use pieces:

- **Frontend** → Vercel (or Netlify) — free, static, always on
- **Backend + 4 tools** → Modal — one platform, no Redis, GPU functions
  that scale to zero when idle

Your original `docker-compose.yml` setup still works unchanged for local
development. This is an additional deployment path, not a replacement.

---

## 1. Set up Modal

```bash
pip install modal
modal setup          # opens a browser to link your Modal account
```

Create the secret Modal needs for RMBG-2.0 (gated on Hugging Face):

```bash
modal secret create artifex-hf-secret HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

Use a **freshly rotated** token here, not the one that was sitting in your
local `.env` — treat that one as already exposed.

## 2. Deploy the backend + AI functions

From the project root:

```bash
modal deploy modal_app/app.py
```

First deploy builds two images (GPU image with torch/diffusers/etc., and a
small CPU image) — this can take 10–20 minutes. Subsequent deploys reuse
cached layers and are much faster, unless you change
`Worker/src/*.py` or the pinned packages in `modal_app/app.py`.

When it finishes, Modal prints a URL for `fastapi_app`, something like:

```
https://your-username--artifex-studio-fastapi-app.modal.run
```

Copy that URL — you need it in the next step.

**Sanity check** before wiring up the frontend:

```bash
curl https://your-username--artifex-studio-fastapi-app.modal.run/health
```

## 3. Point the frontend at it

Edit `Frontend/config.js`:

```js
window.ARTIFEX_API_URL = "https://your-username--artifex-studio-fastapi-app.modal.run";
```

## 4. Deploy the frontend to Vercel

```bash
cd Frontend
npx vercel deploy --prod
```

(Or connect the repo in the Vercel dashboard and set the project's root
directory to `Frontend/` — no build step needed, it's static HTML/CSS/JS.)

## 5. Test end to end

Open your Vercel URL and run each of the four tools once. The **first**
call to each GPU tool will be slow (cold start — container spin-up +
model load, up to ~60s for Artistic Vision since SDXL is 6.5GB). Calls
within the next 5 minutes (`scaledown_window` in `modal_app/app.py`) reuse
the warm container and are fast.

---

## Notes on what changed vs. the Docker setup

| Old (Docker) | New (Vercel + Modal) |
|---|---|
| Nginx serving `Frontend/` | Vercel static hosting |
| FastAPI on a dedicated port | FastAPI served as a Modal ASGI app |
| Redis + Celery task queue | `Function.spawn()` / `FunctionCall.get()` |
| One GPU worker container, always on | 3 separate GPU functions, scale to zero |
| Panoramic Stitching ran on the same GPU worker | Runs as its own CPU-only Modal function (matches the fact that `stitching.py` never used CUDA) |
| `/data` named Docker volume | `artifex-data` Modal Volume |
| `.cache/huggingface`, `.cache/torch` baked into the image | `artifex-model-cache` Modal Volume, mounted at `/cache` |

## Tuning cost vs. cold-start latency

`scaledown_window=300` in `modal_app/app.py` keeps each GPU function's
container warm for 5 minutes after its last call. Lower it to cut idle
cost further (at the price of more frequent cold starts); raise it during
a live demo/judging window so back-to-back tool usage stays fast. You can
also `modal.Function` `.with_options(scaledown_window=...)` this per
environment without editing the file, if you want a demo-specific config.

## Rough cost

GPU billing is per-second and stops entirely when idle. A single Real-ESRGAN
or RMBG/LaMa call on a T4 typically costs well under a cent; an SDXL style
transfer on an A10G is a few cents. Modal's free monthly credit will very
likely cover ongoing testing plus anyone (e.g. a hiring manager) trying the
live demo, as long as it isn't seeing sustained heavy traffic.