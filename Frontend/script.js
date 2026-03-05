const API_URL = `http://${window.location.hostname}:8000`;

// ─── Per-feature timeout limits (milliseconds) ────────────────────
// These reflect realistic worst-case inference times including
// first-run model weight downloads. If a task exceeds this duration
// the frontend gives up and shows an error rather than looping forever.
const TASK_TIMEOUTS = {
  enhance: 5  * 60 * 1000,   // 5 min  — Real-ESRGAN
  edit:    4  * 60 * 1000,   // 4 min  — RMBG-2.0 / LaMa
  style:   8  * 60 * 1000,   // 8 min  — SDXL (large model, slow first run)
  stitch:  4  * 60 * 1000,   // 4 min  — OpenCV panorama
};

const POLL_INTERVAL_MS = 2000;   // check every 2 seconds

// ─── Loading messages ─────────────────────────────────────────────
const LOADING_MSGS = {
  enhance: [
    "Analysing image degradation patterns…",
    "Running RRDB blocks — 23 layers deep…",
    "Hallucinating lost texture at 4× scale…",
    "Reconstructing fine detail with FP16 precision…",
    "Finalising high-resolution output…",
  ],
  edit: [
    "Segmenting foreground from background…",
    "Applying neural alpha matting…",
    "Refining hair and edge boundaries…",
    "Compositing transparent RGBA output…",
  ],
  style: [
    "Encoding style reference with CLIP ViT-H…",
    "Injecting style features into cross-attention layers…",
    "Running SDXL denoising steps 1–10…",
    "Running SDXL denoising steps 11–20…",
    "Running SDXL denoising steps 21–30…",
    "Decoding latents through VAE…",
  ],
  stitch: [
    "Detecting SIFT keypoints across frames…",
    "Computing homography transformations…",
    "Matching exposure histograms…",
    "Warping and blending seams…",
    "Auto-cropping black border regions…",
  ],
};

let _loadingIntervals = {};

function startLoadingMessages(tab) {
  const el = document.getElementById(`${tab}-loading-msg`);
  if (!el) return;
  const msgs = LOADING_MSGS[tab] || [];
  let i = 0;
  el.textContent = msgs[0] || "";
  _loadingIntervals[tab] = setInterval(() => {
    i = (i + 1) % msgs.length;
    el.style.opacity = "0";
    setTimeout(() => {
      el.textContent = msgs[i];
      el.style.opacity = "1";
    }, 300);
  }, 2800);
}

function stopLoadingMessages(tab) {
  clearInterval(_loadingIntervals[tab]);
  delete _loadingIntervals[tab];
  const el = document.getElementById(`${tab}-loading-msg`);
  if (el) el.textContent = "";
}

// ─── Error popup ──────────────────────────────────────────────────
function showError(title, message) {
  // Remove any existing popup first
  const existing = document.getElementById("artifex-error-popup");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "artifex-error-popup";
  overlay.style.cssText = `
    position: fixed; inset: 0; z-index: 9999;
    background: rgba(2, 12, 27, 0.82);
    backdrop-filter: blur(8px);
    display: flex; align-items: center; justify-content: center;
    padding: 1.5rem;
    animation: section-in 0.25s ease forwards;
  `;

  overlay.innerHTML = `
    <div style="
      background: #fff;
      border-radius: 22px;
      padding: 2.2rem 2.4rem;
      max-width: 420px;
      width: 100%;
      box-shadow: 0 32px 64px rgba(0,0,0,0.45);
      border: 1px solid #fee2e2;
      text-align: center;
    ">
      <div style="font-size: 2.4rem; margin-bottom: 0.75rem;">❌</div>
      <h3 style="
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        color: #991b1b;
        margin-bottom: 0.6rem;
        letter-spacing: -0.02em;
      ">${title}</h3>
      <p style="
        font-family: 'DM Sans', sans-serif;
        font-size: 0.875rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 1.5rem;
      ">${message}</p>
      <button onclick="document.getElementById('artifex-error-popup').remove()" style="
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.8rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s;
      ">Dismiss</button>
    </div>
  `;

  // Close on backdrop click
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });

  document.body.appendChild(overlay);
}

// ─── Core polling engine ──────────────────────────────────────────
async function pollTask(taskId, imgElementId, loadingAreaId, resultAreaId, downloadBtnId, tab) {
  startLoadingMessages(tab);

  const deadline = Date.now() + (TASK_TIMEOUTS[tab] || 5 * 60 * 1000);
  let consecutiveNetworkErrors = 0;

  while (true) {
    // ── Timeout guard ──────────────────────────────────────────
    if (Date.now() > deadline) {
      stopLoadingMessages(tab);
      resetUI(tab);
      showError(
        "Task Timed Out",
        `The ${tab} task did not complete within the expected time. ` +
        `This can happen on first run while model weights are downloading. ` +
        `Check <code>docker compose logs worker</code> for details, then try again.`
      );
      return;
    }

    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));

    let data;
    try {
      const res = await fetch(`${API_URL}/status/${taskId}`, {
        signal: AbortSignal.timeout(8000),   // 8-second fetch timeout
      });

      if (!res.ok) {
        throw new Error(`Status endpoint returned HTTP ${res.status}`);
      }

      data = await res.json();
      consecutiveNetworkErrors = 0;   // reset on success

    } catch (fetchErr) {
      consecutiveNetworkErrors++;
      console.warn(`[poll] Network error (${consecutiveNetworkErrors}):`, fetchErr.message);

      // Allow up to 5 consecutive network errors before giving up —
      // handles brief Docker networking hiccups without false failures.
      if (consecutiveNetworkErrors >= 5) {
        stopLoadingMessages(tab);
        resetUI(tab);
        showError(
          "Connection Lost",
          "Could not reach the backend after 5 attempts. " +
          "Check that all containers are running with <code>docker compose ps</code>."
        );
        return;
      }
      continue;   // retry after next interval
    }

    const status = data.status;

    // ── Terminal states ────────────────────────────────────────
    if (status === "SUCCESS") {
      stopLoadingMessages(tab);

      const imageUrl = `${API_URL}/image/${data.result}?t=${Date.now()}`;
      const imgEl = document.getElementById(imgElementId);
      if (imgEl) imgEl.src = imageUrl;

      document.getElementById(loadingAreaId)?.classList.add("hidden");
      document.getElementById(resultAreaId)?.classList.remove("hidden");

      const btn = downloadBtnId ? document.getElementById(downloadBtnId) : null;
      if (btn) {
        btn.onclick = (e) => {
          e.preventDefault();
          forceDownload(imageUrl, `ArtifexStudio_${data.result}`);
        };
      }
      return;
    }

    if (status === "FAILURE") {
      stopLoadingMessages(tab);
      resetUI(tab);
      showError(
        "AI Engine Error",
        data.error
          ? escapeHtml(data.error)
          : "The worker returned a failure with no error message. " +
            "Run <code>docker compose logs worker</code> to see the full traceback."
      );
      return;
    }

    // REVOKED — task was cancelled (e.g. via /cancel endpoint)
    if (status === "REVOKED") {
      stopLoadingMessages(tab);
      resetUI(tab);
      showError("Task Cancelled", "This task was revoked before it could complete.");
      return;
    }

    // PENDING / STARTED / RETRY — still in progress, keep polling.
    // No action needed; the loading UI is already visible.
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ─── SPA Navigation ───────────────────────────────────────────────
function switchTab(tabName) {
  document.querySelectorAll(".tab-content").forEach((el) => el.classList.add("hidden"));
  document.querySelectorAll(".nav-link").forEach((el) => el.classList.remove("active"));

  const section = document.getElementById(`sec-${tabName}`);
  section.classList.remove("hidden");
  section.style.animation = "none";
  section.offsetHeight;
  section.style.animation = "";

  document.getElementById(`nav-${tabName}`).classList.add("active");
}

function toggleMaskInput() {
  const action = document.getElementById("edit-action").value;
  document.getElementById("mask-container").classList.toggle("hidden", action !== "erase");
}

function resetUI(tab) {
  document.getElementById(`ui-${tab}-result`)?.classList.add("hidden");
  document.getElementById(`ui-${tab}-loading`)?.classList.add("hidden");
  document.getElementById(`ui-${tab}-upload`)?.classList.remove("hidden");
  stopLoadingMessages(tab);
}

// ─── 1. Crystal Clarity ───────────────────────────────────────────
document.getElementById("file-enhance").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  document.getElementById("ui-enhance-upload").classList.add("hidden");
  document.getElementById("ui-enhance-loading").classList.remove("hidden");

  const origImg = document.getElementById("enhance-orig");
  if (origImg) origImg.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_URL}/enhance`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    pollTask(data.task_id, "enhance-res", "ui-enhance-loading", "ui-enhance-result", "enhance-download", "enhance");
  } catch (err) {
    resetUI("enhance");
    showError("Upload Failed", escapeHtml(err.message));
  }
});

// ─── 2. Magic Eraser ──────────────────────────────────────────────
async function submitEdit() {
  const imgFile = document.getElementById("file-edit-main").files[0];
  const action  = document.getElementById("edit-action").value;
  const maskInput = document.getElementById("file-edit-mask");
  const maskFile  = maskInput?.files[0];

  if (!imgFile) { showError("Missing Image", "Please upload an image first."); return; }
  if (action === "erase" && !maskFile) {
    showError("Missing Mask", "Generative Erase requires a mask image. Paint the object white in any image editor.");
    return;
  }

  document.getElementById("ui-edit-upload").classList.add("hidden");
  document.getElementById("ui-edit-loading").classList.remove("hidden");

  const formData = new FormData();
  formData.append("image",  imgFile);
  formData.append("action", action);
  if (maskFile) formData.append("mask", maskFile);

  try {
    const res = await fetch(`${API_URL}/edit`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    pollTask(data.task_id, "edit-res", "ui-edit-loading", "ui-edit-result", "edit-download", "edit");
  } catch (err) {
    resetUI("edit");
    showError("Upload Failed", escapeHtml(err.message));
  }
}

// ─── 3. Artistic Vision ───────────────────────────────────────────
async function submitStyle() {
  const contentFile = document.getElementById("file-style-content").files[0];
  const styleFile   = document.getElementById("file-style-reference").files[0];
  const prompt      = document.getElementById("style-prompt")?.value || "";

  if (!contentFile || !styleFile) {
    showError("Missing Images", "Please upload both a canvas image and a style reference.");
    return;
  }

  document.getElementById("ui-style-upload").classList.add("hidden");
  document.getElementById("ui-style-loading").classList.remove("hidden");

  const formData = new FormData();
  formData.append("content_image", contentFile);
  formData.append("style_image",   styleFile);
  formData.append("prompt",        prompt);

  try {
    const res = await fetch(`${API_URL}/style-transfer`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    pollTask(data.task_id, "style-res", "ui-style-loading", "ui-style-result", "style-download", "style");
  } catch (err) {
    resetUI("style");
    showError("Upload Failed", escapeHtml(err.message));
  }
}

// ─── 4. Deep Stitch ───────────────────────────────────────────────
document.getElementById("file-stitch").addEventListener("change", async (e) => {
  const files = e.target.files;
  if (files.length < 2) {
    showError("Too Few Images", "Please select at least 2 overlapping images.");
    return;
  }

  document.getElementById("ui-stitch-upload").classList.add("hidden");
  document.getElementById("ui-stitch-loading").classList.remove("hidden");

  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  try {
    const res = await fetch(`${API_URL}/stitch`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    pollTask(data.task_id, "stitch-res", "ui-stitch-loading", "ui-stitch-result", "stitch-download", "stitch");
  } catch (err) {
    resetUI("stitch");
    showError("Upload Failed", escapeHtml(err.message));
  }
});

// ─── Per-tab reset functions ──────────────────────────────────────
function resetEnhance() {
  resetUI("enhance");
  document.getElementById("file-enhance").value = "";
  document.getElementById("enhance-orig")?.removeAttribute("src");
  document.getElementById("enhance-res")?.removeAttribute("src");
}
function resetEdit() {
  resetUI("edit");
  document.getElementById("file-edit-main").value = "";
  if (document.getElementById("file-edit-mask"))
    document.getElementById("file-edit-mask").value = "";
  document.getElementById("edit-res")?.removeAttribute("src");
}
function resetStyle() {
  resetUI("style");
  document.getElementById("file-style-content").value = "";
  document.getElementById("file-style-reference").value = "";
  document.getElementById("style-res")?.removeAttribute("src");
}
function resetStitch() {
  resetUI("stitch");
  document.getElementById("file-stitch").value = "";
  document.getElementById("stitch-res")?.removeAttribute("src");
}

// ─── Force download ───────────────────────────────────────────────
async function forceDownload(url, filename) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = blobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(blobUrl);
      document.body.removeChild(a);
    }, 300);
  } catch (e) {
    console.error("Force download failed, opening in new tab:", e);
    window.open(url, "_blank");
  }
}