"""
ArtifexStudio — Celery Configuration
──────────────────────────────────────
Shared by both the FastAPI backend (task dispatch) and the Celery worker
(task execution).  The worker loads this via app.config_from_object().

Improvements over v1
  • task_track_started — tasks now surface a STARTED state to the frontend
  • task_soft_time_limit / task_time_limit — a hung GPU inference can no
    longer lock the worker process forever
  • result_expires — Redis no longer accumulates stale results indefinitely
  • task_reject_on_worker_lost — tasks are re-queued if the worker process
    dies mid-execution, rather than silently disappearing
  • worker_max_tasks_per_child raised from 10 → 100: all AI models load at
    child-process startup (not per task). At 10, models were being fully
    reloaded every 10 tasks — a 5–10 minute reload penalty each time.
    100 still recycles children to prevent long-running memory leaks while
    keeping model reloads rare.
  • Task routing — each task type gets its own named queue so you can spin
    up specialised workers (e.g. a second GPU worker for style-transfer only)
    without changing any application code.
  • worker_send_task_events — enables real-time monitoring via Flower
    (docker run -p 5555:5555 mher/flower)
"""

import os

# ─── Connection ───────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB",   "0"))

broker_url      = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
result_backend  = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ─── Serialisation ───────────────────────────────────────────────
task_serializer   = "json"
result_serializer = "json"
accept_content    = ["json"]
timezone          = "UTC"
enable_utc        = True

# ─── Task Lifecycle ───────────────────────────────────────────────
# Report STARTED state so the frontend sees a three-step progression:
# PENDING → STARTED → SUCCESS/FAILURE
task_track_started = True

# Only acknowledge a task after it completes, not when it is received.
# If the worker crashes mid-task the broker will re-deliver it.
task_acks_late = True

# Re-queue tasks that were unacknowledged when a worker process died.
# Pairs with task_acks_late to give at-least-once delivery.
task_reject_on_worker_lost = True

# ─── Time Limits ─────────────────────────────────────────────────
# These are the most important safety valves in the config.
# A hung SDXL inference or a deadlocked OpenCV call would otherwise block
# the worker process permanently without these limits.
#
#   soft_time_limit  → raises SoftTimeLimitExceeded inside the task function,
#                       giving it a chance to clean up GPU memory and return
#                       a structured error dict.
#   time_limit       → sends SIGKILL if the task is still running 60 s later.
#
# Generous limits — style-transfer can legitimately take 2–3 minutes on
# first run when model weights are being downloaded.
task_soft_time_limit = 600   # 10 min soft (SoftTimeLimitExceeded raised)
task_time_limit      = 660   # 11 min hard kill

# ─── Result Storage ───────────────────────────────────────────────
# Auto-expire task results from Redis after 6 hours.
# The actual image files on /data are managed separately by /cleanup.
result_expires = 6 * 60 * 60  # 6 hours

# ─── Worker Tuning ────────────────────────────────────────────────
# GPU AI workers load all models at subprocess startup (module-level).
# Setting this too low causes constant model reloads — at 10 tasks/child,
# the SDXL + LaMa + RMBG models (totalling several GB) would reload every
# 10 tasks.  100 is a reasonable balance between leak prevention and
# avoiding that reload penalty.
worker_max_tasks_per_child  = 100

# Pull only one task at a time. The GPU can only run one inference anyway,
# and prefetching more would block the queue for other workers/users.
worker_prefetch_multiplier  = 1

# ─── Task Routing ─────────────────────────────────────────────────
# Each task type routes to its own named queue.  With a single worker
# this has no effect (the worker listens on all queues).  When you scale
# to multiple workers you can pin workers to specific queues:
#
#   celery -A tasks worker --queues=style,edit   # dedicated GPU worker
#   celery -A tasks worker --queues=enhance      # dedicated enhance worker
#
task_routes = {
    "tasks.task_enhance_image":  {"queue": "enhance"},
    "tasks.task_stitch_images":  {"queue": "stitch"},
    "tasks.task_style_transfer": {"queue": "style"},
    "tasks.task_edit_image":     {"queue": "edit"},
}

task_default_queue = "default"

# ─── Monitoring ───────────────────────────────────────────────────
# Emit real-time task events so Flower / Prometheus can observe the worker.
worker_send_task_events = True
task_send_sent_event    = True