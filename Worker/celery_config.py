import os

# ─── Connection ───────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

broker_url     = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
result_backend = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ─── Serialisation ───────────────────────────────────────────────
task_serializer   = "json"
result_serializer = "json"
accept_content    = ["json"]
timezone          = "UTC"
enable_utc        = True

# ─── Task Lifecycle ───────────────────────────────────────────────
task_track_started         = True
task_acks_late             = True
task_reject_on_worker_lost = True

# ─── Time Limits ─────────────────────────────────────────────────
task_soft_time_limit = 1200   # 20 min
task_time_limit      = 1260   # 21 min hard kill

# ─── Result Storage ───────────────────────────────────────────────
result_expires = 6 * 60 * 60  # 6 hours

# ─── Worker Tuning ────────────────────────────────────────────────
worker_max_tasks_per_child = 100
worker_prefetch_multiplier = 1

# ─── NO task_routes, NO task_default_queue override ───────────────
# The backend sends tasks to Celery's built-in default queue "celery".
# The worker must consume that same queue. Any custom queue names here
# cause a mismatch where tasks queue up and are never picked up.

# ─── Monitoring ───────────────────────────────────────────────────
worker_send_task_events = True
task_send_sent_event    = True