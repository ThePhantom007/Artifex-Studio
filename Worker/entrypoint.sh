#!/bin/bash
set -e

# Fix ownership of the cache directories on the mounted volume.
# This runs as root (before su), then drops to the celery user.
mkdir -p \
    /data/.cache/huggingface/hub \
    /data/.cache/torch/realesrgan \
    /data/.cache/torch/hub

chown -R celery:celery /data/.cache

# Drop to non-root user and exec the actual worker command
exec gosu celery "$@"