#!/bin/bash
set -e

mkdir -p \
    /data/.cache/huggingface/hub \
    /data/.cache/torch/realesrgan \
    /data/.cache/torch/hub

chown -R celery:celery /data/.cache

exec gosu celery "$@"