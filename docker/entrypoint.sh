#!/bin/sh
# API container entrypoint: ensure a champion model exists, then serve.
set -e

python /app/scripts/bootstrap.py

exec uvicorn evdecafs_serve.serving.app:app --host 0.0.0.0 --port 8000
