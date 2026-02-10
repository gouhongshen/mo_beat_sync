#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip >/dev/null
pip install -r requirements.txt

mkdir -p source_movies source_songs outputs/final workdir

WORKER_ARG=()
if [[ -n "${MO_WORKERS:-}" ]]; then
  WORKER_ARG=(--workers "$MO_WORKERS")
fi

GPU_ARG=()
if [[ -n "${MO_GPU_MODE:-}" ]]; then
  GPU_ARG=(--gpu-mode "$MO_GPU_MODE")
fi

CHUNK_ARG=()
if [[ -n "${MO_CHUNK_MIN_MINUTES:-}" && -n "${MO_CHUNK_MAX_MINUTES:-}" ]]; then
  CHUNK_ARG=(--chunk-min-minutes "$MO_CHUNK_MIN_MINUTES" --chunk-max-minutes "$MO_CHUNK_MAX_MINUTES")
fi

OVERLAP_ARG=()
if [[ -n "${MO_CHUNK_OVERLAP_SECONDS:-}" ]]; then
  OVERLAP_ARG=(--chunk-overlap-seconds "$MO_CHUNK_OVERLAP_SECONDS")
fi

PYTHONPATH=src python -m mo_beat_sync \
  --movies-dir source_movies \
  --songs-dir source_songs \
  --output-path outputs/final/beat_sync.mp4 \
  --workdir workdir \
  "${WORKER_ARG[@]}" \
  "${GPU_ARG[@]}" \
  "${CHUNK_ARG[@]}" \
  "${OVERLAP_ARG[@]}"
