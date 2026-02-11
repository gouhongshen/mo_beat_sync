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
if [[ "${MO_INSTALL_ML:-0}" == "1" ]]; then
  pip install -r requirements-ml.txt
fi
if [[ "${MO_INSTALL_MADMOM:-0}" == "1" ]]; then
  set +e
  pip install -r requirements-ml-beat.txt
  if [[ $? -ne 0 ]]; then
    echo "[WARN] madmom install failed, fallback to librosa beat backend."
  fi
  set -e
fi

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

BEAT_ARG=()
if [[ -n "${MO_BEAT_MODEL:-}" ]]; then
  BEAT_ARG=(--beat-model "$MO_BEAT_MODEL")
fi

EMBEDDING_ARG=()
if [[ -n "${MO_EMBEDDING_MODEL:-}" ]]; then
  EMBEDDING_ARG=(--embedding-model "$MO_EMBEDDING_MODEL")
fi

PLANNER_ARG=()
if [[ -n "${MO_PLANNER_BEAM_WIDTH:-}" ]]; then
  PLANNER_ARG+=(--planner-beam-width "$MO_PLANNER_BEAM_WIDTH")
fi
if [[ -n "${MO_PLANNER_PER_STATE_CANDIDATES:-}" ]]; then
  PLANNER_ARG+=(--planner-per-state-candidates "$MO_PLANNER_PER_STATE_CANDIDATES")
fi
if [[ -n "${MO_PLANNER_SLOT_SKIP_PENALTY:-}" ]]; then
  PLANNER_ARG+=(--planner-slot-skip-penalty "$MO_PLANNER_SLOT_SKIP_PENALTY")
fi
if [[ -n "${MO_PLANNER_MIN_REUSE_GAP:-}" ]]; then
  PLANNER_ARG+=(--planner-min-reuse-gap "$MO_PLANNER_MIN_REUSE_GAP")
fi
if [[ -n "${MO_PLANNER_REUSE_PENALTY:-}" ]]; then
  PLANNER_ARG+=(--planner-reuse-penalty "$MO_PLANNER_REUSE_PENALTY")
fi

PYTHONPATH=src python -m mo_beat_sync \
  --movies-dir source_movies \
  --songs-dir source_songs \
  --output-path outputs/final/beat_sync.mp4 \
  --workdir workdir \
  "${WORKER_ARG[@]}" \
  "${GPU_ARG[@]}" \
  "${BEAT_ARG[@]}" \
  "${EMBEDDING_ARG[@]}" \
  "${PLANNER_ARG[@]}" \
  "${CHUNK_ARG[@]}" \
  "${OVERLAP_ARG[@]}"
