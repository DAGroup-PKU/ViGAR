#!/usr/bin/env bash
set -Eeuo pipefail
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${VIGAR_ENV_FILE:-$RELEASE_ROOT/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi
export RELEASE_ROOT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export VIGAR_BATCH_CFG=0
export EPISODE_IMAGE_EDIT_DATASET_FORMAT=lerobot
export EPISODE_IMAGE_EDIT_LEROBOT_VIDEO_KEY=observation.images.cam_high
export EPISODE_IMAGE_EDIT_THREE_CAMERA=true
export EPISODE_IMAGE_EDIT_TARGET_MODE=segment_final
export EPISODE_IMAGE_EDIT_SEGMENT_SOURCE=annotations
# Next-subgoal skipping: in the last 15% of a stage, the target is the next stage's subgoal.
export EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION=0.15
export EPISODE_IMAGE_EDIT_REASONER_TARGET=false

if [[ -x "${PYTHON_BIN:-}" ]]; then
  CUDA_LIBRARIES="$("$PYTHON_BIN" -c 'import site; from pathlib import Path; print(":".join(str(p) for s in site.getsitepackages() for p in (Path(s)/"nvidia").glob("*/lib") if p.is_dir()))')"
  export LD_LIBRARY_PATH="${CUDA_LIBRARIES}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
