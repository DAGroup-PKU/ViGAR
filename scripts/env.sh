#!/usr/bin/env bash
set -Eeuo pipefail
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${GOALWAM_ENV_FILE:-$RELEASE_ROOT/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi
export RELEASE_ROOT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
export GOALWAM_BATCH_CFG=0
export EPISODE_IMAGE_EDIT_DATASET_FORMAT=lerobot
export EPISODE_IMAGE_EDIT_LEROBOT_VIDEO_KEY=observation.images.cam_high
export EPISODE_IMAGE_EDIT_THREE_CAMERA=true
export EPISODE_IMAGE_EDIT_TARGET_MODE=segment_final
export EPISODE_IMAGE_EDIT_SEGMENT_SOURCE=annotations
# Preserve the released planner's target-selection setting.
export EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION=0
export EPISODE_IMAGE_EDIT_REASONER_TARGET=false
