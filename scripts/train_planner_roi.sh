#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${VIGAR_WORKSPACE:?}"
: "${SUBGOAL_PLANNER_DATASET:?Path to the planner LeRobot dataset}"
: "${SUBGOAL_PLANNER_ROI_METADATA:?Path to state_aligned_target_metadata.jsonl.gz}"
: "${SUBGOAL_PLANNER_METADATA_CACHE:?Path to episode_image_edit_metadata.json}"
: "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
export SUBGOAL_PLANNER_OUTPUT="$VIGAR_WORKSPACE/subgoal_planner"
export SUBGOAL_PLANNER_ROI_METADATA SUBGOAL_PLANNER_METADATA_CACHE
export EPISODE_IMAGE_EDIT_DATASET_PATH="$SUBGOAL_PLANNER_DATASET"
export EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION=0.15
export PYTHONPATH="$RELEASE_ROOT/subgoal_planner/training/robotwin:$RELEASE_ROOT/subgoal_planner/training_runtime"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  "$RELEASE_ROOT/subgoal_planner/training/robotwin/train.py" "$@"
