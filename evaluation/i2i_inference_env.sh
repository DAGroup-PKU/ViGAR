#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/env.sh"
: "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
export EPISODE_IMAGE_EDIT_DATASET_PATH="${SUBGOAL_PLANNER_NORMAL_DATASET:?}"
