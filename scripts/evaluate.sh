#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${NATIVE_SIM_PYTHON:?}" "${NATIVE_EVAL_ROOT:?Run prepare_eval.py first}"
: "${EVAL_NODE_RANK:?Set 0 or 1}" "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
export EPISODE_IMAGE_EDIT_DATASET_PATH="${GOALWAM_I2I_NORMAL_DATASET:?}"
export PYTHONPATH="$RELEASE_ROOT/evaluation:$RELEASE_ROOT/evaluation/i2i_service"
exec "$PYTHON_BIN" "$RELEASE_ROOT/evaluation/run_eval.py" "$@"
