#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${NATIVE_SIM_PYTHON:?}" "${NATIVE_EVAL_ROOT:?Run prepare_eval.py first}"
export EVAL_NODE_RANK="${EVAL_NODE_RANK:-0}"
: "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
export PYTHONPATH="$RELEASE_ROOT/evaluation:$RELEASE_ROOT/evaluation/planner_service"
exec "$PYTHON_BIN" "$RELEASE_ROOT/evaluation/run_eval.py" "$@"
