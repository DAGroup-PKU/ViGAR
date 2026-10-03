#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${GOALWAM_POLICY_WORKSPACE:?}" "${GOALWAM_DATASET_MANIFEST:?}"
: "${GOALWAM_GENERATED_GOAL_CACHE:?}" "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"
export PYTHONPATH="$RELEASE_ROOT/policy/training:$RELEASE_ROOT/policy/source:$RELEASE_ROOT/policy/source/third_party/goalwam"
export POLICY8_RUN_NAME="${POLICY8_RUN_NAME:-robotwin_c2r}"
export WANDB_NAME="${WANDB_NAME:-robotwin_c2r}"
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$RELEASE_ROOT/policy/training/train_native.py" \
  --goal-mode multi_view --steps 50000 --batch 16 --save-every 4000 "$@"
