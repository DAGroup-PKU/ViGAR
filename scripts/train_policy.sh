#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${VIGAR_POLICY_WORKSPACE:?}" "${VIGAR_DATASET_MANIFEST:?}"
: "${VIGAR_GENERATED_GOAL_CACHE:?}" "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
# Default: one machine, 8 GPUs x 32 samples = global batch 256.
# Two machines: set NNODES=2, NODE_RANK, MASTER_ADDR and pass --batch 16.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHONPATH="$RELEASE_ROOT/vigar/training:$RELEASE_ROOT/vigar/source:$RELEASE_ROOT/vigar/source/third_party/cosmos_runtime"
export VIGAR_POLICY_RUN_NAME="${VIGAR_POLICY_RUN_NAME:-robotwin_c2r}"
export WANDB_NAME="${WANDB_NAME:-robotwin_c2r}"
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$RELEASE_ROOT/vigar/training/train_native.py" \
  --goal-mode multi_view --steps 50000 --batch 32 --save-every 4000 "$@"
