#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${VIGAR_POLICY_WORKSPACE:?}" "${VIGAR_REAL_ROBOT_DATASET:?}" "${VIGAR_PARQUET_CACHE:?}"
: "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
# Default: one machine, 8 GPUs x 128 samples = global batch 1024.
# Two machines: set NNODES=2, NODE_RANK, MASTER_ADDR and pass --batch 64.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHONPATH="$RELEASE_ROOT/vigar/training:$RELEASE_ROOT/vigar/source:$RELEASE_ROOT/vigar/source/third_party/cosmos_runtime"
export VIGAR_POLICY_RUN_NAME="${VIGAR_POLICY_RUN_NAME:-real_robot_900hr}"
export WANDB_NAME="${WANDB_NAME:-real_robot_900hr}"
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$RELEASE_ROOT/vigar/training/train_real_robot.py" \
  --steps 100000 --batch 128 --save-every 10000 "$@"
