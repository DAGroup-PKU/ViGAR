#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
mode="${1:?Use normal or icl}"; shift
: "${PYTHON_BIN:?}" "${VIGAR_WORKSPACE:?}" "${VIGAR_REAL_ROBOT_DATASET:?}"
: "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
case "$mode" in
  normal) goal_condition=none ;;
  # The episode's final frame conditions both the reasoner and the generator.
  icl) goal_condition=reasoner_and_generator ;;
  *) echo "Expected normal or icl" >&2; exit 2 ;;
esac
# Default: one machine, 8 GPUs x 128 samples = global batch 1024.
# Two machines: set NNODES=2, NODE_RANK, MASTER_ADDR and SAMPLES_PER_GPU=64.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-128}"
export PYTHONPATH="$RELEASE_ROOT/subgoal_planner/inference_runtime"
export EPISODE_IMAGE_EDIT_DATASET_PATH="$VIGAR_REAL_ROBOT_DATASET"
# AgiBot A2 profile: the precomputed 320x384 three-camera canvas.
export EPISODE_IMAGE_EDIT_LEROBOT_VIDEO_KEY=observation.images.head_front_color
# Inputs in the final 15% of a subtask target the next subtask's end frame.
export EPISODE_IMAGE_EDIT_NEXT_SUBGOAL_TAIL_FRACTION=0.15
export EPISODE_IMAGE_EDIT_GOAL_CONDITION_MODE="$goal_condition"
export EPISODE_DECODE_CACHE_MAX="${EPISODE_DECODE_CACHE_MAX:-1024}"
export IMAGINAIRE_OUTPUT_ROOT="$VIGAR_WORKSPACE/planner-real-robot-$mode"
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m cosmos_framework.scripts.train --sft-toml "$RELEASE_ROOT/configs/subgoal_planner_real_robot.toml" -- \
  job.name="real_robot_planner_$mode" \
  model.config.parallelism.data_parallel_shard_degree=8 \
  model.config.parallelism.data_parallel_replicate_degree="$NNODES" \
  dataloader_train.batcher.max_samples_per_batch="$SAMPLES_PER_GPU" \
  dataloader_train.num_workers=16 dataloader_train.prefetch_factor=2 \
  dataloader_train.distributor.dataset.include_final_as_input=true \
  trainer.seed=42 "$@"
