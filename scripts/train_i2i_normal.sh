#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
: "${PYTHON_BIN:?}" "${GOALWAM_WORKSPACE:?}" "${GOALWAM_I2I_NORMAL_DATASET:?}"
: "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
export PYTHONPATH="$RELEASE_ROOT/i2i/inference_runtime"
export EPISODE_IMAGE_EDIT_DATASET_PATH="$GOALWAM_I2I_NORMAL_DATASET"
export IMAGINAIRE_OUTPUT_ROOT="$GOALWAM_WORKSPACE/i2i-normal"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m cosmos_framework.scripts.train --sft-toml "$RELEASE_ROOT/configs/i2i_normal30k.toml" -- \
  dataloader_train.batcher.max_samples_per_batch=128 \
  dataloader_train.distributor.dataset.include_final_as_input=true \
  trainer.seed=42 "$@"
