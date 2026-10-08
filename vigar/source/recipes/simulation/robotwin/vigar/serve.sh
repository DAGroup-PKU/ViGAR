#!/usr/bin/env bash
set -euo pipefail
veomni_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$veomni_root"
export PYTHONPATH="$veomni_root"
export PYTHONNOUSERSITE=1 COSMOS_TRAINING=1 WANDB_MODE=disabled
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "${VIGAR_PYTHON:-$veomni_root/.venv/bin/python}" \
  -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m recipes.simulation.robotwin.vigar.server "$@"
