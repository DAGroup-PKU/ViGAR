#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/env.sh"
phase="${1:?Use roi40k or random500}"; shift
: "${PYTHON_BIN:?}" "${GOALWAM_WORKSPACE:?}" "${GOALWAM_I2I_NORMAL30K:?}"
: "${GOALWAM_I2I_ASSET_ROOT:?}" "${GOALWAM_I2I_METADATA_ROOT:?}"
: "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}" "${QWEN_TOKENIZER_PATH:?}"
resume=()
case "$phase" in
  roi40k)
    entry=roi40k; mode=ee96_mix19; steps=40000; save_every=4000
    export GOALWAM_I2I_WORK="${GOALWAM_I2I_ROI40_WORK:?}"
    export EPISODE_IMAGE_EDIT_DATASET_PATH="${GOALWAM_I2I_ROI40_DATASET:?}"
    ;;
  random500)
    entry=roi_random500; mode=ee4_random500; steps=70000; save_every=5000
    export GOALWAM_I2I_WORK="${GOALWAM_I2I_RANDOM500_WORK:?}"
    export EPISODE_IMAGE_EDIT_DATASET_PATH="${GOALWAM_I2I_RANDOM500_DATASET:?}"
    resume=(--resume "${GOALWAM_I2I_ROI40K:?Full training state is required}")
    ;;
  *) echo "Expected roi40k or random500" >&2; exit 2 ;;
esac
export GOALWAM_I2I_OUTPUT="$GOALWAM_WORKSPACE/i2i-$phase"
export PYTHONPATH="$RELEASE_ROOT/i2i/training/$entry:$RELEASE_ROOT/i2i/training_runtime"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  "$RELEASE_ROOT/i2i/training/$entry/train.py" --mode "$mode" \
  --steps "$steps" --batch 4 --save-every "$save_every" "${resume[@]}" "$@"
