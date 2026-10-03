# Setup and usage

## Environments

The source training stack uses Linux, Python 3.13, PyTorch 2.10 and CUDA 12.8, with matching Transformer Engine, FlashAttention, Wan VAE and TorchCodec/PyAV dependencies. Dependency declarations are retained in `policy/source/pyproject.toml` and each I2I runtime's `pyproject.toml`.

For the policy environment, use the CUDA dependency groups declared by the policy project; the native runtime is a local package under `policy/source/third_party/goalwam`. The I2I source is isolated through launcher-provided `PYTHONPATH`. Do not install both `cosmos_framework` implementations into the same import namespace.

RoboTwin uses a separate Python 3.10 environment. The pinned setup tool is `policy/source/recipes/simulation/robotwin/common/setup.py`, with a separate requirements file in that directory. The upstream simulator revision is `c3ddfa8b97d5519efa828b075999bd0006778e5e`; deployed task source is included under `simulation/RoboTwin`. Install assets separately and use the included deployed source when reproducing this release. Choose a new simulator directory instead of overwriting another experiment's installation.

The public source has been checked on CPU. A fresh Linux GPU installation and a new full benchmark are not claimed by source publication.

## Model downloads

```bash
hf download DAGroup-PKU/ViGAR --local-dir /path/to/vigar-weights
python scripts/verify_weights.py /path/to/vigar-weights
```

The model repository contains native DCP shards and `.metadata`; downloading a subset of shards is insufficient. The policy folder is `robotwin/skipping_colorjitter/iter_000050000`; the planner folder is `i2i/roi/iter_000070000`.

Set `GOALWAM_POLICY_CHECKPOINT` and `GOALWAM_I2I_CHECKPOINT` accordingly. For inference, use the policy bundle's `vae/Wan2.2_VAE.pth` and `text_tokenizer` for `WAN_VAE_PATH` and `QWEN_TOKENIZER_PATH`. The planner checkpoint can be `BASE_CHECKPOINT_PATH` when serving it; training from the foundation model requires that model separately.

The policy retains EMA tensor names and serving dtypes; the I2I planner loads regular weights. Neither artifact contains an optimizer or complete training-resume state.

## Training

Configure `.env` from `.env.example`. W&B credentials come from your environment or credential store; set your own entity/project. Launchers use existing GPUs and do not submit cloud jobs.

Policy training uses 16 GPUs, microbatch 16, global batch 256 and 50k optimizer updates, saving every 4k and at the terminal update. It retains native Cosmos optimization, EMA, masks and geometry. ColorJitter strengths are brightness 0.3, contrast 0.4 and saturation 0.5; one parameter draw applies to all valid views, frames and goals within a sample, preserving padding.

Look-ahead redirects the last 15% of a non-final stage to the next stage's goal and text together; it does not drop input samples. I2I look-ahead remains disabled.

The I2I lineage is normal training with selected 30k EMA, then 40k additional ROI updates with a fresh optimizer, then complete-state continuation on random500 to ROI step70k. The normal-stage wrapper stops at30k while preserving the original200k LR schedule. Head ROI windows are96×96, wrists48×48; target/background weight is4:1 with spatial-mean normalization. Source-image ROI is zero. Valid offscreen targets and explicitly allowed missing geometry retain global loss.

## Closed-loop evaluation

`prepare_eval.py` creates a fresh plan bound to the selected checkpoints. Both nodes must see the same output directory and assets. The plan contains 5000 fixed seed/instruction pairs across 50 tasks and clean/random splits. The manifest SHA256 is `183bba8e3211e6813e5f4c1e0e11093e8c384c3f7efe4db2cf82ee7be526b557`.

Eight I2I/policy pairs use a dynamic five-seed queue and two independent simulator lanes per pair. SAPIEN renders on GPU; physics remains on CPU. Every policy inference receives a goal generated from the same current observation, with matching source/current observation versions. There is no expert-goal or oracle-stage handoff.

LRU/eager CUDA graphs require a same-instance equivalence check; failure falls back to eager. The observation's legacy BGR convention and goal's RGB convention are intentional parts of the trained contract.
